"""Alternate x-encoders for ablation studies.

``model/pi_rnn.py`` is locked, so the ``transformer_backbone`` ablation cannot be a
flag that mutates it. Instead this module provides ``PIRNNObserverTransformer``, which
**subclasses** ``PIRNNObserver`` and swaps ONLY the P_trans-window encoder (GRU ->
Transformer encoder). Every other component — the context encoder, physics
prior/posterior branch, classifier, decoder seed, and both autoregressive forecast
decoders — is reused verbatim from the parent, and the ``forward`` output dict has the
**identical keys and shapes** so ``model/loss.py`` and ``model/train.py`` are unchanged.

Contract preservation:
  * ``h_final`` is dim ``gru_hidden`` (the Transformer pools to the same width the GRU did),
    so ``phys_state_refiner`` (which consumes ``[phys_state, h_final]``) is unchanged.
  * The physics PRIOR still grounds the encoder: instead of seeding the GRU layer-0 hidden
    state, it is injected as a prepended "physics token" (CLS-style) whose final
    representation becomes ``h_final``. This mirrors the GRU's physics-grounded ``h0``.
  * Output keys: ``logits, pred_detuning, pred_detuning_logvar, pred_P_trans,
    pred_P_trans_logvar, phys_state, phys_state_refined, h_final, ctx`` — same as the parent.

Guarded by ``tests/test_ablation_encoders.py`` (and ONLY by it — ``tests/test_model.py``
exercises ``PIRNNObserver``, not this subclass):
  * ``test_transformer_matches_observer_contract`` — output-dict KEY SET and per-key SHAPES
    match ``PIRNNObserver`` exactly. This is the drift guard for the copied tail below.
  * ``test_transformer_permutation_sensitive`` — the positional encoding is present AND
    effective: a time-shuffled window must materially change ``h_final``.
  * ``test_transformer_loss_backward`` — ``model/loss.py`` observer mode runs on this
    output dict and backprops into the Transformer stack.

NOTE: the forecast/classification "tail" of ``forward`` is copied from
``PIRNNObserver.forward`` (the parent provides no hook to override only the encoder).
``pi_rnn.py`` is final, so the copy is stable; if it ever changes, this file must track it.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from model.pi_rnn import ModelConfig, PIRNNObserver


class _SinusoidalPositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding (no learned params)."""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].size(1)])
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class PIRNNObserverTransformer(PIRNNObserver):
    """``PIRNNObserver`` with the GRU window-encoder replaced by a Transformer encoder.

    Drop-in for the observer in ``model/train.py`` via the ``observer_factory`` hook:
    the output-dict contract (keys + shapes) is preserved exactly.
    """

    def __init__(self, config: ModelConfig, n_head: int = 8, dim_feedforward: int | None = None):
        super().__init__(config)
        # Remove the GRU encoder; it is replaced by the Transformer stack below.
        del self.gru_encoder
        del self.h0_projector   # unused: physics prior enters via phys_token_proj, not h0

        d_model = config.gru_hidden
        if d_model % n_head != 0:
            # Fall back to the largest head count that divides d_model.
            n_head = max(h for h in (8, 4, 2, 1) if d_model % h == 0)
        ff = dim_feedforward if dim_feedforward is not None else 2 * d_model

        # Input projection: same per-step input as the GRU saw ([P_trans, ctx]) -> d_model.
        self.input_proj = nn.Linear(1 + config.context_proj_dim, d_model)
        # Physics PRIOR -> a prepended token (CLS-style) that grounds the sequence in physics.
        self.phys_token_proj = nn.Linear(config.phys_state_dim, d_model)
        self.pos_encoder = _SinusoidalPositionalEncoding(d_model, max_len=config.W + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.gru_layers, enable_nested_tensor=False)
        self.transformer_encoder.__doc__ = (
            "Encodes the P_trans window in parallel; the physics-token's final representation "
            "is the pooled h_final (dim gru_hidden), preserving the observer output contract."
        )

    def _encode(self, x: torch.Tensor, ctx: torch.Tensor, phys_state: torch.Tensor) -> torch.Tensor:
        """Return ``h_final`` of dim ``gru_hidden`` from the P_trans window + context + physics prior."""
        ctx_expanded = ctx.unsqueeze(1).expand(-1, self.config.W, -1)
        seq = self.input_proj(torch.cat([x, ctx_expanded], dim=-1))      # [B, W, d_model]
        phys_token = torch.tanh(self.phys_token_proj(phys_state)).unsqueeze(1)  # [B, 1, d_model]
        seq = torch.cat([phys_token, seq], dim=1)                        # [B, W+1, d_model]
        seq = self.pos_encoder(seq)
        encoded = self.transformer_encoder(seq)                          # [B, W+1, d_model]
        return encoded[:, 0, :]                                          # physics-token output == h_final

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> dict[str, torch.Tensor]:
        assert x.ndim == 3 and x.size(1) == self.config.W and x.size(2) == 1
        assert context.ndim == 2 and context.size(1) == self.config.n_context
        assert x.size(0) == context.size(0), f"batch mismatch: x={x.size(0)} context={context.size(0)}"

        # --- shared physics/context branch (parent submodules, verbatim) --- #
        ctx = self.context_encoder(context)
        phys_state_raw = self.physics_estimator(context)
        phys_state = self._bound_phys_state(phys_state_raw)

        # --- ONLY this block differs from PIRNNObserver: Transformer x-encoder --- #
        h_final = self._encode(x, ctx, phys_state)

        # --- shared "tail" (copied from PIRNNObserver.forward to preserve the contract) --- #
        refiner_in = torch.cat([phys_state, h_final], dim=-1)
        phys_state_refined = self._bound_phys_state(phys_state_raw + self.phys_state_refiner(refiner_in))

        fused_obs = torch.cat([h_final, phys_state_refined], dim=-1)
        logits = self.classifier(fused_obs)

        decoder_h0 = torch.tanh(self.decoder_seed(fused_obs))
        lvmin, lvmax = self.config.logvar_min, self.config.logvar_max

        h = decoder_h0
        inp = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        means, logvars = [], []
        for _ in range(self.config.H):
            h = self.detuning_gru_cell(inp, h)
            out = self.detuning_out(h)
            mean = out[:, 0:1]
            logvar = out[:, 1:2].clamp(lvmin, lvmax)
            means.append(mean)
            logvars.append(logvar)
            inp = mean.detach()
        pred_detuning = torch.cat(means, dim=1)
        pred_detuning_logvar = torch.cat(logvars, dim=1)

        h = decoder_h0
        inp = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        means, logvars = [], []
        for _ in range(self.config.H):
            h = self.ptrans_gru_cell(inp, h)
            out = self.ptrans_out(h)
            mean = out[:, 0:1]
            logvar = out[:, 1:2].clamp(lvmin, lvmax)
            means.append(mean)
            logvars.append(logvar)
            inp = mean.detach()
        pred_p_trans = torch.cat(means, dim=1)
        pred_p_trans_logvar = torch.cat(logvars, dim=1)

        return {
            "logits": logits,
            "pred_detuning": pred_detuning,
            "pred_detuning_logvar": pred_detuning_logvar,
            "pred_P_trans": pred_p_trans,
            "pred_P_trans_logvar": pred_p_trans_logvar,
            "phys_state": phys_state,
            "phys_state_refined": phys_state_refined,
            "h_final": h_final,
            "ctx": ctx,
        }
