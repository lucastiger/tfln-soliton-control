"""Contract / drift guards for ``model/ablation_encoders.py``.

``model/pi_rnn.py`` is LOCKED and ``PIRNNObserverTransformer`` copies the
forecast/classification tail of ``PIRNNObserver.forward`` verbatim (the parent exposes no
encoder-only hook to override). Nothing in the type system couples the two, so a change to
the locked parent's output dict would desynchronize the ablation arm *silently* — the run
would still train, just against a different contract than ``model/loss.py`` and
``model/train.py`` expect. These tests are that missing coupling: they fail if the parent's
output KEYS or SHAPES ever drift away from the Transformer variant's.

CPU-only, seeded, fast. No .h5 dataset, no GPU, no checkpoint.
"""

from __future__ import annotations

import torch

from model.ablation_encoders import PIRNNObserverTransformer
from model.loss import LossConfig, PhysicsInformedLoss
from model.pi_rnn import ModelConfig, PIRNNObserver

DEVICE = torch.device("cpu")


def tiny_transformer_config() -> ModelConfig:
    """Small but contract-faithful.

    ``W``/``H``/``n_context``/``n_states``/``phys_state_dim``/``context_proj_dim`` keep their
    real values so the shapes asserted below are the ones the trainer and loss actually see;
    only the widths that drive runtime (``gru_hidden`` -> d_model, ``gru_layers`` -> depth,
    ``decoder_hidden``) are shrunk.
    """
    return ModelConfig(
        W=200, H=50, gru_hidden=64, gru_layers=2, decoder_hidden=32, dropout=0.0,
    )


def make_inputs(mcfg: ModelConfig, batch_size: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.randn(batch_size, mcfg.W, 1),
        torch.randn(batch_size, mcfg.n_context),
    )


# --------------------------------------------------------------------------- #
# 1. Drift guard: output-dict key set + per-key shapes match the locked parent
# --------------------------------------------------------------------------- #
def test_transformer_matches_observer_contract():
    """The Transformer ablation's output dict must match ``PIRNNObserver``'s key-for-key
    and shape-for-shape.

    ``pi_rnn.py`` is locked, so if its output contract is ever changed, THIS test — not a
    silently-diverged training run — is what reports it.
    """
    torch.manual_seed(0)
    mcfg = tiny_transformer_config()
    x, context = make_inputs(mcfg)

    parent = PIRNNObserver(mcfg).eval()
    transformer = PIRNNObserverTransformer(mcfg).eval()

    with torch.no_grad():
        out_parent = parent(x, context)
        out_transformer = transformer(x, context)

    # (a) KEY SET parity — fails if the parent gains, loses, or renames an output key.
    assert set(out_parent) == set(out_transformer), (
        f"output key sets diverged: "
        f"missing_from_transformer={sorted(set(out_parent) - set(out_transformer))}, "
        f"extra_in_transformer={sorted(set(out_transformer) - set(out_parent))}"
    )

    # (b) per-key SHAPE parity — fails if the parent changes any output's dimensionality.
    for key in sorted(out_parent):
        assert out_parent[key].shape == out_transformer[key].shape, (
            f"shape drift on {key!r}: parent={tuple(out_parent[key].shape)} "
            f"transformer={tuple(out_transformer[key].shape)}"
        )

    # h_final's width is what the parent's phys_state_refiner ([phys_state, h_final])
    # consumes; if the Transformer pooled to a different width, the inherited refiner
    # would not accept its own input.
    assert out_transformer["h_final"].shape[1] == mcfg.gru_hidden


# --------------------------------------------------------------------------- #
# 2. Positional-encoding guard
# --------------------------------------------------------------------------- #
def test_transformer_permutation_sensitive():
    """A Transformer encoder is permutation-invariant over its input tokens.

    Without a positional encoding added to the P_trans token embeddings, shuffling the
    200-step window in time would leave ``h_final`` unchanged — the delta collapses to
    float noise (~1e-7) — and the ablation arm would be scientifically void: it would be
    comparing the GRU against a bag-of-samples model rather than against a sequence model.
    A material delta here is the evidence that time order is actually encoded.
    """
    torch.manual_seed(0)
    mcfg = tiny_transformer_config()
    x, context = make_inputs(mcfg)

    transformer = PIRNNObserverTransformer(mcfg).eval()
    perm = torch.randperm(mcfg.W)
    x_shuffled = x[:, perm, :]

    with torch.no_grad():
        h_ordered = transformer(x, context)["h_final"]
        h_shuffled = transformer(x_shuffled, context)["h_final"]

    delta = (h_ordered - h_shuffled).abs().max().item()
    assert delta > 1e-4, (
        f"h_final is (near-)invariant to a time shuffle (max|delta|={delta:.3e}); the "
        "positional encoding is missing or ineffective, so the encoder cannot see time order"
    )


# --------------------------------------------------------------------------- #
# 3. Loss integration: observer objective runs and backprops into the Transformer
# --------------------------------------------------------------------------- #
def test_transformer_loss_backward():
    """``model/loss.py`` observer mode must run on the Transformer's output and push
    gradient into the Transformer stack itself — not only into the inherited parent heads.
    """
    torch.manual_seed(0)
    mcfg = tiny_transformer_config()
    x, context = make_inputs(mcfg)
    batch_size = x.size(0)

    transformer = PIRNNObserverTransformer(mcfg)
    out = transformer(x, context)

    targets = {
        "state_label": torch.randint(0, mcfg.n_states, (batch_size,)),
        "future_detuning": torch.randn(batch_size, mcfg.H),
        "future_P_trans": torch.randn(batch_size, mcfg.H),
    }
    loss_cfg = LossConfig(
        Gamma_th=0.1,
        tau_th=5e-6,
        # test-local, arbitrary; NOT the runtime default. horizon_dt is resolved at runtime from the
        # dataset (snapshot_interval / fsr_hz). Do not treat this literal as a reference value.
        horizon_dt=5e-11,
        thermal_norm="empirical",
    )
    loss = PhysicsInformedLoss(loss_cfg)(out, targets, mode="observer")

    assert loss["total"].ndim == 0
    assert torch.isfinite(loss["total"])

    loss["total"].backward()

    encoder_grads = [
        p.grad for p in transformer.transformer_encoder.parameters() if p.grad is not None
    ]
    assert encoder_grads, "no Transformer-encoder parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in encoder_grads), (
        "Transformer-encoder gradients are not all finite"
    )
    assert any(g.abs().sum() > 0 for g in encoder_grads), (
        "Transformer-encoder gradients are all exactly zero"
    )
