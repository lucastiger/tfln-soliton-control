"""Physics-informed objective and regularization terms for the PI-RNN.

Consumes the output dicts produced by ``model/pi_rnn.py`` (``PIRNNObserver`` /
``PIRNNController``) verbatim. Implements:

  * ``PhysicsInformedLoss`` — supervised classification + heteroscedastic forecast
    (betaNLL) + a low-weight thermal self-consistency prior, with an asymmetric
    controller branch (mean-only detuning MSE + control-effort penalty + the
    action-state classification term that MPC ascends).
  * ``LossScaler`` — adaptive lambda balancing that holds the classification term
    fixed (FN-1) and never balances regularizers to task parity.

Design notes baked into this file (see the original review):

  * The energy-balance ``L_physics`` term is intentionally ABSENT. It reconstructed
    ``U_int`` from ``pred_P_trans`` via an assumed ``P_trans = kappa_c * U_int``
    steady-state relation; no such relation or per-timestep ``U_int`` ground truth
    exists, so the term is gone.
  * ``forward`` takes NO ``context`` argument. The only physics term left
    (``L_thermal``) uses ``phys_state_refined[:, 3]`` (= ``U_int_est``) and config
    constants sourced from ``sin_params.yaml`` — never from ``context``.
  * Forecast losses are Gaussian betaNLL (Seitzer et al. 2022), not MSE: the
    detuning and P_trans heads are heteroscedastic (``*_logvar``). The P_trans
    reconstruction loss is observer-only (controller has no P_trans head).
  * Observer and controller are asymmetric: controller has no P_trans loss; its
    detuning loss is MSE (its forecast is mean-only); it adds a control-effort term.

Callers should log component values via ``component.item()``; only ``total`` is
weighted and is the only tensor used for ``.backward()``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class LossConfig:
    # Observer term weights
    lambda_class_obs: float = 1.0
    lambda_detune_obs: float = 1.0      # betaNLL
    lambda_pred_obs: float = 1.0        # betaNLL, observer only
    lambda_thermal_obs: float = 0.05    # self-consistency regularizer (low)

    # Controller term weights
    lambda_class_ctrl: float = 1.0      # FIXED reference; never auto-shrunk (FN-1)
    lambda_detune_ctrl: float = 1.0     # MSE (controller forecast is mean-only)
    lambda_effort_ctrl: float = 0.01    # |act_detuning_correction|^2
    lambda_thermal_ctrl: float = 0.0    # default OFF; see caveat in thermal term

    # Physics constants (from sin_params.yaml; required for thermal term)
    Gamma_th: float = 0.0
    tau_th: float = 1.0
    horizon_dt: float = 1e-7            # seconds per horizon step (100 ns nominal)

    # NLL controls
    beta_nll: float = 0.5              # 0 = plain NLL; 0.5 robust default (FN-3)
    nll_logvar_floor: float = -10.0    # mirror model clamp

    # Thermal-term normalization: "empirical" (scale-free, default) or "physical"
    thermal_norm: str = "empirical"

    eps: float = 1e-6
    U_INT_IDX: int = 3                 # phys_state index of U_int_est


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #
def beta_gaussian_nll(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    floor: float,
    beta: float,
) -> torch.Tensor:
    """Per-element heteroscedastic NLL with Seitzer betaNLL weighting (stop-grad on the weight).

    ``beta=0`` -> standard Gaussian NLL; ``beta=1`` -> mean-gradient ~ MSE while still
    learning variance. The ``var.detach()`` weight is a stop-grad (FN-3): it prevents the
    model from minimizing loss on hard/late steps by inflating ``logvar`` (variance
    attenuation), which would let the forecast tail sit at a fixed point with large
    uncertainty. Returns ``[B, H]``; the caller reduces.
    """
    logvar = logvar.clamp_min(floor)
    var = torch.exp(logvar)
    nll = 0.5 * (logvar + (target - mean) ** 2 / var)          # [B, H]
    if beta > 0:
        nll = (var.detach() ** beta) * nll
    return nll                                                  # return [B,H]; reduce by caller


def masked_scalar_mean(per_sample: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Masked batch mean over a ``[B]`` per-sample vector.

    ``mask is None`` -> plain mean. Returns a graph-safe zero (not NaN) when no sample
    is valid.
    """
    if mask is None:
        return per_sample.mean()
    m = mask.to(per_sample.dtype)
    denom = m.sum()
    if float(denom) == 0.0:
        return per_sample.sum() * 0.0
    return (per_sample * m).sum() / denom


def reduce_horizon_then_mask(per_step: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Reduce ``[B, H] -> [B]`` with a UNIFORM mean over H (FN-3: no decaying / tail
    down-weighting), then take a masked batch mean.

    ``mask is None`` -> plain mean over B. When ``mask`` is given but selects nothing,
    return a graph-safe zero (``per_step.sum() * 0.0``), not NaN.
    """
    per_sample = per_step.mean(dim=1)                          # uniform over H (FN-3)
    if mask is None:
        return per_sample.mean()
    m = mask.to(per_sample.dtype)
    denom = m.sum()
    if float(denom) == 0.0:
        return per_step.sum() * 0.0
    return (per_sample * m).sum() / denom


def masked_var(target_BH: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Variance over the valid elements only (flatten the valid rows).

    Used as the scale normalizer for the controller detuning MSE. Returns a graph-safe
    zero when no sample is valid.
    """
    if mask is None:
        valid = target_BH.reshape(-1)
    else:
        m = mask.to(torch.bool)
        if int(m.sum()) == 0:
            return target_BH.sum() * 0.0
        valid = target_BH[m].reshape(-1)
    if valid.numel() == 0:
        return target_BH.sum() * 0.0
    return valid.var(unbiased=False)


# --------------------------------------------------------------------------- #
# Loss module
# --------------------------------------------------------------------------- #
class PhysicsInformedLoss(nn.Module):
    """Physics-informed loss for the PI-RNN observer and controller. No learnable params."""

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg

    # ----- physics term ---------------------------------------------------- #
    def _thermal(self, detuning_BH: torch.Tensor, phys_state_refined_B4: torch.Tensor) -> torch.Tensor:
        """Thermal-relaxation shape-consistency prior on the detuning trajectory.

        Rationale (REQUIRED): ``pred_detuning`` and ``phys_state_refined`` are NORMALIZED
        latent quantities, so the physical normalizer ``(Gamma_th/tau_th)**2`` is not
        dimensionally matched to ``d(delta_omega)/dt`` and would make the term
        scale-arbitrary. The default ``empirical`` normalizer makes this a dimensionless
        shape-consistency prior: it penalizes detuning drift whose rate is inconsistent
        with a relaxation set by ``U_int_est``, up to the learned normalization. Switch to
        ``physical`` only once model outputs are calibrated to physical units. This is a
        low-weight self-consistency prior, never validated against a measured
        Delta_T / U_int trajectory.
        """
        cfg = self.cfg
        dwdt = (detuning_BH[:, 1:] - detuning_BH[:, :-1]) / cfg.horizon_dt          # [B, H-1]
        target_rate = (
            -cfg.Gamma_th * phys_state_refined_B4[:, cfg.U_INT_IDX] / cfg.tau_th
        ).unsqueeze(1)                                                              # [B, 1]
        resid = dwdt - target_rate                                                 # [B, H-1]
        if cfg.thermal_norm == "empirical":
            scale2 = dwdt.detach().pow(2).mean() + cfg.eps   # scale-free: normalize by RMS(dwdt)^2
        else:  # "physical"
            scale2 = (cfg.Gamma_th / cfg.tau_th) ** 2 + cfg.eps
        return (resid ** 2).mean() / scale2

    # ----- observer terms -------------------------------------------------- #
    def _observer_terms(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        class_weights: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        # Classification.
        class_obs = F.cross_entropy(predictions["logits"], targets["state_label"], weight=class_weights)
        # Detuning forecast: heteroscedastic betaNLL, uniform horizon reduction.
        detune_nll = beta_gaussian_nll(
            predictions["pred_detuning"], predictions["pred_detuning_logvar"],
            targets["future_detuning"], cfg.nll_logvar_floor, cfg.beta_nll,
        )
        detune_obs = reduce_horizon_then_mask(detune_nll, None)
        # P_trans reconstruction: betaNLL, observer ONLY (no controller P_trans head).
        pred_nll = beta_gaussian_nll(
            predictions["pred_P_trans"], predictions["pred_P_trans_logvar"],
            targets["future_P_trans"], cfg.nll_logvar_floor, cfg.beta_nll,
        )
        pred_obs = reduce_horizon_then_mask(pred_nll, None)
        # Thermal self-consistency prior on the no-intervention detuning forecast.
        thermal_obs = self._thermal(predictions["pred_detuning"], predictions["phys_state_refined"])
        return {
            "class_obs": class_obs,
            "detune_obs": detune_obs,
            "pred_obs": pred_obs,
            "thermal_obs": thermal_obs,
        }

    # ----- controller terms ------------------------------------------------ #
    def _controller_terms(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        class_weights_ctrl: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        # FN-1: act_logits is the PRIMARY output (the only one with a strong, live
        # delta_cmd gradient through fused_act + the unrolled decoder; MPC ascends it).
        # Its weight (lambda_class_ctrl) is the fixed reference and must never be shrunk
        # below the other controller terms by the scaler. No detach on act_logits.
        ce_per = F.cross_entropy(
            predictions["act_logits"], targets["target_state_label"],
            weight=class_weights_ctrl, reduction="none",
        )                                                          # [B]
        class_ctrl = masked_scalar_mean(ce_per, mask)
        # Detuning: MSE (controller forecast is mean-only), normalized by target variance.
        fut = targets["future_detuning_under_action"]
        sq = (predictions["act_pred_detuning"] - fut) ** 2         # [B, H], no detach on act path
        detune_ctrl = reduce_horizon_then_mask(sq, mask) / (masked_var(fut, mask) + cfg.eps)
        # Control effort: penalize action magnitude (minimal-actuation framing). No detach.
        effort_ctrl = reduce_horizon_then_mask(predictions["act_detuning_correction"] ** 2, mask)

        out = {"class_ctrl": class_ctrl, "detune_ctrl": detune_ctrl, "effort_ctrl": effort_ctrl}

        if cfg.lambda_thermal_ctrl > 0:
            # CAVEAT: the relaxation target has no actuator term, so this CANNOT validate
            # the action's effect on detuning — it is an inductive bias only, NEVER a
            # constraint on delta_cmd. Default off. Apply the mask by row-selection so the
            # empirical RMS normalizer is computed over valid rows only.
            adp = predictions["act_pred_detuning"]
            psr = predictions["phys_state_refined"]
            if mask is None:
                out["thermal_ctrl"] = self._thermal(adp, psr)
            else:
                m = mask.to(torch.bool)
                if int(m.sum()) == 0:
                    out["thermal_ctrl"] = adp.sum() * 0.0          # graph-safe zero
                else:
                    out["thermal_ctrl"] = self._thermal(adp[m], psr[m])
        return out

    # ----- forward --------------------------------------------------------- #
    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        mode: Literal["observer", "controller"],
        observer_frozen: bool = False,
        class_weights: torch.Tensor | None = None,
        class_weights_ctrl: torch.Tensor | None = None,
        valid_transition_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        out: dict[str, torch.Tensor] = {}

        if mode == "observer":
            obs = self._observer_terms(predictions, targets, class_weights)
            out.update(obs)
            out["total"] = (
                cfg.lambda_class_obs * obs["class_obs"]
                + cfg.lambda_detune_obs * obs["detune_obs"]
                + cfg.lambda_pred_obs * obs["pred_obs"]
                + cfg.lambda_thermal_obs * obs["thermal_obs"]
            )
            return out

        if mode == "controller":
            ctrl = self._controller_terms(predictions, targets, class_weights_ctrl, valid_transition_mask)
            out.update(ctrl)
            total = (
                cfg.lambda_class_ctrl * ctrl["class_ctrl"]
                + cfg.lambda_detune_ctrl * ctrl["detune_ctrl"]
                + cfg.lambda_effort_ctrl * ctrl["effort_ctrl"]
            )
            if cfg.lambda_thermal_ctrl > 0:
                total = total + cfg.lambda_thermal_ctrl * ctrl["thermal_ctrl"]

            # Frozen pass-through: observer keys carry zero gradient (computed under no_grad
            # inside the model). Do NOT compute any *_obs term. Only when observer_frozen is
            # False (explicit joint fine-tune) do we add the observer terms from the
            # pass-through outputs at their _obs weights.
            if not observer_frozen:
                obs = self._observer_terms(predictions, targets, class_weights)
                out.update(obs)
                total = total + (
                    cfg.lambda_class_obs * obs["class_obs"]
                    + cfg.lambda_detune_obs * obs["detune_obs"]
                    + cfg.lambda_pred_obs * obs["pred_obs"]
                    + cfg.lambda_thermal_obs * obs["thermal_obs"]
                )
            out["total"] = total
            return out

        raise ValueError(f"mode must be 'observer' or 'controller', got {mode!r}")


# --------------------------------------------------------------------------- #
# Adaptive lambda balancing
# --------------------------------------------------------------------------- #
class LossScaler:
    """Adaptive lambda balancing that respects FN-1 and does not force regularizers to parity.

    The reference term's lambda is held FIXED. Each balanced term's lambda is rescaled so
    its weighted contribution equalizes against the reference. Regularizers (``thermal_*``,
    ``effort_ctrl``) are NEVER passed in ``balanced_map`` — forcing a prior to task-parity
    would over-regularize, so they keep their config weights.
    """

    _VANISH_TOL = 1e-12

    def __init__(self, cfg: LossConfig, reference_key: str, balanced_map: dict[str, str],
                 period: int = 5, clamp: tuple[float, float] = (1e-3, 1e3)):
        # reference_key: e.g. "class_obs" or "class_ctrl" -> its lambda is held FIXED.
        # balanced_map: dict mapping component key -> LossConfig lambda-field name,
        #   e.g. {"detune_obs": "lambda_detune_obs", "pred_obs": "lambda_pred_obs"}.
        #   ONLY primary-task terms go here.
        self.cfg = cfg
        self.reference_key = reference_key
        self.reference_field = "lambda_" + reference_key       # uniform LossConfig naming
        if not hasattr(cfg, self.reference_field):
            raise ValueError(f"reference_key {reference_key!r} has no lambda field {self.reference_field!r}")
        self.balanced_map = dict(balanced_map)
        self.period = period
        self.clamp = clamp
        self._reset()

    def _reset(self) -> None:
        keys = [self.reference_key, *self.balanced_map.keys()]
        self._sums = {k: 0.0 for k in keys}
        self._counts = {k: 0 for k in keys}

    def update(self, loss_dict: dict[str, torch.Tensor]) -> None:
        """Accumulate raw (detached, float) values for the reference and every balanced key."""
        for k in self._sums:
            if k in loss_dict:
                v = loss_dict[k]
                self._sums[k] += float(v.detach()) if torch.is_tensor(v) else float(v)
                self._counts[k] += 1

    def _mean(self, key: str) -> float:
        c = self._counts[key]
        return self._sums[key] / c if c > 0 else 0.0

    def maybe_rescale(self, epoch: int) -> LossConfig:
        """Every ``period`` epochs (skip epoch 0), set ``lambda_k = lambda_ref * mean(ref) /
        mean(comp_k)`` so weighted contributions equalize against the fixed reference."""
        if epoch == 0 or self.period <= 0 or epoch % self.period != 0:
            return self.cfg
        ref_mean = self._mean(self.reference_key)
        ref_lambda = getattr(self.cfg, self.reference_field)
        lo, hi = self.clamp
        for key, field in self.balanced_map.items():
            comp_mean = self._mean(key)
            if comp_mean < self._VANISH_TOL:
                # A vanishing component signals a broken graph or dead target, not a scale
                # mismatch; do not upscale it.
                warnings.warn(
                    f"LossScaler: component {key!r} mean ~ 0 ({comp_mean:.3e}); "
                    f"skipping rescale of {field!r} (possible broken graph or dead target).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            new_lambda = ref_lambda * ref_mean / comp_mean
            new_lambda = float(min(max(new_lambda, lo), hi))
            setattr(self.cfg, field, new_lambda)
        self._reset()
        return self.cfg


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import math

    from model.pi_rnn import ModelConfig, PIRNNController, PIRNNObserver

    torch.manual_seed(0)
    mcfg = ModelConfig()
    B, H = 8, mcfg.H

    def synth_observer_targets() -> dict[str, torch.Tensor]:
        return {
            "state_label": torch.randint(0, mcfg.n_states, (B,)),
            "future_detuning": torch.randn(B, H),
            "future_P_trans": torch.randn(B, H),
        }

    def synth_controller_targets() -> dict[str, torch.Tensor]:
        return {
            "target_state_label": torch.randint(0, mcfg.n_states, (B,)),
            "future_detuning_under_action": torch.randn(B, H),
        }

    x = torch.randn(B, mcfg.W, 1)
    context = torch.randn(B, mcfg.n_context)
    target_state = torch.randint(0, mcfg.n_states, (B,))

    OBS_KEYS = {"total", "class_obs", "detune_obs", "pred_obs", "thermal_obs"}
    CTRL_KEYS = {"total", "class_ctrl", "detune_ctrl", "effort_ctrl"}

    loss = PhysicsInformedLoss(LossConfig())

    # ---- 1. Observer mode ------------------------------------------------- #
    observer = PIRNNObserver(mcfg)
    obs_pred = observer(x, context)
    obs_t = synth_observer_targets()
    out = loss(obs_pred, obs_t, mode="observer")
    assert set(out.keys()) == OBS_KEYS, set(out.keys())
    assert all(torch.isfinite(v).all() for v in out.values()), "observer terms not finite"
    out["total"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in observer.parameters()), \
        "no observer param received gradient"
    print("[1] observer mode: keys + finiteness + grad OK")

    # ---- 2. Controller frozen --------------------------------------------- #
    observer2 = PIRNNObserver(mcfg)
    controller = PIRNNController(mcfg, observer2, freeze_observer=True)
    delta_cmd = torch.randn(B, 1, requires_grad=True)
    ctrl_pred = controller(x, context, delta_cmd, target_state)
    ctrl_t = synth_controller_targets()
    out = loss(ctrl_pred, ctrl_t, mode="controller", observer_frozen=True)
    assert set(out.keys()) == CTRL_KEYS, set(out.keys())
    assert not any(k.endswith("_obs") for k in out), "frozen controller leaked *_obs keys"
    out["total"].backward()
    head_params = [p for n, p in controller.named_parameters() if not n.startswith("observer.")]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head_params), \
        "no controller-head param received gradient"
    assert all(p.grad is None for p in controller.observer.parameters()), \
        "frozen observer received gradient"
    print("[2] controller frozen: ctrl-only keys + head grad + observer grad None OK")

    # ---- 3. Controller unfrozen (joint fine-tune) ------------------------- #
    observer3 = PIRNNObserver(mcfg)
    controller3 = PIRNNController(mcfg, observer3, freeze_observer=False)
    delta_cmd3 = torch.randn(B, 1, requires_grad=True)
    ctrl_pred3 = controller3(x, context, delta_cmd3, target_state)
    joint_t = {**synth_observer_targets(), **synth_controller_targets()}
    out = loss(ctrl_pred3, joint_t, mode="controller", observer_frozen=False)
    assert set(out.keys()) == OBS_KEYS | CTRL_KEYS, set(out.keys())
    out["total"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in controller3.observer.parameters()), \
        "unfrozen observer received no gradient"
    print("[3] controller unfrozen: union keys + observer grad OK")

    # ---- 4. Masking ------------------------------------------------------- #
    observer4 = PIRNNObserver(mcfg)
    controller4 = PIRNNController(mcfg, observer4, freeze_observer=True)
    dc4 = torch.randn(B, 1, requires_grad=True)
    pred4 = controller4(x, context, dc4, target_state)
    ctrl_t4 = synth_controller_targets()
    some_false = torch.ones(B, dtype=torch.bool)
    some_false[:3] = False
    out_partial = loss(pred4, ctrl_t4, mode="controller", observer_frozen=True,
                       valid_transition_mask=some_false)
    assert all(torch.isfinite(v).all() for v in out_partial.values()), "partial mask -> non-finite"
    all_false = torch.zeros(B, dtype=torch.bool)
    pred4b = controller4(x, context, torch.randn(B, 1, requires_grad=True), target_state)
    out_zero = loss(pred4b, ctrl_t4, mode="controller", observer_frozen=True,
                    valid_transition_mask=all_false)
    for k in ("class_ctrl", "detune_ctrl", "effort_ctrl"):
        assert out_zero[k].item() == 0.0, f"{k} not exactly 0 under all-False mask"
    out_zero["total"].backward()
    assert torch.isfinite(out_zero["total"]), "all-False total not finite"
    print("[4] masking: partial finite + all-False terms == 0 + NaN-free backward OK")

    # ---- 5. betaNLL correctness ------------------------------------------- #
    mean = torch.tensor([[0.5, -1.0]])
    logvar = torch.tensor([[0.0, 1.0]])
    target = torch.tensor([[1.0, 0.0]])
    floor = -10.0
    lv = logvar.clamp_min(floor)
    plain = 0.5 * (lv + (target - mean) ** 2 / torch.exp(lv))
    got0 = beta_gaussian_nll(mean, logvar, target, floor, beta=0.0)
    assert torch.allclose(got0, plain), "beta=0 != closed-form Gaussian NLL"
    got_half = beta_gaussian_nll(mean, logvar, target, floor, beta=0.5)
    expected_half = torch.exp(lv).pow(0.5) * plain
    assert torch.allclose(got_half, expected_half), "beta=0.5 != var^0.5 * plain_nll"
    print("[5] betaNLL: beta=0 closed-form + beta=0.5 weighting OK")

    # ---- 6. FN-3 uniform horizon weighting -------------------------------- #
    spike = torch.zeros(B, H)
    spike[:, -1] = 1.0
    reduced = reduce_horizon_then_mask(spike, None)
    assert abs(reduced.item() - 1.0 / H) < 1e-6, f"horizon reduction not uniform: {reduced.item()} vs {1.0/H}"
    print(f"[6] FN-3: last-step spike reduces to 1/H ({reduced.item():.6f}) -> no tail down-weighting OK")

    # ---- 7. Gradient guard: loss -> act terms -> delta_cmd ---------------- #
    observer7 = PIRNNObserver(mcfg)
    controller7 = PIRNNController(mcfg, observer7, freeze_observer=True)
    dc7 = torch.randn(B, 1)
    dc7.requires_grad_(True)
    pred7 = controller7(x, context, dc7, target_state)
    out7 = loss(pred7, synth_controller_targets(), mode="controller", observer_frozen=True)
    out7["total"].backward()
    assert dc7.grad is not None and dc7.grad.norm().item() > 0.0, "delta_cmd gradient path broken"
    print(f"[7] gradient guard: delta_cmd.grad.norm = {dc7.grad.norm().item():.4e} (nonzero) OK")

    # ---- 8. Thermal term -------------------------------------------------- #
    det = torch.randn(B, H)
    psr = torch.rand(B, 4)
    finite_cfg = LossConfig(Gamma_th=0.1, tau_th=5e-6, horizon_dt=1e-7)
    for norm in ("empirical", "physical"):
        finite_cfg.thermal_norm = norm
        val = PhysicsInformedLoss(finite_cfg)._thermal(det, psr)
        assert torch.isfinite(val), f"_thermal not finite for {norm}"
    # Scale-freeness: empirical invariant to a global scale of detuning; physical is not.
    # Use Gamma_th=0 so the relaxation target is 0 and resid == dwdt exactly.
    sf_cfg = LossConfig(Gamma_th=0.0, tau_th=1.0, horizon_dt=1.0, thermal_norm="empirical")
    emp = PhysicsInformedLoss(sf_cfg)
    e1 = emp._thermal(det, psr)
    e10 = emp._thermal(det * 10.0, psr)
    assert torch.allclose(e1, e10, atol=1e-5), f"empirical not scale-free: {e1.item()} vs {e10.item()}"
    ph_cfg = LossConfig(Gamma_th=0.0, tau_th=1.0, horizon_dt=1.0, thermal_norm="physical")
    phys = PhysicsInformedLoss(ph_cfg)
    p1 = phys._thermal(det, psr)
    p10 = phys._thermal(det * 10.0, psr)
    assert not torch.allclose(p1, p10, atol=1e-5), "physical unexpectedly scale-free"
    print(f"[8] thermal: finite both norms + empirical scale-free ({e1.item():.6f}) + physical not OK")

    # ---- 9. LossScaler ---------------------------------------------------- #
    cfg9 = LossConfig()
    scaler = LossScaler(cfg9, reference_key="class_obs",
                        balanced_map={"detune_obs": "lambda_detune_obs", "pred_obs": "lambda_pred_obs"},
                        period=5)
    for _ in range(4):
        scaler.update({
            "class_obs": torch.tensor(1.0),
            "detune_obs": torch.tensor(10.0),   # 10x the reference
            "pred_obs": torch.tensor(1.0),
        })
    cfg9 = scaler.maybe_rescale(epoch=5)
    assert math.isclose(cfg9.lambda_detune_obs, 0.1, rel_tol=1e-6), cfg9.lambda_detune_obs
    assert cfg9.lambda_class_obs == 1.0, "reference lambda mutated"
    assert cfg9.lambda_thermal_obs == 0.05, "regularizer lambda mutated"
    assert math.isclose(cfg9.lambda_pred_obs, 1.0, rel_tol=1e-6), cfg9.lambda_pred_obs
    # Vanishing component -> warn + no change.
    cfg9b = LossConfig()
    scaler2 = LossScaler(cfg9b, reference_key="class_obs",
                         balanced_map={"detune_obs": "lambda_detune_obs"}, period=5)
    for _ in range(4):
        scaler2.update({"class_obs": torch.tensor(1.0), "detune_obs": torch.tensor(1e-15)})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg9b = scaler2.maybe_rescale(epoch=5)
    assert any("detune_obs" in str(wi.message) for wi in w), "expected vanishing-component warning"
    assert cfg9b.lambda_detune_obs == 1.0, "vanishing component lambda mutated"
    print("[9] LossScaler: ~10x rescale + reference/regularizer fixed + vanishing-component warn OK")

    print("ALL LOSS CHECKS PASSED")
