"""Diagnostic for the thermal self-consistency term (``thermal_obs``) in ``model/loss.py``.

WHY THIS EXISTS
---------------
``L_physics`` (energy balance) was removed as unimplementable, so ``thermal_obs`` is the
only physics term left in the observer objective. This script answers, empirically,
whether that term still behaves like a *relaxation-rate constraint* at the data-derived
``horizon_dt``, or whether it has collapsed into a *scale-free flatness prior* on the
detuning forecast.

WHAT IT MEASURES (see ``diagnose_batch``)
-----------------------------------------
The implemented residual (``PhysicsInformedLoss._thermal``) is, verbatim::

    dwdt        = (detuning[:, 1:] - detuning[:, :-1]) / horizon_dt        # [B, H-1]
    target_rate = (-Gamma_th * phys_state_refined[:, 3] / tau_th)[:, None] # [B, 1]
    resid       = dwdt - target_rate
    scale2      = dwdt.detach().pow(2).mean() + eps      # thermal_norm == "empirical"
                | (Gamma_th / tau_th) ** 2 + eps         # thermal_norm == "physical"
    thermal     = (resid ** 2).mean() / scale2

For ``thermal_norm="empirical"`` this is *algebraically identical* (multiply numerator
and denominator by ``horizon_dt**2``) to::

    mean((diff - target_rate * horizon_dt) ** 2) / (mean(diff ** 2) + eps * horizon_dt ** 2)

with ``diff = detuning[:, 1:] - detuning[:, :-1]``, so ``horizon_dt`` enters ONLY through
the product ``target_rate * horizon_dt`` (the ``eps * horizon_dt**2`` term is numerically
irrelevant at any ``horizon_dt`` below ~1e-3). If ``|target_rate * horizon_dt| << |diff|``
the residual degenerates to ``mean(diff**2) / mean(diff**2).detach()`` -- a dimensionless
flatness prior that pushes the detuning forecast toward a constant, carrying no
information about the thermal relaxation rate.

That equivalence does NOT hold for ``thermal_norm="physical"``: there the denominator has
no ``horizon_dt``, so the term scales as ``1 / horizon_dt**2`` overall.

Reported per batch:
  (a) distribution of ``diff`` (the finite difference actually used): mean|.|, median,
      p5/p95, std
  (b) distribution of ``target_rate * horizon_dt`` over the same batch
  (c) the ratio ``mean|target_rate * horizon_dt| / mean|diff|`` -- the core number
  (d) ``thermal_obs`` itself (cross-checked against ``PhysicsInformedLoss._thermal``)
  (e) ``||d thermal_obs / d params||_2`` vs ``||d class_obs / d params||_2`` on the SAME
      batch and SAME parameters, raw and lambda-weighted, plus a decomposition of the
      thermal gradient into its flatness part (the ``Gamma_th = 0`` limit) and the
      constraint-carrying remainder.

This module NEVER imports-and-patches, mutates, or shadows ``model/loss.py``. It
reconstructs the residual independently from model outputs and calls the real ``_thermal``
read-only as a self-consistency check.

CLI
---
    python -m analysis.thermal_prior_diagnostic                       # synthetic batch, fresh model
    python -m analysis.thermal_prior_diagnostic --h5 data/synthetic/dataset.h5
    python -m analysis.thermal_prior_diagnostic --ckpt runs/x/best_model.pt
    python -m analysis.thermal_prior_diagnostic --json-out out.json --markdown-out out.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.loss import LossConfig, PhysicsInformedLoss  # noqa: E402
from model.pi_rnn import ModelConfig, PIRNNObserver  # noqa: E402
from model.train import resolve_horizon_dt  # noqa: E402
from simulator.lle_solver import _load_config  # noqa: E402

DEFAULT_SWEEP: tuple[float, ...] = (5e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6)
DEFAULT_PHYS_CONFIG = "config/sin_params.yaml"
DEFAULT_H5 = "data/synthetic/dataset.h5"


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #
@dataclass
class Stats:
    """Summary of a 1-D sample."""

    mean_abs: float
    median: float
    p5: float
    p95: float
    std: float
    n: int

    @classmethod
    def of(cls, t: torch.Tensor) -> "Stats":
        flat = t.detach().reshape(-1).double()
        q = torch.quantile(flat, torch.tensor([0.05, 0.50, 0.95], dtype=flat.dtype))
        return cls(
            mean_abs=float(flat.abs().mean()),
            median=float(q[1]),
            p5=float(q[0]),
            p95=float(q[2]),
            std=float(flat.std(unbiased=False)),
            n=int(flat.numel()),
        )


@dataclass
class ThermalDiagnostic:
    """Everything measured for one (batch, parameters, horizon_dt) triple."""

    horizon_dt: float
    thermal_norm: str
    Gamma_th: float
    tau_th: float
    eps: float

    diff: Stats                       # (a) finite difference used by the residual
    rms_diff: float                   # RMS(diff); sets the empirical normalizer's scale
    target_rate: Stats                # target_rate itself (1/s scale)
    target_rate_dt: Stats             # (b) target_rate * horizon_dt
    ratio_mean: float                 # (c) mean|target_rate*dt| / mean|diff|
    ratio_of_means: float             # mean|target_rate*dt| / mean|diff| (same, kept explicit)
    ratio_p95: float                  # p95|target_rate*dt| / p95|diff| (tail check)

    thermal_obs: float                # (d)
    thermal_obs_reference: float      # PhysicsInformedLoss._thermal on the same inputs
    thermal_obs_flatness_limit: float # same term with Gamma_th -> 0 (pure flatness prior)
    class_obs: float

    grad_norm_thermal: float          # (e)
    grad_norm_class: float
    grad_ratio: float                 # ||g_thermal|| / ||g_class||
    grad_norm_thermal_weighted: float # lambda_thermal_obs * ||g_thermal||
    grad_norm_class_weighted: float   # lambda_class_obs  * ||g_class||
    grad_ratio_weighted: float

    grad_norm_thermal_flatness: float # ||g|| of the Gamma_th -> 0 term
    grad_norm_constraint_part: float  # ||g_thermal - g_flatness||
    constraint_fraction: float        # ||g_thermal - g_flatness|| / ||g_flatness||
    # The empirical normalizer divides by RMS(diff)^2 while the residual is quadratic in
    # diff, so ||g_thermal|| scales as 1/RMS(diff): the flatness pressure GROWS as the
    # forecast flattens. This product is the scale-invariant part and is the number to
    # compare across models/seeds.
    grad_norm_thermal_times_rms_diff: float

    lambda_thermal_obs: float
    lambda_class_obs: float

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("diff", "target_rate", "target_rate_dt"):
            d[k] = asdict(getattr(self, k))
        return d


@dataclass
class Provenance:
    batch_source: str
    batch_size: int
    seed: int
    device: str
    model_source: str
    phys_config: str
    h5_path: str | None
    horizon_dt_resolved: float | None
    horizon_dt_provenance: str | None
    Gamma_th: float
    tau_th: float
    fsr_hz: float
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Inputs: physics constants, model, batch
# --------------------------------------------------------------------------- #
def load_physics(config_path: str | Path = DEFAULT_PHYS_CONFIG) -> dict[str, float]:
    """Read ``Gamma_th`` / ``tau_th_s`` / ``fsr_hz`` from the physics YAML (same source
    ``model/train.py::build_loss_config`` uses)."""
    phys = _load_config(config_path)
    return {
        "Gamma_th": float(phys["Gamma_th"]),
        "tau_th": float(phys["tau_th_s"]),
        "fsr_hz": float(phys["fsr_hz"]),
    }


def make_model(
    model_cfg: ModelConfig,
    ckpt: str | Path | None = None,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[PIRNNObserver, str]:
    """Fresh seeded observer, or one restored from a checkpoint. Returns (model, source)."""
    torch.manual_seed(seed)
    model = PIRNNObserver(model_cfg).to(device)
    source = f"fresh_seeded(seed={seed})"
    if ckpt is not None:
        blob = torch.load(str(ckpt), map_location=device, weights_only=False)
        state = blob
        if isinstance(blob, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in blob and isinstance(blob[key], dict):
                    state = blob[key]
                    break
        model.load_state_dict(state)
        source = f"checkpoint({ckpt})"
    # eval(): dropout off, so repeated forwards on the same batch are bitwise identical
    # and the gradient comparison is not contaminated by mask noise.
    model.eval()
    return model, source


def synthetic_batch(
    model_cfg: ModelConfig, batch_size: int, seed: int, device: str = "cpu"
) -> dict[str, torch.Tensor]:
    """Seeded synthetic batch with the shapes/dtypes the dataloader emits.

    ``x`` is a normalized P_trans window (dataloader divides by ``P0``, so O(1) positive),
    ``context`` is the 4-scalar operating point, ``label`` a class index.
    """
    g = torch.Generator(device="cpu").manual_seed(seed + 991)
    B, W, C = batch_size, model_cfg.W, model_cfg.n_context
    x = (1.0 + 0.3 * torch.randn(B, W, 1, generator=g)).abs()
    context = torch.randn(B, C, generator=g)
    label = torch.randint(0, model_cfg.n_states, (B,), generator=g)
    return {
        "x": x.to(device),
        "context": context.to(device),
        "label": label.to(device),
        "_source": "synthetic",  # type: ignore[dict-item]
    }


def dataset_batch(
    h5_path: str | Path,
    model_cfg: ModelConfig,
    batch_size: int,
    seed: int,
    config_path: str | Path = DEFAULT_PHYS_CONFIG,
    split: str = "train",
    stride: int = 10,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """One deterministic batch drawn from the real dataset (only used when the h5 exists)."""
    from data.dataloader import SolitonDataset  # local import: h5 path only

    ds = SolitonDataset(
        h5_path=h5_path,
        split=split,
        W=model_cfg.W,
        H=model_cfg.H,
        stride=stride,
        config_path=config_path,
    )
    g = torch.Generator(device="cpu").manual_seed(seed + 991)
    n = len(ds)
    take = min(batch_size, n)
    idx = torch.randperm(n, generator=g)[:take].tolist()
    items = [ds[i] for i in idx]
    return {
        "x": torch.stack([it["x"] for it in items]).to(device),
        "context": torch.stack([it["context"] for it in items]).to(device),
        "label": torch.stack([it["label"] for it in items]).to(device),
        "_source": f"dataset({h5_path}, split={split}, n_windows={n})",  # type: ignore[dict-item]
    }


# --------------------------------------------------------------------------- #
# Core measurement
# --------------------------------------------------------------------------- #
def _flat_grad_norm(
    scalar: torch.Tensor, params: list[torch.nn.Parameter]
) -> tuple[float, torch.Tensor]:
    """``(||dscalar/dparams||_2, flattened gradient)``; unused params contribute zeros."""
    grads = torch.autograd.grad(
        scalar, params, retain_graph=True, create_graph=False, allow_unused=True
    )
    parts = [
        (torch.zeros_like(p) if g is None else g).reshape(-1).double()
        for p, g in zip(params, grads)
    ]
    flat = torch.cat(parts)
    return float(flat.norm()), flat


def diagnose_batch(
    model: PIRNNObserver,
    batch: dict[str, torch.Tensor],
    loss_cfg: LossConfig,
) -> ThermalDiagnostic:
    """Reconstruct every component of ``thermal_obs`` for one batch at one ``horizon_dt``.

    The reconstruction mirrors ``PhysicsInformedLoss._thermal`` line for line; the real
    ``_thermal`` is then called on the identical tensors and the two values are compared,
    so a future edit to the loss that breaks this mirror shows up as a mismatch rather
    than as a silently wrong diagnostic.
    """
    params = [p for p in model.parameters() if p.requires_grad]

    out = model(batch["x"], batch["context"])
    detuning = out["pred_detuning"]                       # [B, H]
    psr = out["phys_state_refined"]                       # [B, 4]

    # ---- (a) the finite difference actually used ------------------------------------
    diff = detuning[:, 1:] - detuning[:, :-1]             # [B, H-1]
    dwdt = diff / loss_cfg.horizon_dt

    # ---- (b) the relaxation target and its horizon-weighted form ---------------------
    target_rate = (
        -loss_cfg.Gamma_th * psr[:, loss_cfg.U_INT_IDX] / loss_cfg.tau_th
    ).unsqueeze(1)                                        # [B, 1]
    target_rate_dt = target_rate * loss_cfg.horizon_dt

    # ---- (d) the residual, reconstructed -------------------------------------------
    resid = dwdt - target_rate
    if loss_cfg.thermal_norm == "empirical":
        scale2 = dwdt.detach().pow(2).mean() + loss_cfg.eps
    elif loss_cfg.thermal_norm == "physical":
        scale2 = torch.as_tensor(
            (loss_cfg.Gamma_th / loss_cfg.tau_th) ** 2 + loss_cfg.eps,
            dtype=dwdt.dtype, device=dwdt.device,
        )
    else:
        raise ValueError(f"unknown thermal_norm {loss_cfg.thermal_norm!r}")
    thermal = (resid ** 2).mean() / scale2

    # Read-only cross-check against the implemented term.
    thermal_ref = PhysicsInformedLoss(loss_cfg)._thermal(detuning, psr)

    # Flatness limit: identical term with the relaxation target switched off. The gap
    # between this and `thermal` is *all* of the term's physics content.
    resid_flat = dwdt
    thermal_flat = (resid_flat ** 2).mean() / scale2

    class_obs = torch.nn.functional.cross_entropy(out["logits"], batch["label"])

    # ---- (e) gradient norms on the same graph ---------------------------------------
    gn_thermal, g_thermal = _flat_grad_norm(thermal, params)
    gn_flat, g_flat = _flat_grad_norm(thermal_flat, params)
    gn_class, _ = _flat_grad_norm(class_obs, params)
    gn_constraint = float((g_thermal - g_flat).norm())

    diff_s = Stats.of(diff)
    rms_diff = float(diff.detach().double().pow(2).mean().sqrt())
    trd_s = Stats.of(target_rate_dt)
    ratio = trd_s.mean_abs / diff_s.mean_abs if diff_s.mean_abs > 0 else math.inf
    p95_diff = max(abs(diff_s.p95), abs(diff_s.p5))
    p95_trd = max(abs(trd_s.p95), abs(trd_s.p5))

    return ThermalDiagnostic(
        horizon_dt=float(loss_cfg.horizon_dt),
        thermal_norm=loss_cfg.thermal_norm,
        Gamma_th=float(loss_cfg.Gamma_th),
        tau_th=float(loss_cfg.tau_th),
        eps=float(loss_cfg.eps),
        diff=diff_s,
        rms_diff=rms_diff,
        target_rate=Stats.of(target_rate),
        target_rate_dt=trd_s,
        ratio_mean=ratio,
        ratio_of_means=ratio,
        ratio_p95=(p95_trd / p95_diff if p95_diff > 0 else math.inf),
        thermal_obs=float(thermal.detach()),
        thermal_obs_reference=float(thermal_ref.detach()),
        thermal_obs_flatness_limit=float(thermal_flat.detach()),
        class_obs=float(class_obs.detach()),
        grad_norm_thermal=gn_thermal,
        grad_norm_class=gn_class,
        grad_ratio=(gn_thermal / gn_class if gn_class > 0 else math.inf),
        grad_norm_thermal_weighted=loss_cfg.lambda_thermal_obs * gn_thermal,
        grad_norm_class_weighted=loss_cfg.lambda_class_obs * gn_class,
        grad_ratio_weighted=(
            (loss_cfg.lambda_thermal_obs * gn_thermal)
            / (loss_cfg.lambda_class_obs * gn_class)
            if gn_class > 0 else math.inf
        ),
        grad_norm_thermal_flatness=gn_flat,
        grad_norm_constraint_part=gn_constraint,
        constraint_fraction=(gn_constraint / gn_flat if gn_flat > 0 else math.inf),
        grad_norm_thermal_times_rms_diff=gn_thermal * rms_diff,
        lambda_thermal_obs=float(loss_cfg.lambda_thermal_obs),
        lambda_class_obs=float(loss_cfg.lambda_class_obs),
    )


def sweep_horizon_dt(
    model: PIRNNObserver,
    batch: dict[str, torch.Tensor],
    base_cfg: LossConfig,
    values: tuple[float, ...] | list[float] = DEFAULT_SWEEP,
) -> list[ThermalDiagnostic]:
    """Same batch, same parameters, same seed -- only ``horizon_dt`` varies."""
    results = []
    for dt in values:
        cfg = LossConfig(**{**asdict(base_cfg), "horizon_dt": float(dt)})
        results.append(diagnose_batch(model, batch, cfg))
    return results


def seed_spread(
    model_cfg: ModelConfig,
    base_cfg: LossConfig,
    seeds: list[int],
    batch_size: int,
    h5: str | None = None,
    phys_config: str = DEFAULT_PHYS_CONFIG,
    stride: int = 10,
    device: str = "cpu",
) -> list[tuple[int, ThermalDiagnostic]]:
    """Re-run the base diagnostic over several (model init, batch) seeds.

    The gradient-norm ratio depends on the model's current forecast scale (see
    ``grad_norm_thermal_times_rms_diff``), so a single seed's ratio is not a stable
    number. This reports the spread so the verdict rests on the invariant parts.
    """
    out = []
    for s in seeds:
        model, _ = make_model(model_cfg, ckpt=None, seed=s, device=device)
        if h5 is not None and Path(h5).exists():
            b = dataset_batch(h5, model_cfg, batch_size, s, config_path=phys_config,
                              stride=stride, device=device)
        else:
            b = synthetic_batch(model_cfg, batch_size, s, device=device)
        b.pop("_source", None)
        out.append((s, diagnose_batch(model, b, base_cfg)))
    return out


def crossover_horizon_dt(diag: ThermalDiagnostic, target_ratio: float = 1.0) -> float:
    """``horizon_dt`` at which ``mean|target_rate*dt| == target_ratio * mean|diff|``.

    ``target_rate * dt`` is exactly linear in ``dt`` and ``diff`` does not depend on it
    (the model forward never sees ``horizon_dt``), so this is exact, not interpolated:
    ``dt_cross = dt * target_ratio / ratio(dt)``.
    """
    if diag.ratio_mean <= 0:
        return math.inf
    return diag.horizon_dt * target_ratio / diag.ratio_mean


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(x: float, prec: int = 3) -> str:
    if x != x:
        return "nan"
    if x in (math.inf, -math.inf):
        return "inf"
    return f"{x:.{prec}e}"


def format_distribution_table(d: ThermalDiagnostic) -> str:
    rows = [
        ("diff = pred_detuning[:,1:] - pred_detuning[:,:-1]", d.diff),
        ("target_rate  (= -Gamma_th*U_int_est/tau_th, 1/s)", d.target_rate),
        ("target_rate * horizon_dt", d.target_rate_dt),
    ]
    lines = [
        "| quantity | mean\\|.\\| | median | p5 | p95 | std |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in rows:
        lines.append(
            f"| {name} | {_fmt(s.mean_abs)} | {_fmt(s.median)} | {_fmt(s.p5)} | "
            f"{_fmt(s.p95)} | {_fmt(s.std)} |"
        )
    return "\n".join(lines)


def format_sweep_table(results: list[ThermalDiagnostic]) -> str:
    lines = [
        "| horizon_dt (s) | mean\\|target_rate*dt\\| | ratio to mean\\|diff\\| | thermal_obs | "
        "\\|\\|g_thermal\\|\\| | \\|\\|g_class\\|\\| | grad ratio | weighted grad ratio | "
        "constraint frac |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for d in results:
        lines.append(
            f"| {_fmt(d.horizon_dt, 2)} | {_fmt(d.target_rate_dt.mean_abs)} | "
            f"{_fmt(d.ratio_mean)} | {d.thermal_obs:.6f} | {_fmt(d.grad_norm_thermal)} | "
            f"{_fmt(d.grad_norm_class)} | {_fmt(d.grad_ratio)} | "
            f"{_fmt(d.grad_ratio_weighted)} | {_fmt(d.constraint_fraction)} |"
        )
    return "\n".join(lines)


def format_seed_table(rows: list[tuple[int, ThermalDiagnostic]]) -> str:
    lines = [
        "| seed | RMS(diff) | ratio (c) | thermal_obs | \\|\\|g_thermal\\|\\| | "
        "\\|\\|g_class\\|\\| | grad ratio | \\|\\|g_thermal\\|\\|*RMS(diff) | constraint frac |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s, d in rows:
        lines.append(
            f"| {s} | {_fmt(d.rms_diff)} | {_fmt(d.ratio_mean)} | {d.thermal_obs:.6f} | "
            f"{_fmt(d.grad_norm_thermal)} | {_fmt(d.grad_norm_class)} | "
            f"{_fmt(d.grad_ratio)} | {_fmt(d.grad_norm_thermal_times_rms_diff)} | "
            f"{_fmt(d.constraint_fraction)} |"
        )
    return "\n".join(lines)


def build_report(
    prov: Provenance,
    base: ThermalDiagnostic,
    sweep: list[ThermalDiagnostic],
    seeds: list[tuple[int, ThermalDiagnostic]] | None = None,
) -> str:
    dt_cross = crossover_horizon_dt(base, 1.0)
    dt_cross_10pct = crossover_horizon_dt(base, 0.1)
    bracket_lo = max(
        (d.horizon_dt for d in sweep if d.ratio_mean < 1.0), default=None
    )
    bracket_hi = min(
        (d.horizon_dt for d in sweep if d.ratio_mean >= 1.0), default=None
    )
    max_err = max(abs(d.thermal_obs - d.thermal_obs_reference) for d in sweep + [base])

    lines = [
        "# thermal_obs prior diagnostic",
        "",
        "## Provenance",
        "",
        f"- batch: `{prov.batch_source}`  (B={prov.batch_size}, seed={prov.seed}, "
        f"device={prov.device})",
        f"- model: `{prov.model_source}`",
        f"- physics config: `{prov.phys_config}`  "
        f"(Gamma_th={_fmt(prov.Gamma_th)}, tau_th={_fmt(prov.tau_th)} s, "
        f"fsr_hz={_fmt(prov.fsr_hz)})",
        f"- dataset h5: `{prov.h5_path}`",
        f"- resolved horizon_dt: {_fmt(prov.horizon_dt_resolved or float('nan'))} s  "
        f"(provenance: {prov.horizon_dt_provenance})",
        f"- thermal_norm: `{base.thermal_norm}`, lambda_thermal_obs="
        f"{base.lambda_thermal_obs}, lambda_class_obs={base.lambda_class_obs}",
        "",
        "## (a)-(b) Residual component distributions "
        f"(at horizon_dt = {_fmt(base.horizon_dt, 3)} s)",
        "",
        format_distribution_table(base),
        "",
        "## (c)-(e) Core numbers at the resolved horizon_dt",
        "",
        f"- (c) mean|target_rate*dt| / mean|diff| = **{_fmt(base.ratio_mean)}**",
        f"      p95-vs-p95 tail ratio             = {_fmt(base.ratio_p95)}",
        f"- (d) thermal_obs                        = {base.thermal_obs:.9f}",
        f"      flatness limit (Gamma_th -> 0)     = {base.thermal_obs_flatness_limit:.9f}",
        f"      reference (loss.py::_thermal)      = {base.thermal_obs_reference:.9f}  "
        f"(|delta| = {_fmt(abs(base.thermal_obs - base.thermal_obs_reference))})",
        f"- (e) ||d thermal_obs / d params||_2      = {_fmt(base.grad_norm_thermal)}",
        f"      ||d class_obs   / d params||_2      = {_fmt(base.grad_norm_class)}",
        f"      ratio                               = **{_fmt(base.grad_ratio)}**",
        f"      lambda-weighted ratio               = **{_fmt(base.grad_ratio_weighted)}**",
        f"      ||g_flatness|| (Gamma_th -> 0)      = {_fmt(base.grad_norm_thermal_flatness)}",
        f"      ||g_thermal - g_flatness||          = {_fmt(base.grad_norm_constraint_part)}",
        f"      constraint fraction                 = **{_fmt(base.constraint_fraction)}**",
        "",
        "## horizon_dt sweep (fixed batch, fixed seed, fixed parameters)",
        "",
        format_sweep_table(sweep),
        "",
        "### Crossover",
        "",
        f"- ratio == 1 (product comparable to diff): horizon_dt = **{_fmt(dt_cross)} s**",
        f"- ratio == 0.1 (product first non-negligible): horizon_dt = {_fmt(dt_cross_10pct)} s",
        f"- swept bracket: last 'flatness prior' point = "
        f"{_fmt(bracket_lo) if bracket_lo else 'n/a (all points below 1)'}; "
        f"first 'rate-matching' point = "
        f"{_fmt(bracket_hi) if bracket_hi else 'n/a (no swept point reaches 1)'}",
        "",
        f"_Self-consistency: max |reconstruction - loss.py::_thermal| over all rows = "
        f"{_fmt(max_err)}._",
        "",
    ]
    if seeds:
        lines += [
            "## Seed spread (independent model init + batch, at the resolved horizon_dt)",
            "",
            format_seed_table(seeds),
            "",
        ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(
    h5: str | None = None,
    ckpt: str | None = None,
    phys_config: str = DEFAULT_PHYS_CONFIG,
    batch_size: int = 64,
    seed: int = 0,
    device: str = "cpu",
    horizon_dt: float | None = None,
    sweep: tuple[float, ...] | list[float] = DEFAULT_SWEEP,
    thermal_norm: str = "empirical",
    lambda_thermal_obs: float = 0.05,
    lambda_class_obs: float = 1.0,
    W: int = 200,
    H: int = 50,
    stride: int = 10,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Full diagnostic. Returns ``{"provenance", "base", "sweep", "seeds", "report"}``."""
    phys = load_physics(phys_config)
    h5_path = h5 if h5 is not None else DEFAULT_H5
    dt_resolved, dt_prov = resolve_horizon_dt(horizon_dt, h5_path, phys["fsr_hz"])

    model_cfg = ModelConfig(W=W, H=H)
    model, model_source = make_model(model_cfg, ckpt=ckpt, seed=seed, device=device)

    if h5 is not None and Path(h5).exists():
        batch = dataset_batch(
            h5, model_cfg, batch_size, seed, config_path=phys_config,
            stride=stride, device=device,
        )
    else:
        batch = synthetic_batch(model_cfg, batch_size, seed, device=device)
    batch_source = str(batch.pop("_source"))

    base_cfg = LossConfig(
        lambda_class_obs=lambda_class_obs,
        lambda_thermal_obs=lambda_thermal_obs,
        Gamma_th=phys["Gamma_th"],
        tau_th=phys["tau_th"],
        horizon_dt=dt_resolved,
        thermal_norm=thermal_norm,
    )

    base = diagnose_batch(model, batch, base_cfg)
    sweep_results = sweep_horizon_dt(model, batch, base_cfg, sweep)
    seed_rows = (
        seed_spread(model_cfg, base_cfg, seeds, batch_size, h5=h5,
                    phys_config=phys_config, stride=stride, device=device)
        if seeds else []
    )

    prov = Provenance(
        batch_source=batch_source,
        batch_size=int(batch["x"].shape[0]),
        seed=seed,
        device=device,
        model_source=model_source,
        phys_config=phys_config,
        h5_path=h5_path,
        horizon_dt_resolved=dt_resolved,
        horizon_dt_provenance=dt_prov,
        Gamma_th=phys["Gamma_th"],
        tau_th=phys["tau_th"],
        fsr_hz=phys["fsr_hz"],
        extra={"t_r_s": 1.0 / phys["fsr_hz"], "W": W, "H": H},
    )
    report = build_report(prov, base, sweep_results, seed_rows)
    return {
        "provenance": asdict(prov),
        "base": base.as_dict(),
        "sweep": [d.as_dict() for d in sweep_results],
        "seeds": [{"seed": s, **d.as_dict()} for s, d in seed_rows],
        "crossover_horizon_dt": crossover_horizon_dt(base, 1.0),
        "crossover_horizon_dt_10pct": crossover_horizon_dt(base, 0.1),
        "report": report,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--h5", default=None,
                   help="dataset h5; omitted or missing -> synthetic batch path")
    p.add_argument("--ckpt", default=None, help="observer checkpoint; omitted -> fresh seeded model")
    p.add_argument("--phys-config", default=DEFAULT_PHYS_CONFIG)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--horizon-dt", type=float, default=None,
                   help="explicit override; default resolves from dataset metadata")
    p.add_argument("--sweep", type=float, nargs="*", default=list(DEFAULT_SWEEP))
    p.add_argument("--thermal-norm", default="empirical", choices=("empirical", "physical"))
    p.add_argument("--lambda-thermal-obs", type=float, default=0.05)
    p.add_argument("--lambda-class-obs", type=float, default=1.0)
    p.add_argument("--W", type=int, default=200)
    p.add_argument("--H", type=int, default=50)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="extra seeds for a spread table (independent model init + batch)")
    p.add_argument("--json-out", default=None)
    p.add_argument("--markdown-out", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = _parse_args(argv)
    res = run(
        h5=a.h5, ckpt=a.ckpt, phys_config=a.phys_config, batch_size=a.batch_size,
        seed=a.seed, device=a.device, horizon_dt=a.horizon_dt, sweep=tuple(a.sweep),
        thermal_norm=a.thermal_norm, lambda_thermal_obs=a.lambda_thermal_obs,
        lambda_class_obs=a.lambda_class_obs, W=a.W, H=a.H, stride=a.stride,
        seeds=a.seeds,
    )
    print(res["report"])
    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v for k, v in res.items() if k != "report"}
        Path(a.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[wrote] {a.json_out}")
    if a.markdown_out:
        Path(a.markdown_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.markdown_out).write_text(res["report"], encoding="utf-8")
        print(f"[wrote] {a.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
