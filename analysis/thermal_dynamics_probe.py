"""Dynamic probe of the thermal term (``thermal_obs``) during observer training.

Part A (``analysis/thermal_prior_diagnostic.py``) measured the term at INITIALIZATION and
found it dominant in gradient norm (lambda-weighted ratio ~41x vs ``class_obs``), almost
entirely physics-free (constraint fraction ~1e-4), and self-normalizing such that
``||g_thermal|| ~ 1 / RMS(diff)``. That last property predicts a *self-amplifying* flatness
prior: the flatter the detuning forecast gets, the harder the term pushes.

A static measurement cannot tell whether that prediction materializes. This probe runs
short, tightly matched training loops and answers:

  Q-A  Does the 1/RMS(diff) runaway actually happen under training, or does the
       classification gradient hold it in check?
  Q-B  Does the term ACCELERATE decoder collapse (FN-3 / T3: cross-batch variance of
       ``pred_detuning[:, -1]`` going to zero)?
  Q-C  Do ``g_thermal`` and ``g_class`` oppose each other, or merely coexist?

THREE ARMS, identical seed / identical initial weights / identical batch sequence. The
loss configuration is the ONLY difference:

  ARM "full"          lambda_thermal_obs = 0.05, Gamma_th = <config value>
  ARM "flatness_only" lambda_thermal_obs = 0.05, Gamma_th = 0.0
                      -- term present at the same weight with ALL physics content removed.
                         This is the control that separates "flatness prior" from "physics".
  ARM "no_thermal"    lambda_thermal_obs = 0.0

WEIGHTED VS UNWEIGHTED COMPONENTS
``model/loss.py::forward`` returns each component RAW (unweighted): ``out["class_obs"]``,
``out["thermal_obs"]`` etc. are the bare terms, and only ``out["total"]`` applies the
``lambda_*`` factors (loss.py:277-282). So differentiating ``out["thermal_obs"]`` directly
gives the UNWEIGHTED gradient, which is what M1/M2 ask for; the lambda-weighted ratio in M3
is formed explicitly from the config lambdas. No adjustment was needed.

SYNTHETIC SURROGATE -- SCOPE LIMIT
No dataset is required or used. Batches come from
``thermal_prior_diagnostic.synthetic_batch`` (so the two tools agree on batch construction);
supervision targets are generated here as smooth, per-sample, context-dependent
trajectories, so both the classification and the forecast heads have real signal to learn
and the collapse detector has a non-degenerate reference. This is a surrogate task: it
establishes the term's gradient *mechanics*, not its effect on real soliton data. Absolute
accuracies are meaningless here; the arm-to-arm differences are the measurement.

CLI::

    python -m analysis.thermal_dynamics_probe                       # 3 seeds, 300 steps
    python -m analysis.thermal_dynamics_probe --steps 1000 --seeds 0 1 2
    python -m analysis.thermal_dynamics_probe --json-out out.json --markdown-out out.md
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.thermal_prior_diagnostic import (  # noqa: E402
    _flat_grad_norm,
    load_physics,
    make_model,
    synthetic_batch,
)
from model.loss import LossConfig, PhysicsInformedLoss  # noqa: E402
from model.pi_rnn import ModelConfig, PIRNNObserver  # noqa: E402
from model.train import collapse_diagnostics, resolve_horizon_dt  # noqa: E402

ARM_NAMES = ("full", "flatness_only", "no_thermal")

# Optimizer/regularization settings mirrored from config/training_config.yaml so the
# dynamics are the ones the real Phase-1 run would see.
LR = 3.0e-4
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
LAMBDA_THERMAL = 0.05
LAMBDA_CLASS = 1.0


# --------------------------------------------------------------------------- #
# Synthetic supervised task
# --------------------------------------------------------------------------- #
def make_batch_pool(
    model_cfg: ModelConfig,
    n_batches: int,
    batch_size: int,
    seed: int,
    device: str = "cpu",
) -> list[dict[str, torch.Tensor]]:
    """A fixed pool of batches, cycled during training so all arms see identical data.

    ``x`` / ``context`` come verbatim from ``thermal_prior_diagnostic.synthetic_batch``.
    Targets are deterministic functions of ``context`` through fixed random projections,
    so the task is genuinely learnable rather than pure noise:

      * ``state_label``     = argmax(context @ W_lab)          -- learnable classification
      * ``future_detuning`` = a_b * exp(-t/20) + c_b           -- per-sample, non-flat, with
                                                                  real cross-batch variance
                                                                  at the final horizon step
      * ``future_P_trans``  = p_b * (1 - exp(-t/12)) + q_b
    """
    g = torch.Generator(device="cpu").manual_seed(seed + 4242)
    C, H, S = model_cfg.n_context, model_cfg.H, model_cfg.n_states
    # Fixed projections shared by every batch in the pool (the "task").
    W_lab = torch.randn(C, S, generator=g)
    w_a = torch.randn(C, generator=g)
    w_c = torch.randn(C, generator=g)
    w_p = torch.randn(C, generator=g)
    w_q = torch.randn(C, generator=g)
    t = torch.arange(H, dtype=torch.float32)
    decay = torch.exp(-t / 20.0)
    rise = 1.0 - torch.exp(-t / 12.0)

    pool = []
    for i in range(n_batches):
        b = synthetic_batch(model_cfg, batch_size, seed * 1000 + i, device=device)
        b.pop("_source", None)
        ctx = b["context"]
        a = torch.tanh(ctx @ w_a).unsqueeze(1)
        c = torch.tanh(ctx @ w_c).unsqueeze(1) * 0.5
        p = torch.tanh(ctx @ w_p).unsqueeze(1)
        q = torch.tanh(ctx @ w_q).unsqueeze(1) * 0.5
        pool.append({
            "x": b["x"],
            "context": ctx,
            "state_label": (ctx @ W_lab).argmax(dim=1),
            "future_detuning": a * decay.unsqueeze(0) + c,
            "future_P_trans": p * rise.unsqueeze(0) + q,
        })
    return pool


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
@dataclass
class Arm:
    name: str
    lambda_thermal_obs: float
    Gamma_th: float

    def loss_config(self, phys: dict[str, float], horizon_dt: float) -> LossConfig:
        return LossConfig(
            lambda_class_obs=LAMBDA_CLASS,
            lambda_detune_obs=1.0,
            lambda_pred_obs=1.0,
            lambda_thermal_obs=self.lambda_thermal_obs,
            beta_nll=0.5,
            nll_logvar_floor=-10.0,
            thermal_norm="empirical",
            Gamma_th=self.Gamma_th,
            tau_th=phys["tau_th"],
            horizon_dt=horizon_dt,
        )


def build_arms(phys: dict[str, float]) -> list[Arm]:
    return [
        Arm("full", LAMBDA_THERMAL, phys["Gamma_th"]),
        Arm("flatness_only", LAMBDA_THERMAL, 0.0),
        # Gamma_th kept at the real value so g_thermal stays MEASURABLE in this arm even
        # though it is not in the optimized objective (lambda = 0).
        Arm("no_thermal", 0.0, phys["Gamma_th"]),
    ]


def assert_bit_identical(models: list[PIRNNObserver], names: list[str]) -> dict[str, Any]:
    """Every arm must start from bitwise-equal weights or the comparison is void."""
    ref = models[0].state_dict()
    mismatches: list[str] = []
    for m, name in zip(models[1:], names[1:]):
        sd = m.state_dict()
        if set(sd) != set(ref):
            mismatches.append(f"{name}: key set differs")
            continue
        for k, v in sd.items():
            if not torch.equal(v, ref[k]):
                mismatches.append(f"{name}:{k}")
    n_tensors = len(ref)
    ok = not mismatches
    if not ok:
        raise AssertionError(
            f"arms do NOT start from bit-identical weights; mismatches: {mismatches[:10]}"
        )
    return {
        "bit_identical": True,
        "n_tensors_compared": n_tensors,
        "n_arms": len(models),
        "method": "torch.equal on every state_dict tensor vs arm 0",
    }


# --------------------------------------------------------------------------- #
# Per-step measurement
# --------------------------------------------------------------------------- #
def measure(
    model: PIRNNObserver,
    probe: dict[str, torch.Tensor],
    arm_cfg: LossConfig,
    logvar_max: float,
) -> dict[str, float]:
    """M1-M10 on a fixed probe batch, in eval() mode (no dropout -> deterministic).

    Gradients are taken of the RAW components returned by ``PhysicsInformedLoss.forward``
    (see module docstring: the repo returns components unweighted).
    """
    was_training = model.training
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]

    out = model(probe["x"], probe["context"])
    targets = {
        "state_label": probe["state_label"],
        "future_detuning": probe["future_detuning"],
        "future_P_trans": probe["future_P_trans"],
    }
    loss_fn = PhysicsInformedLoss(arm_cfg)
    comps = loss_fn(out, targets, mode="observer")

    # Flatness control: same term, same weights, Gamma_th -> 0.
    flat_cfg = LossConfig(**{**asdict(arm_cfg), "Gamma_th": 0.0})
    thermal_flat = PhysicsInformedLoss(flat_cfg)._thermal(
        out["pred_detuning"], out["phys_state_refined"]
    )

    gn_thermal, g_thermal = _flat_grad_norm(comps["thermal_obs"], params)   # M1
    gn_class, g_class = _flat_grad_norm(comps["class_obs"], params)         # M2
    gn_flat, g_flat = _flat_grad_norm(thermal_flat, params)
    gn_constraint = float((g_thermal - g_flat).norm())                      # M9 numerator

    diff = out["pred_detuning"][:, 1:] - out["pred_detuning"][:, :-1]
    rms_diff = float(diff.detach().double().pow(2).mean().sqrt())           # M4

    denom = float(g_thermal.norm()) * float(g_class.norm())
    cosine = float(torch.dot(g_thermal, g_class)) / denom if denom > 0 else float("nan")  # M6

    diag = collapse_diagnostics(out, logvar_max=logvar_max)                 # M7, M8

    # M7 measures cross-batch variance at the FINAL horizon step only. At init that step
    # already sits in the decoder's input-independent fixed point (FN-3), so the detector
    # can be pinned at the float floor before any optimizer step and would then be blind
    # to arm-to-arm differences. The full cross-batch variance profile along the horizon
    # locates where the collapse boundary is and whether the arms move it.
    det = out["pred_detuning"].detach().double()
    var_profile = det.var(dim=0, unbiased=False)                            # [H]
    alive = (var_profile > 1e-12).nonzero().flatten()
    effective_horizon = int(alive[-1].item()) + 1 if alive.numel() else 0
    H = var_profile.numel()

    # Gradient budget: the Trainer clips the TOTAL gradient to max_norm = GRAD_CLIP, so a
    # loud auxiliary term shrinks every other term's effective step by the clip factor.
    g_total = (
        arm_cfg.lambda_class_obs * g_class
        + arm_cfg.lambda_thermal_obs * g_thermal
    )
    # (detune/pred terms omitted from this particular budget estimate; the full pre-clip
    # norm of the real objective is recorded separately during the training step.)
    gn_budget = float(g_total.norm())

    rec = {
        "grad_norm_thermal": gn_thermal,                                    # M1
        "grad_norm_class": gn_class,                                        # M2
        "grad_ratio": gn_thermal / gn_class if gn_class > 0 else math.inf,  # M3
        "grad_ratio_weighted": (
            (arm_cfg.lambda_thermal_obs * gn_thermal)
            / (arm_cfg.lambda_class_obs * gn_class) if gn_class > 0 else math.inf
        ),                                                                  # M3
        "rms_diff": rms_diff,                                               # M4
        "product_M5": gn_thermal * rms_diff,                                # M5
        "cosine_thermal_class": cosine,                                     # M6
        "final_step_detuning_var": diag["final_step_detuning_var"],         # M7
        "final_step_ptrans_var": diag["final_step_ptrans_var"],
        "detuning_var_h0": float(var_profile[0]),
        "detuning_var_h10": float(var_profile[min(10, H - 1)]),
        "detuning_var_h25": float(var_profile[min(25, H - 1)]),
        "effective_horizon": float(effective_horizon),
        "frac_logvar_pinned": diag["frac_logvar_pinned"],                   # M8
        "constraint_fraction": (
            gn_constraint / gn_flat if gn_flat > 0 else math.inf
        ),                                                                  # M9
        "grad_norm_class_plus_thermal": gn_budget,
        "class_obs": float(comps["class_obs"].detach()),                    # M10
        "detune_obs": float(comps["detune_obs"].detach()),
        "pred_obs": float(comps["pred_obs"].detach()),
        "thermal_obs": float(comps["thermal_obs"].detach()),
        "total": float(comps["total"].detach()),
    }
    if was_training:
        model.train()
    return rec


def _independent_collapse_var(out: dict[str, torch.Tensor]) -> float:
    """Reimplementation of the T3 detector, used once to cross-check ``collapse_diagnostics``."""
    v = out["pred_detuning"][:, -1].detach().double()
    return float(((v - v.mean()) ** 2).mean())


# --------------------------------------------------------------------------- #
# One arm
# --------------------------------------------------------------------------- #
def run_arm(
    arm: Arm,
    model: PIRNNObserver,
    pool: list[dict[str, torch.Tensor]],
    probe: dict[str, torch.Tensor],
    arm_cfg: LossConfig,
    steps: int,
    record_every: int,
    seed: int,
    logvar_max: float,
) -> list[dict[str, float]]:
    """Train ``steps`` optimizer steps, recording M1-M10 every ``record_every`` steps.

    Dropout masks are made identical across arms by re-seeding the global RNG from
    ``(seed, step)`` before each training forward, so the ONLY thing that differs between
    arms is the loss configuration.
    """
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    loss_fn = PhysicsInformedLoss(arm_cfg)
    records: list[dict[str, float]] = []

    for step in range(steps + 1):
        if step % record_every == 0:
            rec = measure(model, probe, arm_cfg, logvar_max)
            rec["step"] = step
            records.append(rec)
        if step == steps:
            break

        batch = pool[step % len(pool)]
        # Identical dropout masks across arms at the same step.
        torch.manual_seed(seed * 100_003 + step)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["x"], batch["context"])
        targets = {
            "state_label": batch["state_label"],
            "future_detuning": batch["future_detuning"],
            "future_P_trans": batch["future_P_trans"],
        }
        comps = loss_fn(out, targets, mode="observer")
        comps["total"].backward()
        pre_clip = clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], GRAD_CLIP
        )
        if records and records[-1]["step"] == step:
            # Pre-clip total-gradient norm of the actual optimized objective, and the
            # factor by which clipping shrinks every term's effective step.
            pcn = float(pre_clip)
            records[-1]["total_grad_norm_preclip"] = pcn
            records[-1]["clip_factor"] = min(1.0, GRAD_CLIP / pcn) if pcn > 0 else 1.0
        optimizer.step()

    return records


# --------------------------------------------------------------------------- #
# One seed = three arms
# --------------------------------------------------------------------------- #
def run_seed(
    seed: int,
    steps: int,
    record_every: int,
    batch_size: int,
    n_pool: int,
    phys: dict[str, float],
    horizon_dt: float,
    device: str = "cpu",
) -> dict[str, Any]:
    model_cfg = ModelConfig()
    arms = build_arms(phys)

    # One model per arm, each built from the SAME seed -> must be bit-identical.
    models = [make_model(model_cfg, ckpt=None, seed=seed, device=device)[0] for _ in arms]
    identity = assert_bit_identical(models, [a.name for a in arms])

    pool = make_batch_pool(model_cfg, n_pool, batch_size, seed, device=device)
    probe = pool[0]

    # One-time cross-check of the T3 detector against an independent reimplementation.
    with torch.no_grad():
        models[0].eval()
        o = models[0](probe["x"], probe["context"])
    repo_var = collapse_diagnostics(o, logvar_max=model_cfg.logvar_max)["final_step_detuning_var"]
    own_var = _independent_collapse_var(o)
    identity["collapse_detector_crosscheck"] = {
        "repo_collapse_diagnostics": repo_var,
        "independent_reimplementation": own_var,
        "abs_delta": abs(repo_var - own_var),
    }

    out: dict[str, Any] = {"seed": seed, "identity": identity, "arms": {}}
    for arm, model in zip(arms, models):
        cfg = arm.loss_config(phys, horizon_dt)
        out["arms"][arm.name] = {
            "lambda_thermal_obs": arm.lambda_thermal_obs,
            "Gamma_th": arm.Gamma_th,
            "records": run_arm(arm, model, pool, probe, cfg, steps, record_every,
                               seed, model_cfg.logvar_max),
        }
    return out


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _cv(xs: list[float]) -> float:
    """Coefficient of variation (std/mean) over a series; inf if the mean vanishes."""
    xs = [x for x in xs if x == x and abs(x) != math.inf]
    if len(xs) < 2:
        return float("nan")
    m = statistics.fmean(xs)
    if m == 0:
        return math.inf
    return statistics.pstdev(xs) / abs(m)


def _series(seed_res: dict[str, Any], arm: str, key: str) -> list[float]:
    return [r[key] for r in seed_res["arms"][arm]["records"] if key in r]


def summarize_seed(seed_res: dict[str, Any]) -> dict[str, Any]:
    """O3/O4/O5 per-seed summary numbers."""
    s: dict[str, Any] = {"seed": seed_res["seed"], "arms": {}}
    for arm in ARM_NAMES:
        prod = _series(seed_res, arm, "product_M5")
        gt = _series(seed_res, arm, "grad_norm_thermal")
        rd = _series(seed_res, arm, "rms_diff")
        cos = _series(seed_res, arm, "cosine_thermal_class")
        var = _series(seed_res, arm, "final_step_detuning_var")
        cf = _series(seed_res, arm, "constraint_fraction")
        clip = _series(seed_res, arm, "clip_factor")
        eh = _series(seed_res, arm, "effective_horizon")
        v0 = _series(seed_res, arm, "detuning_var_h0")
        s["arms"][arm] = {
            "product_M5_cv": _cv(prod),
            "product_M5_first": prod[0], "product_M5_last": prod[-1],
            "grad_norm_thermal_first": gt[0], "grad_norm_thermal_last": gt[-1],
            "grad_norm_thermal_growth": (gt[-1] / gt[0]) if gt[0] > 0 else math.inf,
            "rms_diff_first": rd[0], "rms_diff_last": rd[-1],
            "rms_diff_ratio": (rd[-1] / rd[0]) if rd[0] > 0 else math.inf,
            "cosine_mean": statistics.fmean(cos), "cosine_min": min(cos),
            "cosine_max": max(cos), "cosine_last": cos[-1],
            "cosine_frac_negative": sum(1 for c in cos if c < 0) / len(cos),
            "final_var_first": var[0], "final_var_last": var[-1],
            "final_var_min": min(var),
            "constraint_fraction_mean": statistics.fmean(cf),
            "clip_factor_mean": statistics.fmean(clip) if clip else float("nan"),
            "effective_horizon_first": eh[0], "effective_horizon_last": eh[-1],
            "effective_horizon_min": min(eh), "effective_horizon_max": max(eh),
            "detuning_var_h0_first": v0[0], "detuning_var_h0_last": v0[-1],
        }
    # O5: arm1 vs arm2 across every recorded scalar.
    s["arm1_vs_arm2"] = compare_arms(seed_res, "full", "flatness_only")
    s["arm1_vs_arm3"] = compare_arms(seed_res, "full", "no_thermal")
    return s


def compare_arms(seed_res: dict[str, Any], a: str, b: str) -> dict[str, Any]:
    """Max/mean relative difference between two arms over every recorded metric."""
    ra = seed_res["arms"][a]["records"]
    rb = seed_res["arms"][b]["records"]
    keys = [k for k in ra[0] if k != "step"]
    per_key: dict[str, float] = {}
    for k in keys:
        rels = []
        for x, y in zip(ra, rb):
            if k not in x or k not in y:
                continue
            xv, yv = x[k], y[k]
            if any(v != v or abs(v) == math.inf for v in (xv, yv)):
                continue
            scale = max(abs(xv), abs(yv))
            rels.append(abs(xv - yv) / scale if scale > 0 else 0.0)
        if rels:
            per_key[k] = max(rels)
    worst = max(per_key.items(), key=lambda kv: kv[1]) if per_key else ("n/a", float("nan"))
    return {
        "max_rel_diff_per_metric": per_key,
        "worst_metric": worst[0],
        "worst_max_rel_diff": worst[1],
        "max_rel_diff_over_all_metrics": worst[1],
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _f(x: float, p: int = 3) -> str:
    if x != x:
        return "nan"
    if abs(x) == math.inf:
        return "inf"
    return f"{x:.{p}e}"


O1_COLS = [
    ("step", "step", 0),
    ("grad_norm_thermal", "M1 ‖g_th‖", 3),
    ("grad_norm_class", "M2 ‖g_cls‖", 3),
    ("grad_ratio", "M3 ratio", 3),
    ("grad_ratio_weighted", "M3 w.ratio", 3),
    ("rms_diff", "M4 RMS(diff)", 3),
    ("product_M5", "M5 prod", 3),
    ("cosine_thermal_class", "M6 cos", 4),
    ("final_step_detuning_var", "M7 finvar", 3),
    ("frac_logvar_pinned", "M8 pinned", 3),
    ("constraint_fraction", "M9 cfrac", 3),
    ("total", "M10 total", 4),
    ("class_obs", "class", 4),
    ("thermal_obs", "thermal", 6),
]


def format_arm_table(records: list[dict[str, float]], every: int = 1) -> str:
    head = "| " + " | ".join(c[1] for c in O1_COLS) + " |"
    sep = "|" + "---|" * len(O1_COLS)
    lines = [head, sep]
    for i, r in enumerate(records):
        if i % every:
            continue
        cells = []
        for key, _, prec in O1_COLS:
            v = r.get(key, float("nan"))
            if key == "step":
                cells.append(str(int(v)))
            elif key in ("cosine_thermal_class", "frac_logvar_pinned", "total",
                         "class_obs", "thermal_obs"):
                cells.append("nan" if v != v else f"{v:.{prec}f}")
            else:
                cells.append(_f(v, prec))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_collapse_table(seed_res: dict[str, Any], every: int = 1) -> str:
    steps = [r["step"] for r in seed_res["arms"]["full"]["records"]]
    cols = {a: _series(seed_res, a, "final_step_detuning_var") for a in ARM_NAMES}
    lines = [
        "| step | full | flatness_only | no_thermal | full/no_thermal | flat/no_thermal |",
        "|---|---|---|---|---|---|",
    ]
    for i, st in enumerate(steps):
        if i % every:
            continue
        f_, fl, nt = cols["full"][i], cols["flatness_only"][i], cols["no_thermal"][i]
        lines.append(
            f"| {st} | {_f(f_)} | {_f(fl)} | {_f(nt)} | "
            f"{_f(f_ / nt) if nt > 0 else 'inf'} | {_f(fl / nt) if nt > 0 else 'inf'} |"
        )
    return "\n".join(lines)


def format_horizon_table(seed_res: dict[str, Any], every: int = 1) -> str:
    """Where the decoder still carries cross-batch information, per arm.

    ``eff_H`` is the last horizon index with cross-batch variance > 1e-12, +1; ``var@h0``
    is the cross-batch variance at the FIRST horizon step (not yet inside the fixed point,
    so unlike M7 it is not pinned at the float floor and can resolve arm differences).
    """
    steps = [r["step"] for r in seed_res["arms"]["full"]["records"]]
    lines = [
        "| step | eff_H full | eff_H flat | eff_H no_th | var@h0 full | var@h0 flat | "
        "var@h0 no_th | RMS(diff) full | RMS(diff) flat | RMS(diff) no_th |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    eh = {a: _series(seed_res, a, "effective_horizon") for a in ARM_NAMES}
    v0 = {a: _series(seed_res, a, "detuning_var_h0") for a in ARM_NAMES}
    rd = {a: _series(seed_res, a, "rms_diff") for a in ARM_NAMES}
    for i, st in enumerate(steps):
        if i % every:
            continue
        lines.append(
            f"| {st} | {int(eh['full'][i])} | {int(eh['flatness_only'][i])} | "
            f"{int(eh['no_thermal'][i])} | {_f(v0['full'][i])} | "
            f"{_f(v0['flatness_only'][i])} | {_f(v0['no_thermal'][i])} | "
            f"{_f(rd['full'][i])} | {_f(rd['flatness_only'][i])} | {_f(rd['no_thermal'][i])} |"
        )
    return "\n".join(lines)


def build_report(results: list[dict[str, Any]], summaries: list[dict[str, Any]],
                 steps: int, record_every: int, table_every: int,
                 meta: dict[str, Any]) -> str:
    L: list[str] = ["# thermal_obs dynamics probe", ""]
    L += [
        "## Setup",
        "",
        f"- steps per arm: {steps}, recorded every {record_every}",
        f"- batch pool: {meta['n_pool']} batches x B={meta['batch_size']}, cycled; "
        f"probe batch = pool[0], measured in eval() mode",
        f"- optimizer: AdamW(lr={LR}, weight_decay={WEIGHT_DECAY}), "
        f"grad clip max_norm={GRAD_CLIP} (mirrors config/training_config.yaml)",
        f"- horizon_dt = {_f(meta['horizon_dt'])} s ({meta['horizon_dt_provenance']}), "
        f"Gamma_th = {_f(meta['Gamma_th'])}, tau_th = {_f(meta['tau_th'])} s",
        "- loss components are returned UNWEIGHTED by model/loss.py; gradients below are "
        "of the raw components",
        "",
    ]
    for res, summ in zip(results, summaries):
        seed = res["seed"]
        ident = res["identity"]
        L += [
            f"## Seed {seed}",
            "",
            f"- bit-identical initial weights across all {ident['n_arms']} arms: "
            f"**{ident['bit_identical']}** "
            f"({ident['n_tensors_compared']} state_dict tensors, {ident['method']})",
            f"- T3 detector cross-check: repo collapse_diagnostics = "
            f"{_f(ident['collapse_detector_crosscheck']['repo_collapse_diagnostics'])}, "
            f"independent reimplementation = "
            f"{_f(ident['collapse_detector_crosscheck']['independent_reimplementation'])}, "
            f"|delta| = {_f(ident['collapse_detector_crosscheck']['abs_delta'])}",
            "",
        ]
        for arm in ARM_NAMES:
            L += [f"### O1 -- seed {seed}, ARM `{arm}`", "",
                  format_arm_table(res["arms"][arm]["records"], table_every), ""]
        L += [f"### O2 -- seed {seed}, collapse comparison "
              "(final_step_detuning_var)", "",
              format_collapse_table(res, table_every), "",
              f"### O2b -- seed {seed}, horizon-profile collapse "
              "(effective horizon + un-pinned early-step variance)", "",
              format_horizon_table(res, table_every), ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(
    seeds: list[int] | None = None,
    steps: int = 300,
    record_every: int = 10,
    table_every: int = 3,
    batch_size: int = 32,
    n_pool: int = 8,
    phys_config: str = "config/sin_params.yaml",
    h5: str | None = None,
    horizon_dt: float | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    seeds = seeds if seeds else [0, 1, 2]
    phys = load_physics(phys_config)
    dt, dt_prov = resolve_horizon_dt(
        horizon_dt, h5 or "data/synthetic/dataset.h5", phys["fsr_hz"]
    )
    results = [
        run_seed(s, steps, record_every, batch_size, n_pool, phys, dt, device=device)
        for s in seeds
    ]
    summaries = [summarize_seed(r) for r in results]
    meta = {
        "n_pool": n_pool, "batch_size": batch_size, "horizon_dt": dt,
        "horizon_dt_provenance": dt_prov, "Gamma_th": phys["Gamma_th"],
        "tau_th": phys["tau_th"], "seeds": seeds, "steps": steps,
    }
    return {
        "meta": meta,
        "results": results,
        "summaries": summaries,
        "report": build_report(results, summaries, steps, record_every, table_every, meta),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--record-every", type=int, default=10)
    p.add_argument("--table-every", type=int, default=3,
                   help="print every Nth recorded row in the O1/O2 tables")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-pool", type=int, default=8)
    p.add_argument("--phys-config", default="config/sin_params.yaml")
    p.add_argument("--h5", default=None, help="only used to resolve horizon_dt provenance")
    p.add_argument("--horizon-dt", type=float, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--json-out", default=None)
    p.add_argument("--markdown-out", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = _parse_args(argv)
    res = run(
        seeds=a.seeds, steps=a.steps, record_every=a.record_every,
        table_every=a.table_every, batch_size=a.batch_size, n_pool=a.n_pool,
        phys_config=a.phys_config, h5=a.h5, horizon_dt=a.horizon_dt, device=a.device,
    )
    print(res["report"])
    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(
            json.dumps({k: v for k, v in res.items() if k != "report"}, indent=2),
            encoding="utf-8",
        )
        print(f"[wrote] {a.json_out}")
    if a.markdown_out:
        Path(a.markdown_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.markdown_out).write_text(res["report"], encoding="utf-8")
        print(f"[wrote] {a.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
