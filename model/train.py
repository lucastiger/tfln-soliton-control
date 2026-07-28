"""Training pipeline for the TFLN soliton-control PI-RNN.

Two phases, matching ``model/pi_rnn.py``:

  * Phase 1 — Observer (``train_observer``): fully runnable on the existing
    open-loop sweep dataset (``data/dataloader.py``). Classifies the current
    soliton state and forecasts no-intervention dynamics.
  * Phase 2 — Controller (``train_controller``): loop is implemented but GATED.
    The current dataloader emits no switching/action data, so Phase 2 raises
    ``NotImplementedError`` until a switching dataset is supplied. See the
    ``train_controller`` docstring for the two binding gates (closed-loop
    distribution shift; action-space density).

This module integrates against the LOCKED signatures of ``model/pi_rnn.py``,
``model/loss.py`` and ``data/dataloader.py`` and does not modify them. The only
key-rename debt handled here: the dataloader emits the class label under
``"label"``; the loss expects ``"state_label"`` (see ``assemble_observer_targets``).

Physics constants for the thermal loss term (``Gamma_th``, ``tau_th``,
``horizon_dt``) are injected from ``config/tfln_params.yaml`` at runtime via
``simulator.lle_solver._load_config`` — never from the model context vector.

CLI::

    python -m model.train --config config/training_config.yaml --experiment_name my_run \
        [--phase observer|controller] \
        [--ablation full|no_physics_loss|transformer_backbone|si3n4_pretrain] \
        [--device cuda|cpu] [--epochs N] [--override key=value ...]
"""

from __future__ import annotations

import argparse
import json
import random
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Protocol, runtime_checkable

import h5py
import numpy as np
import torch
import yaml
from torch.nn.utils import clip_grad_norm_

from data.dataloader import N_CLASSES, get_dataloaders
from model.loss import LossConfig, LossScaler, PhysicsInformedLoss
from model.pi_rnn import (
    SINGLE_SOLITON_IDX,
    ModelConfig,
    PIRNNController,
    PIRNNObserver,
)
from simulator.lle_solver import _load_config

# Fixed model dimensions that the architecture/data contract pins (not user-tunable).
N_CONTEXT: int = 4

ObserverFactory = Callable[[ModelConfig], torch.nn.Module]


# --------------------------------------------------------------------------- #
# Config plumbing
# --------------------------------------------------------------------------- #
def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def to_namespace(obj: Any) -> Any:
    """Recursively convert nested dicts to attribute-access ``SimpleNamespace`` trees."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_namespace(v) for v in obj]
    return obj


def namespace_to_dict(obj: Any) -> Any:
    """Inverse of ``to_namespace`` (for YAML snapshotting / serialization)."""
    if isinstance(obj, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [namespace_to_dict(v) for v in obj]
    return obj


def parse_override_value(raw: str) -> Any:
    """Parse a CLI ``--override`` scalar via YAML (so ``3`` -> int, ``true`` -> bool, ``null`` -> None)."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def apply_overrides(cfg_dict: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Apply dotted-key overrides in place, e.g. ``{"optim.epochs": 3}``."""
    if not overrides:
        return cfg_dict
    for dotted, value in overrides.items():
        keys = dotted.split(".")
        node = cfg_dict
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value
    return cfg_dict


def resolve_device(spec: str | None) -> torch.device:
    if spec is None or spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[train] seed = {seed}")


def _read_snapshot_interval(h5_path: str | Path | None) -> int | None:
    """Single source of truth for reading ``metadata['snapshot_interval']`` from a dataset h5.

    Returns ``int(h5["metadata"].attrs["snapshot_interval"])`` when the file exists and the
    attr is present, else ``None``. A missing/``None`` path, a missing attr, or an unreadable
    file all map to ``None`` — this never raises on a not-found or corrupt dataset. Opened
    read-only.
    """
    if h5_path is None:
        return None
    path = Path(h5_path)
    if not path.exists():
        return None
    try:
        with h5py.File(path, "r") as h5:
            if "metadata" in h5 and "snapshot_interval" in h5["metadata"].attrs:
                return int(h5["metadata"].attrs["snapshot_interval"])
    except OSError:
        return None
    return None


def resolve_horizon_dt(
    explicit: float | None,
    h5_path: str | Path,
    fsr_hz: float,
) -> tuple[float, str]:
    """Resolve the forecast-horizon snapshot spacing (seconds per horizon step).

    The empirical thermal residual is ``(diff - target_rate*horizon_dt)^2 / mean(diff^2)``,
    so ``horizon_dt`` scales the relaxation-target weight and MUST track the dataset rather
    than be a magic constant. Precedence:

      a. EXPLICIT OVERRIDE — ``explicit`` is a non-null number -> used verbatim;
         provenance ``"explicit_config"``.
      b. DERIVED — read ``snapshot_interval`` (via ``_read_snapshot_interval``, the single
         read path) and combine with ``t_r = 1/fsr_hz`` -> ``snapshot_interval / fsr_hz``;
         provenance ``"derived(snapshot_interval=<n>, t_r=<t_r>)"``.
      c. FALLBACK — h5 missing, unreadable, or has no ``metadata['snapshot_interval']`` attr
         (older datasets / unit tests) -> ``RuntimeWarning`` and ``10 / fsr_hz``; provenance
         ``"fallback(snapshot_interval=10)"``. NOT the old 1e-7.

    ``fsr_hz`` is coerced to float (tfln_params.yaml stores it as an unsigned-exponent string).
    """
    if explicit is not None:
        return float(explicit), "explicit_config"

    fsr_hz = float(fsr_hz)
    t_r = 1.0 / fsr_hz
    snapshot_interval = _read_snapshot_interval(h5_path)

    if snapshot_interval is None:
        warnings.warn(
            f"resolve_horizon_dt: could not read metadata['snapshot_interval'] from "
            f"{Path(h5_path)!s} (missing file, missing attr, or unreadable); falling back to "
            "snapshot_interval=10.", RuntimeWarning, stacklevel=2,
        )
        return float(10 / fsr_hz), "fallback(snapshot_interval=10)"

    horizon_dt = snapshot_interval / fsr_hz
    return float(horizon_dt), f"derived(snapshot_interval={snapshot_interval}, t_r={t_r:.3e})"


def assert_matching_snapshot_interval(
    observer_h5: str | Path | None,
    switching_h5: str | Path | None,
) -> None:
    """Guard against mixing two snapshot-interval time bases in Phase-2 controller training.

    In ``train_controller`` the predicted detuning is ``observer_baseline.detach() + correction``:
    the baseline is forecast at the OBSERVER dataset's snapshot spacing while the correction is
    supervised against the SWITCHING dataset's spacing. If the two datasets were generated with
    different ``snapshot_interval`` values, that sum sits on incoherent time bases.

    Reads both intervals via ``_read_snapshot_interval``. Raises ``ValueError`` only when BOTH
    are known and unequal; if either is ``None`` (dataset absent or older/no-attr) the check
    cannot be made yet and this returns silently.
    """
    obs_si = _read_snapshot_interval(observer_h5)
    sw_si = _read_snapshot_interval(switching_h5)
    if obs_si is not None and sw_si is not None and obs_si != sw_si:
        raise ValueError(
            f"snapshot_interval mismatch: observer dataset = {obs_si}, switching dataset = "
            f"{sw_si}. The observer-baseline forecast and the switching-correction supervision "
            "would sit on different time bases (observer_baseline.detach() + correction in "
            "train_controller). Regenerate both datasets with the same snapshot_interval."
        )


# --------------------------------------------------------------------------- #
# Target assembly (handles the label -> state_label rename debt)
# --------------------------------------------------------------------------- #
def assemble_observer_targets(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Build the observer ``targets`` dict from a dataloader batch.

    Key-rename debt lives here: the dataloader emits the class label as ``"label"``;
    the loss expects ``"state_label"``.
    """
    return {
        "state_label": batch["label"].to(device).long(),
        "future_detuning": batch["future_detuning"].to(device).float(),
        "future_P_trans": batch["future_P_trans"].to(device).float(),
    }


def assemble_controller_targets(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Build the controller ``targets`` dict from a switching-dataset batch (see ``SwitchingDataset``)."""
    return {
        "target_state_label": batch["target_state_label"].to(device).long(),
        "future_detuning_under_action": batch["future_detuning_under_action"].to(device).float(),
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
class ClassMetricAccumulator:
    """Streaming overall accuracy + per-class recall over ``n_states`` classes.

    Tracks the figure-of-merit class (``focus_idx``, single_soliton) recall separately.
    Per-class recall for a class with zero support in the split is reported as ``None``
    (JSON null), to distinguish "no support" from "0% recall".
    """

    def __init__(self, n_states: int = N_CLASSES, focus_idx: int = SINGLE_SOLITON_IDX):
        self.n_states = n_states
        self.focus_idx = focus_idx
        self.correct = 0
        self.total = 0
        self.tp = torch.zeros(n_states, dtype=torch.long)
        self.support = torch.zeros(n_states, dtype=torch.long)

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        preds = logits.detach().argmax(dim=-1).cpu()
        labels = labels.detach().cpu().view(-1)
        self.correct += int((preds == labels).sum())
        self.total += int(labels.numel())
        for c in range(self.n_states):
            cmask = labels == c
            self.support[c] += int(cmask.sum())
            self.tp[c] += int((preds[cmask] == c).sum())

    def compute(self) -> dict[str, Any]:
        accuracy = (self.correct / self.total) if self.total > 0 else 0.0
        per_class_recall: list[float | None] = []
        for c in range(self.n_states):
            sup = int(self.support[c])
            per_class_recall.append(float(self.tp[c]) / sup if sup > 0 else None)
        return {
            "accuracy": accuracy,
            "per_class_recall": per_class_recall,
            "single_soliton_recall": per_class_recall[self.focus_idx],
        }


def collapse_diagnostics(
    out: dict[str, torch.Tensor],
    logvar_max: float,
    pin_tol: float = 1e-3,
    last_k: int = 10,
) -> dict[str, float]:
    """FN-3 / T3 collapse diagnostics on a single forward output.

    The collapse detector uses the BATCH-dimension variance at the final horizon step
    (within-trajectory flatness is expected; cross-trajectory flatness is the symptom).
    Also reports the fraction of late-horizon detuning logvar pinned at ``logvar_max``
    (an uninformative forecast tail => effective horizon < H).
    """
    det_last = out["pred_detuning"][:, -1]
    ptr_last = out["pred_P_trans"][:, -1]
    det_var = float(det_last.var(unbiased=False).item())
    ptr_var = float(ptr_last.var(unbiased=False).item())
    # Late-horizon logvar pinning (prompt 5.3): fraction of the last K = min(last_k, H) detuning
    # logvar entries pinned at the head's ceiling. One-sided test against logvar_max (the head
    # clamps to logvar_max, so values cannot exceed it).
    K = min(last_k, out["pred_detuning_logvar"].size(1))
    lv_tail = out["pred_detuning_logvar"][:, -K:]
    frac_pinned = float((lv_tail >= (logvar_max - pin_tol)).float().mean().item())
    return {
        "final_step_detuning_var": det_var,
        "final_step_ptrans_var": ptr_var,
        "frac_logvar_pinned": frac_pinned,
    }


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class Trainer:
    """Epoch loop, validation, checkpointing, logging, early stopping for one model.

    ``mode`` selects observer vs controller plumbing (forward signature, target
    assembly, loss kwargs, and the logits/label pair used for classification metrics).
    Only ``loss["total"]`` is backpropagated; every other component is logged via ``.item()``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_fn: PhysicsInformedLoss,
        train_loader: Any,
        val_loader: Any,
        cfg: SimpleNamespace,
        device: torch.device,
        mode: str = "observer",
        observer_frozen: bool = False,
        class_weights: torch.Tensor | None = None,
        scaler: LossScaler | None = None,
    ):
        if mode not in ("observer", "controller"):
            raise ValueError(f"mode must be 'observer' or 'controller', got {mode!r}")
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.mode = mode
        self.observer_frozen = observer_frozen
        self.class_weights = class_weights
        self.scaler = scaler

        self.trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.logvar_max = float(getattr(cfg.model, "logvar_max", 5.0))
        self.grad_clip = float(cfg.optim.grad_clip)
        self.epochs = int(cfg.optim.epochs)

        self.amp = bool(getattr(cfg.train, "amp", False)) and device.type == "cuda"
        self._grad_scaler = torch.amp.GradScaler(device.type, enabled=self.amp)

        # Output directory and artifacts.
        self.out_dir = Path(cfg.experiment.out_dir) / cfg.experiment.name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.out_dir / "metrics.jsonl"
        self.best_model_path = self.out_dir / "best_model.pt"
        with (self.out_dir / "config.snapshot.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(namespace_to_dict(cfg), f, sort_keys=False)

        # Early stop monitors a DIFFERENT metric than best-checkpoint selection (by design).
        self.early_stop_metric = str(getattr(cfg.train, "early_stop_metric", "val_total_loss"))
        self.early_stop_patience = int(getattr(cfg.train, "early_stop_patience", 10))
        self.checkpoint_metric = str(getattr(cfg.train, "checkpoint_metric", "val_accuracy"))
        self.checkpoint_every = int(getattr(cfg.train, "checkpoint_every", 5))

        self._best_ckpt_value = -float("inf")   # checkpoint_metric is mode "max" (val_accuracy)
        self._best_es_value = float("inf")       # early_stop_metric is mode "min" (val total loss)
        self._es_counter = 0
        self.best: dict[str, Any] = {}
        self._val_diag_batch: dict[str, torch.Tensor] | None = None

    # ----- per-batch plumbing --------------------------------------------- #
    def _forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = batch["x"].to(self.device).float()
        context = batch["context"].to(self.device).float()
        if self.mode == "observer":
            return self.model(x, context)
        delta_cmd = batch["delta_cmd"].to(self.device).float()
        target_state = batch["target_state"].to(self.device).long()
        return self.model(x, context, delta_cmd, target_state)

    def _targets(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.mode == "observer":
            return assemble_observer_targets(batch, self.device)
        return assemble_controller_targets(batch, self.device)

    def _loss(self, out: dict[str, torch.Tensor], targets: dict[str, torch.Tensor],
              batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.mode == "observer":
            return self.loss_fn(out, targets, mode="observer", class_weights=self.class_weights)
        mask = batch.get("valid_transition_mask")
        if mask is not None:
            mask = mask.to(self.device)
        return self.loss_fn(
            out, targets, mode="controller",
            observer_frozen=self.observer_frozen,
            class_weights_ctrl=self.class_weights,
            valid_transition_mask=mask,
        )

    def _logits_and_labels(self, out: dict[str, torch.Tensor],
                           targets: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "observer":
            return out["logits"], targets["state_label"]
        # Controller classification is over act_logits vs the target state.
        return out["act_logits"], targets["target_state_label"]

    # ----- train ----------------------------------------------------------- #
    def _train_epoch(self) -> dict[str, float]:
        self.model.train()  # PIRNNController.train() re-eval()s a frozen observer internally
        sums: dict[str, float] = {}
        n_batches = 0
        for batch in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                out = self._forward(batch)
                targets = self._targets(batch)
                loss = self._loss(out, targets, batch)
            self._grad_scaler.scale(loss["total"]).backward()
            self._grad_scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.trainable_params, self.grad_clip)   # MANDATORY (200-step BPTT)
            self._grad_scaler.step(self.optimizer)
            self._grad_scaler.update()
            if self.scaler is not None:
                self.scaler.update(loss)
            for k, v in loss.items():
                sums[k] = sums.get(k, 0.0) + float(v.item())
            n_batches += 1
        self.scheduler.step()  # per-epoch
        return {k: v / max(n_batches, 1) for k, v in sums.items()}

    # ----- validate -------------------------------------------------------- #
    @torch.no_grad()
    def _validate(self) -> tuple[dict[str, float], dict[str, Any], dict[str, float]]:
        self.model.eval()
        sums: dict[str, float] = {}
        n_batches = 0
        acc = ClassMetricAccumulator(N_CLASSES, SINGLE_SOLITON_IDX)
        for batch in self.val_loader:
            if self._val_diag_batch is None:
                self._val_diag_batch = batch  # cache the first val batch once for collapse diagnostics
            out = self._forward(batch)
            targets = self._targets(batch)
            loss = self._loss(out, targets, batch)
            for k, v in loss.items():
                sums[k] = sums.get(k, 0.0) + float(v.item())
            logits, labels = self._logits_and_labels(out, targets)
            acc.update(logits, labels)
            n_batches += 1
        means = {k: v / max(n_batches, 1) for k, v in sums.items()}
        metrics = acc.compute()
        diag = self._collapse_diag()
        return means, metrics, diag

    @torch.no_grad()
    def _collapse_diag(self) -> dict[str, float]:
        if self._val_diag_batch is None:
            return {"final_step_detuning_var": float("nan"),
                    "final_step_ptrans_var": float("nan"),
                    "frac_logvar_pinned": float("nan")}
        out = self._forward(self._val_diag_batch)
        diag = collapse_diagnostics(out, self.logvar_max)
        if diag["final_step_detuning_var"] < 1e-12:
            warnings.warn("collapse diagnostic: cross-batch variance of pred_detuning[:, -1] ~ 0 "
                          "(possible T3 forecast collapse).", RuntimeWarning, stacklevel=2)
        if diag["final_step_ptrans_var"] < 1e-12:
            warnings.warn("collapse diagnostic: cross-batch variance of pred_P_trans[:, -1] ~ 0 "
                          "(possible T3 forecast collapse).", RuntimeWarning, stacklevel=2)
        return diag

    # ----- checkpointing --------------------------------------------------- #
    def _rng_state(self) -> dict[str, Any]:
        state = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _checkpoint_payload(self, epoch: int, val_metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": epoch,
            "model_config": asdict(self.model.config),
            "loss_config": asdict(self.loss_fn.cfg),
            "val_metrics": val_metrics,
            "rng_state": self._rng_state(),
        }

    # ----- public eval (used by si3n4 ablation: frozen zero-shot eval) ----- #
    @torch.no_grad()
    def evaluate(self, loader: Any) -> dict[str, Any]:
        prev = self.val_loader
        prev_diag = self._val_diag_batch
        self.val_loader = loader
        self._val_diag_batch = None
        try:
            means, metrics, diag = self._validate()
        finally:
            self.val_loader = prev
            self._val_diag_batch = prev_diag
        return {"loss": means, "metrics": metrics, "diag": diag}

    # ----- fit ------------------------------------------------------------- #
    def fit(self) -> dict[str, Any]:
        self.metrics_path.write_text("", encoding="utf-8")  # truncate log
        for epoch in range(self.epochs):
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            lr = self.optimizer.param_groups[0]["lr"]
            t0 = time.time()
            train_means = self._train_epoch()
            val_means, val_metrics, diag = self._validate()
            elapsed = time.time() - t0

            # Adaptive lambda rebalancing (FN-1 respected by the scaler's own config).
            if self.scaler is not None:
                before = self._lambda_snapshot()
                self.scaler.maybe_rescale(epoch)  # mutates loss_fn.cfg in place (shared object)
                self._log_lambda_changes(before)

            gpu_mem = (torch.cuda.max_memory_allocated() / 1e6) if self.device.type == "cuda" else None
            record = {
                "epoch": epoch,
                "lr": lr,
                "train": train_means,
                "val": {**val_means, **val_metrics},
                "diag": diag,
                "gpu_mem_mb": gpu_mem,
                "time_s": elapsed,
            }
            self._append_metrics(record)
            self._print_epoch(record)

            self._save_periodic(epoch, val_metrics)
            self._update_best(epoch, val_means, val_metrics)
            if self._early_stop(val_means):
                print(f"[train] early stop at epoch {epoch} "
                      f"(no {self.early_stop_metric} improvement in {self.early_stop_patience} epochs)")
                break
        return self.best

    # ----- fit helpers ----------------------------------------------------- #
    def _lambda_snapshot(self) -> dict[str, float]:
        cfg = self.loss_fn.cfg
        return {k: getattr(cfg, k) for k in vars(LossConfig()) if k.startswith("lambda_")}

    def _log_lambda_changes(self, before: dict[str, float]) -> None:
        after = self._lambda_snapshot()
        changed = {k: (before[k], after[k]) for k in after if abs(after[k] - before[k]) > 1e-12}
        if changed:
            pretty = ", ".join(f"{k}: {o:.4g}->{n:.4g}" for k, (o, n) in changed.items())
            print(f"[scaler] rebalanced lambdas: {pretty}")

    def _append_metrics(self, record: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _print_epoch(self, record: dict[str, Any]) -> None:
        v = record["val"]
        ss = v["single_soliton_recall"]
        ss_str = f"{ss:.3f}" if ss is not None else "n/a"
        print(
            f"[{record['epoch']:03d}] lr={record['lr']:.2e} "
            f"train_total={record['train'].get('total', float('nan')):.4f} "
            f"val_total={v.get('total', float('nan')):.4f} "
            f"val_acc={v['accuracy']:.4f} ss_recall={ss_str} "
            f"t={record['time_s']:.1f}s"
        )

    def _save_periodic(self, epoch: int, val_metrics: dict[str, Any]) -> None:
        if self.checkpoint_every > 0 and (epoch + 1) % self.checkpoint_every == 0:
            path = self.out_dir / f"ckpt_epoch{epoch}.pt"
            torch.save(self._checkpoint_payload(epoch, val_metrics), path)

    def _update_best(self, epoch: int, val_means: dict[str, float], val_metrics: dict[str, Any]) -> None:
        value = val_metrics["accuracy"]  # checkpoint_metric == val_accuracy (mode max)
        if value > self._best_ckpt_value:
            self._best_ckpt_value = value
            torch.save(self._checkpoint_payload(epoch, val_metrics), self.best_model_path)
            self.best = {
                "best_epoch": epoch,
                "val_accuracy": value,
                "val_total_loss": val_means.get("total"),
                "per_class_recall": val_metrics["per_class_recall"],
                "single_soliton_recall": val_metrics["single_soliton_recall"],
                "best_model_path": str(self.best_model_path),
            }

    def _early_stop(self, val_means: dict[str, float]) -> bool:
        value = val_means.get("total", float("inf"))  # early_stop_metric == val_total_loss (mode min)
        if value < self._best_es_value - 1e-12:
            self._best_es_value = value
            self._es_counter = 0
        else:
            self._es_counter += 1
        return self._es_counter >= self.early_stop_patience


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
def build_model_config(cfg: SimpleNamespace) -> ModelConfig:
    m = cfg.model
 
    # AFTER (catches BOTH typos AND valid-but-unread fields like context_proj_dim)
    _ALLOWED_MODEL_KEYS = {"gru_hidden","gru_layers","dropout","decoder_hidden",
                           "delta_cmd_max","logvar_min","logvar_max"}   # exactly what build_model_config maps
    _unknown = set(vars(m)) - _ALLOWED_MODEL_KEYS
    if _unknown:
        raise ValueError(f"unknown model config keys (silently ignored otherwise): {sorted(_unknown)}")
     
    return ModelConfig(
        W=int(cfg.data.W),
        H=int(cfg.data.H),
        n_context=N_CONTEXT,
        n_states=N_CLASSES,
        gru_hidden=int(m.gru_hidden),
        gru_layers=int(m.gru_layers),
        dropout=float(m.dropout),
        decoder_hidden=int(m.decoder_hidden),
        delta_cmd_max=float(m.delta_cmd_max),
        logvar_min=float(m.logvar_min),
        logvar_max=float(m.logvar_max),
    )


def build_loss_config(
    cfg: SimpleNamespace, phys: dict[str, Any], horizon_dt: float | None = None
) -> LossConfig:
    """Build a ``LossConfig``, injecting thermal constants from ``tfln_params.yaml``.

    ``Gamma_th``/``tau_th`` come from the physics config. ``horizon_dt`` is data-derived:
    if a pre-resolved value is passed it is used directly (the factory paths do this so the
    provenance can be logged once); otherwise it is resolved here via ``resolve_horizon_dt``
    from ``cfg.loss.horizon_dt`` (explicit override), the dataset metadata, or the fallback.
    """
    l = cfg.loss
    if horizon_dt is None:
        horizon_dt, _ = resolve_horizon_dt(
            getattr(l, "horizon_dt", None), cfg.data.h5_path, phys["fsr_hz"])
    return LossConfig(
        lambda_class_obs=float(l.lambda_class_obs),
        lambda_detune_obs=float(l.lambda_detune_obs),
        lambda_pred_obs=float(l.lambda_pred_obs),
        lambda_thermal_obs=float(l.lambda_thermal_obs),
        lambda_class_ctrl=float(getattr(l, "lambda_class_ctrl", 1.0)),
        lambda_detune_ctrl=float(getattr(l, "lambda_detune_ctrl", 1.0)),
        lambda_effort_ctrl=float(getattr(l, "lambda_effort_ctrl", 0.01)),
        lambda_thermal_ctrl=float(getattr(l, "lambda_thermal_ctrl", 0.0)),
        beta_nll=float(l.beta_nll),
        nll_logvar_floor=float(l.nll_logvar_floor),
        thermal_norm=str(l.thermal_norm),
        Gamma_th=float(phys["Gamma_th"]),
        tau_th=float(phys["tau_th_s"]),
        horizon_dt=float(horizon_dt),
    )


def build_loaders(cfg: SimpleNamespace) -> tuple[Any, Any, Any]:
    d = cfg.data
    return get_dataloaders(
        h5_path=d.h5_path,
        config_path=d.config_path,
        W=int(d.W),
        H=int(d.H),
        stride=int(d.stride),
        batch_size=int(d.batch_size),
        num_workers=int(d.num_workers),
        preload=bool(d.preload),
        max_ram_gb=float(d.max_ram_gb),
        random_state=int(d.random_state),
    )


def build_optimizer(model: torch.nn.Module, cfg: SimpleNamespace) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg.optim.lr),
        weight_decay=float(cfg.optim.weight_decay),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: SimpleNamespace) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup over ``warmup_epochs`` then cosine decay to ~0, stepped once per epoch."""
    epochs = int(cfg.optim.epochs)
    warmup = int(cfg.optim.warmup_epochs)
    warmup = max(0, min(warmup, epochs))
    if warmup <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=0.0)
    if warmup >= epochs:
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0 / warmup, end_factor=1.0, total_iters=warmup)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0 / warmup, end_factor=1.0, total_iters=warmup)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup, eta_min=0.0)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup])


def build_scaler(cfg: SimpleNamespace, loss_cfg: LossConfig) -> LossScaler | None:
    s = getattr(cfg, "scaler", None)
    if s is None or not bool(getattr(s, "enabled", False)):
        return None
    balanced_map = namespace_to_dict(s.balanced_map)
    # Coerce to float defensively: YAML parses unsigned-exponent literals like `1.0e3`
    # as strings, which would crash LossScaler.maybe_rescale's min/max clamp.
    clamp_raw = s.clamp if isinstance(s.clamp, (list, tuple)) else (1e-3, 1e3)
    clamp = (float(clamp_raw[0]), float(clamp_raw[1]))
    return LossScaler(
        loss_cfg,
        reference_key=str(s.reference_key),
        balanced_map=balanced_map,
        period=int(getattr(s, "period", 5)),
        clamp=clamp,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- #
# Phase 1 — observer
# --------------------------------------------------------------------------- #
def train_observer(
    cfg: SimpleNamespace,
    device: torch.device,
    observer_factory: ObserverFactory | None = None,
) -> dict[str, Any]:
    """Train the PIRNNObserver (Phase 1). ``observer_factory`` lets ablations swap the
    model class (e.g. the Transformer-encoder variant) while keeping every other piece
    — loss, loaders, Trainer — identical."""
    phys = _load_config(cfg.data.config_path)
    horizon_dt, provenance = resolve_horizon_dt(
        getattr(cfg.loss, "horizon_dt", None), cfg.data.h5_path, phys["fsr_hz"])
    print(f"[train] horizon_dt = {horizon_dt:.3e} s  (provenance: {provenance})")
    cfg.loss.horizon_dt = horizon_dt                 # record resolved value for config.snapshot.yaml
    cfg.loss.horizon_dt_provenance = provenance
    model_cfg = build_model_config(cfg)
    loss_cfg = build_loss_config(cfg, phys, horizon_dt=horizon_dt)

    factory = observer_factory or PIRNNObserver
    model = factory(model_cfg).to(device)
    loss_fn = PhysicsInformedLoss(loss_cfg)

    train_loader, val_loader, test_loader = build_loaders(cfg)
    class_weights = train_loader.dataset.class_weights.to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = build_scaler(cfg, loss_cfg)  # shares loss_cfg with loss_fn (mutated in place)

    trainer = Trainer(
        model, optimizer, scheduler, loss_fn, train_loader, val_loader, cfg, device,
        mode="observer", observer_frozen=False, class_weights=class_weights, scaler=scaler,
    )
    best = trainer.fit()
    _print_final_summary(cfg, best, phase="observer")
    return {
        "trainer": trainer,
        "best": best,
        "model": model,
        "loss_fn": loss_fn,
        "loaders": (train_loader, val_loader, test_loader),
    }


# --------------------------------------------------------------------------- #
# Phase 2 — controller (loop implemented; data GATED)
# --------------------------------------------------------------------------- #
@runtime_checkable
class SwitchingDataset(Protocol):
    """Expected interface for the Phase-2 switching dataset. **DOES NOT EXIST YET.**

    Each ``__getitem__`` must return a dict with (batched shapes shown):

        x:                              [B, W, 1]   float   — P_trans window (normalized)
        context:                        [B, 4]      float   — operating-point vector
        delta_cmd:                      [B, 1]      float   — applied detuning correction (normalized)
        target_state:                   [B]         long    — operator-specified target soliton state
        target_state_label:             [B]         long    — actual state reached at horizon H under the action
        future_detuning_under_action:   [B, H]      float   — detuning trajectory observed under the action
        valid_transition_mask:          [B]         bool    — True where the (downward) transition is valid/labeled

    See ``train_controller`` for the two binding gates that the dataset must satisfy.
    """

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]: ...


def _switching_contract_message() -> str:
    return (
        "Phase-2 controller training requires a switching/action dataset, which does NOT "
        "exist yet. The observer dataloader (data/dataloader.py) emits no delta_cmd, "
        "target_state, target_state_label, future_detuning_under_action, or "
        "valid_transition_mask, and MUST NOT be reused or faked for Phase 2.\n\n"
        "Set phase2_controller.switching_h5_path and implement a SwitchingDataset whose "
        "batches satisfy the contract in model.train.SwitchingDataset:\n"
        "  x[B,W,1], context[B,4], delta_cmd[B,1], target_state[B] long, "
        "target_state_label[B] long, future_detuning_under_action[B,H], "
        "valid_transition_mask[B] bool.\n\n"
        "Two binding gates must hold before Phase 2 is trustworthy:\n"
        "  (a) closed-loop distribution shift — the observer trains on open-loop sweeps; "
        "add closed-loop/perturbed trajectories to the OBSERVER set before trusting "
        "deployment, or the frozen observer is queried off-distribution at MPC time.\n"
        "  (b) action-space density — act_logits only has a usable ascent gradient if the "
        "switching set contains MULTIPLE delta_cmd magnitudes per starting state "
        "(undershoot/success/overshoot). Point-labeling 'only the action that worked' "
        "trains fine but fails silently at MPC ascent."
    )


def build_switching_loaders(cfg: SimpleNamespace) -> tuple[Any, Any]:
    """Build (train, val) switching loaders. Raises until a SwitchingDataset is implemented."""
    raise NotImplementedError(
        "build_switching_loaders: no SwitchingDataset implementation exists. "
        + _switching_contract_message()
    )


def _load_pretrained_observer(cfg: SimpleNamespace, device: torch.device) -> PIRNNObserver:
    ckpt_path = Path(cfg.phase2_controller.pretrained_observer)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"pretrained observer checkpoint not found: {ckpt_path}. "
            "Run Phase 1 (train_observer) first."
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ModelConfig(**ckpt["model_config"])
    observer = PIRNNObserver(model_cfg)
    observer.load_state_dict(ckpt["model_state_dict"])
    return observer.to(device)


def train_controller(cfg: SimpleNamespace, device: torch.device) -> dict[str, Any]:
    """Train the PIRNNController (Phase 2). Loop is implemented; DATA is gated.

    Binding gates recorded for the next reader (see ``_switching_contract_message``):
      (a) **closed-loop distribution shift** — the observer trains on open-loop sweeps;
          closed-loop / perturbed trajectories must be added to the OBSERVER training set
          before the frozen observer can be trusted at deployment (MPC queries it on the
          closed-loop distribution).
      (b) **action-space density** — ``act_logits`` only has a usable gradient for MPC
          ascent if the switching dataset contains MULTIPLE ``delta_cmd`` magnitudes per
          starting state (undershoot / success / overshoot). A point-labeled set
          ("only the action that worked") trains fine but fails silently at ascent.

    **LossScaler (Phase-2, deferred):**
      ``scaler`` is intentionally ``None`` until both a switching dataset and a
      controller-oriented scaler config exist; the observer scaler does not transfer.
      Required controller-scaler contract when enabled:

        reference_key = ``"class_ctrl"``  — classification is the primary / MPC-ascent
                                            term (FN-1); held fixed as the reference.
        balanced      = ``"detune_ctrl"`` — the supervised forecast term, balanced
                                            against the reference.
        NEVER balanced: ``"effort_ctrl"`` and any ``thermal_*`` term — these regularizers
                        keep their config weights.

      Concrete lambda values and the ``balanced_map`` / ``clamp`` MUST be tuned against
      real switching-dataset loss magnitudes; they cannot be set a priori. The unfrozen
      joint fine-tune (loss emits the observer + controller union) needs a separate
      reference-key decision and is unresolved here.

    Raises ``NotImplementedError`` when no switching dataset is configured.
    """
    observer_h5 = cfg.data.h5_path
    sw_path = getattr(cfg.phase2_controller, "switching_h5_path", None)
    # GUARD (first statement, before any checkpoint load / loader build): the frozen observer's
    # baseline is forecast at the observer dataset's snapshot spacing, while the correction is
    # supervised at the switching dataset's spacing — summing them across different
    # snapshot_interval values would silently mix time bases. Inert today (no switching dataset
    # -> intervals unreadable -> None), arms automatically once a real one is wired in. Must
    # precede build_switching_loaders' NotImplementedError and _load_pretrained_observer.
    assert_matching_snapshot_interval(observer_h5, sw_path)

    if sw_path is None:
        raise NotImplementedError(_switching_contract_message())

    observer = _load_pretrained_observer(cfg, device)
    model_cfg = observer.config
    freeze = bool(cfg.phase2_controller.freeze_observer)
    controller = PIRNNController(model_cfg, observer, freeze_observer=freeze).to(device)

    phys = _load_config(cfg.data.config_path)
    # Derive from the SWITCHING dataset (the data the controller actually trains on); its
    # snapshot spacing governs the controller forecast horizon, not the observer dataset's.
    horizon_dt, provenance = resolve_horizon_dt(
        getattr(cfg.loss, "horizon_dt", None), sw_path, phys["fsr_hz"])
    print(f"[train] horizon_dt = {horizon_dt:.3e} s  (provenance: {provenance})")
    cfg.loss.horizon_dt = horizon_dt                 # record resolved value for config.snapshot.yaml
    cfg.loss.horizon_dt_provenance = provenance
    loss_cfg = build_loss_config(cfg, phys, horizon_dt=horizon_dt)
    loss_fn = PhysicsInformedLoss(loss_cfg)

    # Only requires_grad params (frozen observer excluded by construction).
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, controller.parameters()),
        lr=float(cfg.optim.lr), weight_decay=float(cfg.optim.weight_decay),
    )
    scheduler = build_scheduler(optimizer, cfg)
    # Controller scaler is deferred (Phase-2); the required controller-scaler contract lives
    # in this function's docstring (kept in one place so comment and docstring don't drift).
    scaler = None

    train_loader, val_loader = build_switching_loaders(cfg)  # raises (no SwitchingDataset yet)
    class_weights = getattr(train_loader.dataset, "class_weights", None)
    if class_weights is not None:
        class_weights = class_weights.to(device)

    trainer = Trainer(
        controller, optimizer, scheduler, loss_fn, train_loader, val_loader, cfg, device,
        mode="controller", observer_frozen=freeze, class_weights=class_weights, scaler=scaler,
    )
    best = trainer.fit()
    _print_final_summary(cfg, best, phase="controller")
    return {"trainer": trainer, "best": best, "model": controller, "loss_fn": loss_fn}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _print_final_summary(cfg: SimpleNamespace, best: dict[str, Any], phase: str) -> None:
    print("\n" + "=" * 60)
    print(f"FINAL SUMMARY — {cfg.experiment.name} (phase={phase})")
    print("=" * 60)
    if not best:
        print("No best epoch recorded (training did not complete a validation pass).")
        return
    print(f"best epoch          : {best.get('best_epoch')}")
    print(f"best val_accuracy   : {best.get('val_accuracy'):.4f}")
    bvl = best.get("val_total_loss")
    print(f"val_total_loss@best : {bvl:.4f}" if bvl is not None else "val_total_loss@best : n/a")
    print("per-class recall    :")
    from data.dataloader import STATE_NAMES  # local import to avoid any import-order coupling
    for c, r in enumerate(best.get("per_class_recall", [])):
        name = STATE_NAMES.get(c, str(c))
        rstr = f"{r:.3f}" if r is not None else "n/a (no support)"
        star = "  <-- figure of merit" if c == SINGLE_SOLITON_IDX else ""
        print(f"    [{c}] {name:<16}: {rstr}{star}")
    print(f"best_model.pt       : {best.get('best_model_path')}")
    print("=" * 60 + "\n")


def run_training(
    config_path: str | Path,
    experiment_name: str | None = None,
    overrides: dict[str, Any] | None = None,
    *,
    phase: str = "observer",
    device: str | None = None,
) -> dict[str, Any]:
    """Load config, resolve device, build everything, and run a single training phase.

    ``phase`` and ``device`` are keyword-only extensions used by the CLI; the documented
    positional contract remains ``run_training(config_path, experiment_name, overrides)``.
    """
    cfg_dict = load_yaml(config_path)
    apply_overrides(cfg_dict, overrides)
    if experiment_name is not None:
        cfg_dict.setdefault("experiment", {})["name"] = experiment_name
    cfg = to_namespace(cfg_dict)

    dev = resolve_device(device or getattr(cfg.train, "device", "auto"))
    print(f"[train] device = {dev}")
    set_seed(int(cfg.experiment.seed))

    h5_path = Path(cfg.data.h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(
            f"dataset not found: {h5_path}. Generate it first with "
            "`python -m data.dataset_generator` (see data/dataset_generator.py)."
        )

    if phase == "observer":
        return train_observer(cfg, dev)
    if phase == "controller":
        return train_controller(cfg, dev)
    raise ValueError(f"phase must be 'observer' or 'controller', got {phase!r}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train the TFLN soliton-control PI-RNN.")
    ap.add_argument("--config", default="config/training_config.yaml")
    ap.add_argument("--experiment_name", default=None)
    ap.add_argument("--phase", choices=["observer", "controller"], default="observer")
    ap.add_argument("--ablation",
                    choices=["full", "no_physics_loss", "transformer_backbone", "si3n4_pretrain"],
                    default=None)
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--override", nargs="*", default=[],
                    help="dotted-key overrides, e.g. optim.lr=1e-4 loss.lambda_thermal_obs=0.0")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    overrides: dict[str, Any] = {}
    for item in args.override:
        if "=" not in item:
            raise ValueError(f"--override expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        overrides[key] = parse_override_value(raw)
    if args.epochs is not None:
        overrides["optim.epochs"] = args.epochs

    if args.ablation is not None:
        from analysis.ablations import run_ablation  # lazy import avoids a circular dependency
        run_ablation(
            args.ablation,
            base_config_path=args.config,
            overrides=overrides,
            device=args.device,
            experiment_name=args.experiment_name,
        )
        return

    run_training(
        args.config,
        experiment_name=args.experiment_name,
        overrides=overrides,
        phase=args.phase,
        device=args.device,
    )


if __name__ == "__main__":
    main()
