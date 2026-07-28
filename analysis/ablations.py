"""Ablation study orchestration for the PI-RNN observer.

``run_ablation`` is the single entry point. It imports the factory functions and the
``train_observer`` driver from ``model/train.py`` and only varies one axis per run; the
Trainer, loss, and (where possible) data are otherwise identical to the ``full`` baseline.
``model/train.py``'s ``--ablation`` CLI dispatches here via a lazy import, so there is no
circular dependency (this module imports ``model.train`` at top level; ``model.train``
imports this module only inside its ``main()``).

Four ablations, each writing to ``runs/ablation_{name}/``:

  full                  baseline observer run (no change).
  no_physics_loss       lambda_thermal_obs = 0.0 (and lambda_thermal_ctrl = 0.0). NOTE:
                        thermal is the ONLY physics term; there is no energy-balance term
                        and no lambda2/lambda3 to remove.
  transformer_backbone  swaps the GRU encoder for a Transformer encoder via the
                        observer_factory hook (see model/ablation_encoders.py). pi_rnn.py
                        is untouched; the output-dict contract is preserved verbatim.
  si3n4_pretrain        DATA-side: trains on separately generated Si3N4 loaders, then
                        zero-shot evaluates the frozen model on the TFLN test split.
                        Raises FileNotFoundError if the Si3N4 config/dataset is absent.

Out of scope here: the closed-loop MPC access-rate study ("Ablation 4") belongs to
``mpc/`` + the simulator, not the trainer.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from model.ablation_encoders import PIRNNObserverTransformer
from model.train import (
    apply_overrides,
    build_loaders,
    load_yaml,
    resolve_device,
    set_seed,
    to_namespace,
    train_observer,
)

ABLATIONS = ("full", "no_physics_loss", "transformer_backbone", "si3n4_pretrain")

SI3N4_CONFIG_PATH = "config/si3n4_params.yaml"
SI3N4_H5_PATH = "data/synthetic/dataset_si3n4.h5"


def _jsonable(obj: Any) -> Any:
    """Recursively coerce a metrics payload into something ``json.dumps`` accepts.

    ``Trainer.evaluate`` returns plain Python floats today, but the accumulators it feeds
    from hold torch tensors / numpy scalars, so coerce defensively rather than letting a
    stray tensor turn the zero-shot artifact into a ``TypeError`` at the end of a long run.
    ``None`` is preserved (``per_class_recall`` uses it for "no support", which must stay
    distinguishable from 0.0 recall).
    """
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if obj is None or isinstance(obj, (bool, str, int, float)):
        return obj
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


def _require_file(path: str | Path, what: str) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{what} not found: {p}. This ablation must NOT be faked with TFLN data — "
            "generate the required artifact first."
        )


def _prepare_cfg(base_dict: dict[str, Any], name: str, overrides: dict[str, Any] | None,
                 mutations: dict[str, Any] | None):
    cfg_dict = copy.deepcopy(base_dict)
    apply_overrides(cfg_dict, overrides)       # caller-supplied overrides first
    apply_overrides(cfg_dict, mutations)       # then the ablation-specific mutations
    cfg_dict.setdefault("experiment", {})["name"] = f"ablation_{name}"
    return to_namespace(cfg_dict)


def run_ablation(
    name: str,
    base_config_path: str | Path = "config/training_config.yaml",
    overrides: dict[str, Any] | None = None,
    device: str | None = None,
    experiment_name: str | None = None,  # accepted for CLI symmetry; ablation dir is canonical
) -> dict[str, Any]:
    """Run one named ablation. Returns the ``train_observer`` result dict (plus, for
    ``si3n4_pretrain``, a ``zero_shot_tfln`` entry with the frozen TFLN-test metrics)."""
    if name not in ABLATIONS:
        raise ValueError(f"unknown ablation {name!r}; choose from {ABLATIONS}")

    base_dict = load_yaml(base_config_path)
    dev = resolve_device(device or base_dict.get("train", {}).get("device", "auto"))

    if name == "full":
        cfg = _prepare_cfg(base_dict, name, overrides, mutations=None)
        set_seed(int(cfg.experiment.seed))
        _require_file(cfg.data.h5_path, "TFLN dataset")
        return train_observer(cfg, dev)

    if name == "no_physics_loss":
        # Thermal is the ONLY physics term. Do NOT invent lambda2/lambda3.
        mutations = {"loss.lambda_thermal_obs": 0.0, "loss.lambda_thermal_ctrl": 0.0}
        cfg = _prepare_cfg(base_dict, name, overrides, mutations)
        set_seed(int(cfg.experiment.seed))
        _require_file(cfg.data.h5_path, "TFLN dataset")
        return train_observer(cfg, dev)

    if name == "transformer_backbone":
        cfg = _prepare_cfg(base_dict, name, overrides, mutations=None)
        set_seed(int(cfg.experiment.seed))
        _require_file(cfg.data.h5_path, "TFLN dataset")
        # Same loss/loaders/Trainer; only the x-encoder differs.
        return train_observer(cfg, dev, observer_factory=PIRNNObserverTransformer)

    # si3n4_pretrain: train on Si3N4, zero-shot eval on the TFLN test split.
    _require_file(SI3N4_CONFIG_PATH, "Si3N4 physics config")
    _require_file(SI3N4_H5_PATH, "Si3N4 dataset")

    mutations = {"data.h5_path": SI3N4_H5_PATH, "data.config_path": SI3N4_CONFIG_PATH}
    cfg = _prepare_cfg(base_dict, name, overrides, mutations)
    set_seed(int(cfg.experiment.seed))
    result = train_observer(cfg, dev)

    # Build the TFLN test loader from the UNMODIFIED base data section and evaluate the
    # frozen pretrained model on it (zero-shot transfer).
    tfln_cfg = _prepare_cfg(base_dict, name, overrides, mutations=None)
    _require_file(tfln_cfg.data.h5_path, "TFLN dataset")
    _, _, tfln_test_loader = build_loaders(tfln_cfg)
    zero_shot = result["trainer"].evaluate(tfln_test_loader)
    result["zero_shot_tfln"] = zero_shot

    # Persist it: the zero-shot transfer numbers ARE this ablation's deliverable, and the
    # run directory otherwise only holds the Si3N4 *training* metrics.jsonl. Trainer.evaluate
    # returns {"loss": ..., "metrics": ..., "diag": ...}; write the whole envelope.
    zero_shot_dir = Path(cfg.experiment.out_dir) / cfg.experiment.name
    zero_shot_dir.mkdir(parents=True, exist_ok=True)
    zero_shot_path = zero_shot_dir / "zero_shot_tfln.json"
    zero_shot_path.write_text(json.dumps(_jsonable(zero_shot), indent=2), encoding="utf-8")

    metrics = zero_shot["metrics"]
    ss = metrics["single_soliton_recall"]
    print("\n[si3n4_pretrain] zero-shot on TFLN test split:")
    print(f"    accuracy            : {metrics['accuracy']:.4f}")
    print(f"    single_soliton_recall: {ss:.4f}" if ss is not None else
          "    single_soliton_recall: n/a (no support)")
    return result
