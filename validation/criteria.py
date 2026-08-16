"""Convergence-aware acceptance criteria for the cross-code comparison.

WHY THE PREVIOUS CRITERIA WERE REPLACED
---------------------------------------
The v1 criteria (preserved below as :data:`TOLERANCES_V1_HISTORICAL`) were five
relative tolerances plus one "exact integer match". Every one of them was
defective, and the defects are *measured*, not asserted -- see
``docs/CONVERGENCE_LLE.md`` and ``docs/VALIDATION_METHODOLOGY.md``:

* **exact integer match on the DW index** is a discontinuous criterion on a
  continuous quantity. The measured centroid disagreement was 0.65 modes out of
  3075 (0.02%) and it failed only because the two values straddled ``x.5``. Any
  sub-mode disagreement has roughly a coin-flip chance of failing.
* **2% on the 3 dB span** was applied to an INTEGER extent whose error floor is
  ``1/span``. At span 443 one quantum is 0.226%, so the v1 "0.23% pass" was
  exactly one quantum -- the smallest non-zero value the metric can express. At
  span 64 the same quantum is 1.6% and identical solution quality fails. The
  verdict tracked the span's magnitude, not the accuracy.
* **2% on peak power** was applied to raw ``max|E_j|^2`` on a grid where the
  soliton FWHM is 4.34 samples; the sampling-phase bound alone is 4.0%.
* **2% on the -60 dBc line count** was applied to a level-crossing count with a
  measured conditioning of 85-121 lines per dB, whose core (|mu| <= 1500) is
  IDENTICAL between the two codes (3001 = 3001) at every refinement level. It
  measured where a nearly-flat wing crosses an arbitrary line.
* **2% is tighter than either code's own discretization uncertainty** at the
  shared one-step-per-round-trip discretization.

THE REPLACEMENT, IN ONE SENTENCE
--------------------------------
Criteria are split into three classes; the quantitative ones are applied only to
well-conditioned, continuous functionals; and their tolerances are **derived
from measured convergence studies rather than chosen**.

* ``HARD`` -- convention, bookkeeping and exact algebraic identities. A failure
  invalidates the comparison outright.
* ``GATED`` -- quantitative agreement, tolerance derived from the two codes'
  measured numerical uncertainties. A failure is a real failure.
* ``DIAGNOSTIC`` -- reported with a value and a conditioning number, and
  structurally unable to affect the overall verdict.

``overall`` is PASS iff no HARD or GATED check is FAIL or NOT_MEASURED.
NOT_MEASURED on a HARD or GATED check is a failure, never a silent pass.

TOLERANCE DERIVATION
--------------------
::

    tol(obs) = clip(K * sqrt(U_ours^2 + U_pylle^2), floor(obs), 0.25)

with ``K = COVERAGE_FACTOR = 2.0``. ``U_ours`` comes from Prompt B's
``convergence_lle_dw30k.json`` at the discretization actually used, ``U_pylle``
from Prompt D's ``pylle_refinement_dw30k.json``. **Both** studies are consumed
for every GATED check, so no criterion treats either code as ground truth.

If either uncertainty is unavailable -- the underlying convergence study is
``NON_MONOTONE``, ``ORDER_OUT_OF_RANGE`` or missing -- the tolerance is marked
``UNDERIVED`` and the check is demoted to DIAGNOSTIC for that run, with a loud
warning. It never falls back to a hand-picked number: an undefendable tolerance
is worse than no tolerance.

A ``tolerance_fingerprint`` (sha256 over the name -> tolerance mapping) is taken
at derivation, **before any comparison value is read**, and re-checked at
evaluation. Any change between the two raises, which is what makes "no
post-hoc tolerance tuning" a mechanical guarantee rather than a promise.

WHAT THIS MODULE DELIBERATELY DOES NOT TOUCH
--------------------------------------------
``validation/analytic_cw.py`` gates at 1e-12 against exact mathematics, and
``mms.py``/``convergence.py`` check observed order of accuracy. Those are
mathematical-verification tests, the criterion style is correct for them, and
they stay strict and untouched. Nothing here loosens them.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "validation" / "results"
SCHEMA_PATH = REPO_ROOT / "validation" / "schemas" / "crosscheck_v2.schema.json"

CRITERIA_VERSION = "2.0"
SCHEMA_VERSION = "2.0"

# --------------------------------------------------------------------------
# The ONLY numeric literals permitted in this module (acceptance criterion A2):
# the coverage factor, the floor tolerances, and the HARD thresholds. No GATED
# tolerance is ever written down -- each is computed in derive_tolerances().
# --------------------------------------------------------------------------
COVERAGE_FACTOR = 2.0        # two-sigma-style coverage on the combined uncertainty
MAX_TOL = 0.25               # nothing above this is a meaningful agreement test

FLOOR_TOL_RELATIVE = {       # relative floors, per GATED criterion id
    "G1": 1e-6,
    "G2": 1e-6,
    "G3": 1e-4,
    "G4": 1e-4,
    "G6": 1e-4,
}
FLOOR_TOL_ABSOLUTE_MODES = {"G5": 0.05}      # G5 is absolute, in mode numbers

HARD_THRESHOLDS = {
    "H1": 1e-12,   # parameter round trip
    "H2": 1e-6,    # dispersion refit
    "H5": 1e-12,   # detuning endpoint match
    "H6": 1e-6,    # delta_omega_eff == programmed
}

# --------------------------------------------------------------------------
# SUPERSEDED. Kept verbatim so the repository retains what the v1 comparison was
# judged against. Not used by any code path here; see the module docstring for
# the measured defect behind each entry.
# --------------------------------------------------------------------------
TOLERANCES_V1_HISTORICAL = {
    "dw_peak_mode_indices": 0.0,        # EXACT integer match
    "spectral_span_3db": 0.02,
    "soliton_peak_power": 0.02,
    "existence_range_edges": 0.05,
    "comb_line_count_60dbc": 0.02,
}


class CriterionClass(str, Enum):
    HARD = "HARD"
    GATED = "GATED"
    DIAGNOSTIC = "DIAGNOSTIC"


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_MEASURED = "NOT_MEASURED"
    UNDERIVED = "UNDERIVED"
    DIAGNOSTIC = "DIAGNOSTIC"


VERDICTS = {v.value for v in Verdict}


class CriteriaError(RuntimeError):
    """Raised on a fingerprint mismatch or a malformed report."""


# ==========================================================================
# Criterion table -- DATA, not code paths (acceptance criterion A1)
# ==========================================================================
@dataclass(frozen=True)
class Criterion:
    cid: str
    name: str
    cls: CriterionClass
    quantity: str
    comparison: str            # "threshold" | "bool" | "relative" | "absolute"
                               # | "bracket" | "report"
    ours_key: str | None = None
    pylle_key: str | None = None
    conv_key: str | None = None    # key in BOTH convergence studies
    units: str = "-"
    note: str = ""


CRITERIA: tuple[Criterion, ...] = (
    # ---------------- HARD ----------------
    Criterion("H1", "parameter_round_trip_max_rel", CriterionClass.HARD,
              "ours -> pyLLE -> ours round trip of every scalar", "threshold",
              units="relative",
              note="exact algebraic inversion; anything above 1e-12 is a translation bug"),
    Criterion("H2", "dispersion_refit_max_rel", CriterionClass.HARD,
              "pyLLE's spline-refit D_int vs ours, after un-mirroring", "threshold",
              units="relative",
              note="both codes must integrate the same dispersion array"),
    Criterion("H3", "dispersion_mirror_applied", CriterionClass.HARD,
              "D_int handed to pyLLE is mirrored mu -> -mu", "bool",
              note="forced by the conjugate field map; omitting it produces two "
                   "plausible solitons that disagree only on asymmetric observables"),
    Criterion("H4", "pump_mode_reference_matches", CriterionClass.HARD,
              "|f_pmp_used - csv_mu0_resonance|", "threshold", units="Hz",
              note="our D_int is referenced to CSV mu=0, which is 7.11 FSR from the "
                   "nominal c/lambda; pyLLE locates the pump by frequency"),
    Criterion("H5", "detuning_endpoint_match", CriterionClass.HARD,
              "|dw_final_ours - (-dw_final_pylle)| / |dw_final_ours|", "threshold",
              units="relative",
              note="upstream pyLLE's probe cadence returned the field at round trip "
                   "4976 of 5000 (29.9328 kappa vs our 30.0000 kappa). Prompt D's "
                   "patch 0003 fixes it; this gate proves it on every run"),
    Criterion("H6", "delta_omega_eff_equals_programmed", CriterionClass.HARD,
              "max|delta_omega_eff - programmed ramp| / max|ramp|", "threshold",
              units="relative",
              note="proves our thermo-optic shift is off, so both codes sit at the "
                   "same detuning"),
    Criterion("H7", "seed_arrays_identical", CriterionClass.HARD,
              "sha256 of each code's seed after the documented conjugate map", "bool",
              note="both codes must start from the same physical state"),

    # ---------------- GATED ----------------
    Criterion("G1", "intracavity_power_U_mean", CriterionClass.GATED,
              "sum|E_j|^2/N/t_r  (Parseval-exact mean intracavity power)", "relative",
              ours_key="U_mean_w", pylle_key="U_mean_w", conv_key="U_mean_w",
              units="W", note="translation- and sampling-invariant"),
    Criterion("G2", "comb_energy_fraction", CriterionClass.GATED,
              "(sum|E_mu|^2 - |E_0|^2)/sum|E_mu|^2", "relative",
              ours_key="comb_frac", pylle_key="comb_frac", conv_key="comb_frac",
              units="-", note="translation- and sampling-invariant"),
    Criterion("G3", "band_limited_peak_power", CriterionClass.GATED,
              "max|E_up|^2/t_r on a 32x zero-padded FFT", "relative",
              ours_key="P_peak_w", pylle_key="P_peak_w", conv_key="P_peak_w",
              units="W", note="removes the 4.0% sampling-phase bias of the raw peak"),
    Criterion("G4", "subbin_3db_span", CriterionClass.GATED,
              "3 dB envelope span by dB-linear level crossing", "relative",
              ours_key="S3_modes", pylle_key="S3_modes", conv_key="S3_modes",
              units="modes", note="removes the 1/span quantization floor"),
    Criterion("G5", "dw_centroid_fixed_band", CriterionClass.GATED,
              "pedestal-subtracted power centroid over a fixed mu band", "absolute",
              ours_key="mu_DW", pylle_key="mu_DW", conv_key="mu_DW",
              units="modes",
              note="continuous functional of the spectrum; two independent "
                   "discretizations cannot be required to round to the same integer, "
                   "but can be required to agree within their combined uncertainty"),
    Criterion("G6", "dw_band_power_dbc", CriterionClass.GATED,
              "10*log10(sum p over DW band / max p)", "relative",
              ours_key="dw_power_dbc", pylle_key="dw_power_dbc",
              conv_key="dw_power_dbc", units="dBc",
              note="the DW AMPLITUDE -- measured by v1 and never compared, and it "
                   "moves 7.1 dB across our own refinement ladder"),
    Criterion("G7", "existence_edges", CriterionClass.GATED,
              "single-soliton existence bracket, lower and upper edge", "bracket",
              units="kappa",
              note="brackets, not the biased tightest-surviving point v1 used"),

    # ---------------- DIAGNOSTIC ----------------
    Criterion("D1", "comb_line_count_60dbc", CriterionClass.DIAGNOSTIC,
              "count of modes >= -60 dBc, with dN60/ddB and core/mid/edge split",
              "report", units="lines",
              note="conditioning 85-121 lines per dB; core identical (3001) between "
                   "codes at every level -- it measures the wing crossing"),
    Criterion("D2", "integer_3db_span", CriterionClass.DIAGNOSTIC,
              "legacy integer 3 dB extent and its quantum 1/span", "report",
              units="modes"),
    Criterion("D3", "integer_dw_argmax", CriterionClass.DIAGNOSTIC,
              "raw argmax mode index on each side", "report", units="modes"),
    Criterion("D4", "raw_peak_power", CriterionClass.DIAGNOSTIC,
              "legacy max|E_j|^2/t_r and the sub-sample offset", "report", units="W"),
    Criterion("D5", "spectral_residual_by_band", CriterionClass.DIAGNOSTIC,
              "median |dB| residual per mu band", "report", units="dB"),
    Criterion("D6", "spectral_edge_dbc", CriterionClass.DIAGNOSTIC,
              "spectrum at mu = +/- mu_half; warn if above -100 dBc", "report",
              units="dBc",
              note="the committed run sits at -52.2 dBc: the comb is not contained"),
    Criterion("D7", "dint_phase_budget", CriterionClass.DIAGNOSTIC,
              "modes with |D_int|*t_r > pi and the smallest such |mu|", "report",
              units="modes"),
)

CRITERIA_BY_ID = {c.cid: c for c in CRITERIA}
HARD_IDS = tuple(c.cid for c in CRITERIA if c.cls is CriterionClass.HARD)
GATED_IDS = tuple(c.cid for c in CRITERIA if c.cls is CriterionClass.GATED)
DIAGNOSTIC_IDS = tuple(c.cid for c in CRITERIA if c.cls is CriterionClass.DIAGNOSTIC)

# GATED ids whose tolerance is derived from a convergence pair (G7 is a bracket
# test and derives its own scale from the bracket half-widths).
DERIVED_GATED_IDS = tuple(c.cid for c in CRITERIA
                          if c.cls is CriterionClass.GATED and c.conv_key)


# ==========================================================================
# Numerical helpers
# ==========================================================================
def rel(a: float, b: float) -> float:
    """|a-b| / max(|a|,|b|), and exactly 0 when both are 0."""
    a, b = float(a), float(b)
    denom = max(abs(a), abs(b))
    return 0.0 if denom == 0.0 else abs(a - b) / denom


def combine(u_a: float, u_b: float, k: float = COVERAGE_FACTOR) -> float:
    """Coverage-scaled quadrature sum of two uncertainties."""
    return k * math.sqrt(float(u_a) ** 2 + float(u_b) ** 2)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


# ==========================================================================
# Tolerance derivation
# ==========================================================================
@dataclass
class DerivedTolerance:
    cid: str
    name: str
    tolerance: float | None
    status: str                     # "DERIVED" | "UNDERIVED"
    kind: str                       # "relative" | "absolute"
    u_ours: float | None = None
    u_pylle: float | None = None
    coverage_factor: float = COVERAGE_FACTOR
    floor: float | None = None
    reason: str = ""
    source_ours: dict[str, Any] = field(default_factory=dict)
    source_pylle: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToleranceSet:
    entries: dict[str, DerivedTolerance]
    fingerprint: str
    provenance: dict[str, Any]

    def tol(self, cid: str) -> float | None:
        return self.entries[cid].tolerance

    def is_derived(self, cid: str) -> bool:
        return self.entries[cid].status == "DERIVED"


def _fingerprint(entries: dict[str, DerivedTolerance]) -> str:
    """sha256 over the (criterion -> tolerance) mapping ONLY.

    Deliberately excludes provenance and uncertainties: what must not change
    between derivation and evaluation is the numbers the verdicts are measured
    against.
    """
    payload = json.dumps(
        {cid: [e.status, None if e.tolerance is None else repr(float(e.tolerance))]
         for cid, e in sorted(entries.items())},
        sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _u_from_ours(conv_json: dict, conv_key: str, level_key: str) -> tuple[float | None, dict]:
    """Uncertainty for one observable from Prompt B's convergence study.

    ``level_key`` selects the discretization actually used by the comparison
    (``numerical_uncertainty_at_n1`` or ``..._at_n8``). The value is only
    accepted when the underlying convergence entry converged: a NON_MONOTONE or
    ORDER_OUT_OF_RANGE study has no defensible uncertainty, only a fallback
    band, and R3 requires that to surface as UNDERIVED.
    """
    src = {"file": "convergence_lle_dw30k.json", "key": conv_key,
           "level_key": level_key}
    conv = (conv_json.get("convergence") or {}).get(conv_key)
    if conv is None:
        return None, dict(src, reason="observable absent from the convergence study")
    src["convergence_status"] = conv.get("status")
    if conv.get("status") != "OK":
        return None, dict(src, reason=f"convergence status {conv.get('status')}")
    block = conv_json.get(level_key) or {}
    if conv_key not in block:
        return None, dict(src, reason=f"{level_key} has no entry for {conv_key}")
    val = block[conv_key]
    if val is None:
        return None, dict(src, reason="uncertainty is null")
    src["discretization"] = level_key.replace("numerical_uncertainty_at_", "n_substeps=")
    return float(val), src


def _u_from_pylle(ref_json: dict, conv_key: str, tag: str) -> tuple[float | None, dict]:
    """Uncertainty for one observable from Prompt D's pyLLE refinement study."""
    full = f"{tag}:{conv_key}"
    src = {"file": "pylle_refinement_dw30k.json", "key": full}
    conv = (ref_json.get("convergence") or {}).get(full)
    if conv is None:
        return None, dict(src, reason="observable absent from the refinement study")
    src["convergence_status"] = conv.get("status")
    if conv.get("status") != "OK":
        return None, dict(src, reason=f"convergence status {conv.get('status')}")
    val = conv.get("U_obs")
    if val is None:
        return None, dict(src, reason="U_obs is null")
    src["discretization"] = f"dt ladder, tol tag '{tag}'"
    return float(val), src


def derive_tolerances(ours_path: Path | str | None = None,
                      pylle_path: Path | str | None = None,
                      *,
                      ours_level: str = "numerical_uncertainty_at_n1",
                      pylle_tag: str = "tight",
                      argv: list[str] | None = None) -> ToleranceSet:
    """Derive every GATED tolerance from the two measured convergence studies.

    Called BEFORE any comparison value is read; the returned fingerprint is what
    :func:`evaluate` re-checks.
    """
    ours_path = Path(ours_path or RESULTS_DIR / "convergence_lle_dw30k.json")
    pylle_path = Path(pylle_path or RESULTS_DIR / "pylle_refinement_dw30k.json")
    ours_json = json.loads(Path(ours_path).read_text())
    pylle_json = json.loads(Path(pylle_path).read_text())

    entries: dict[str, DerivedTolerance] = {}
    for cid in DERIVED_GATED_IDS:
        c = CRITERIA_BY_ID[cid]
        absolute = cid in FLOOR_TOL_ABSOLUTE_MODES
        floor = (FLOOR_TOL_ABSOLUTE_MODES[cid] if absolute
                 else FLOOR_TOL_RELATIVE[cid])
        u_o, src_o = _u_from_ours(ours_json, c.conv_key, ours_level)
        u_p, src_p = _u_from_pylle(pylle_json, c.conv_key, pylle_tag)

        if u_o is None or u_p is None:
            why = []
            if u_o is None:
                why.append(f"ours: {src_o.get('reason')}")
            if u_p is None:
                why.append(f"pyLLE: {src_p.get('reason')}")
            entries[cid] = DerivedTolerance(
                cid, c.name, None, "UNDERIVED",
                "absolute" if absolute else "relative",
                u_o, u_p, COVERAGE_FACTOR, floor,
                "; ".join(why), src_o, src_p)
            continue

        if absolute:
            # G5: uncertainties are RELATIVE in the source studies; convert to
            # modes using the magnitude the study itself extrapolated to, so the
            # criterion is expressed in the same units as the quantity.
            scale_o = abs(float(
                (ours_json["convergence"][c.conv_key] or {}).get("extrapolated") or 0.0))
            scale_p = abs(float(
                (pylle_json["convergence"][f"{pylle_tag}:{c.conv_key}"] or {})
                .get("extrapolated") or 0.0))
            u_o_abs, u_p_abs = u_o * scale_o, u_p * scale_p
            raw = combine(u_o_abs, u_p_abs)
            src_o = dict(src_o, absolute_scale=scale_o, u_absolute=u_o_abs)
            src_p = dict(src_p, absolute_scale=scale_p, u_absolute=u_p_abs)
        else:
            raw = combine(u_o, u_p)

        # MAX_TOL is a ceiling on RELATIVE agreement: a "tolerance" above 25%
        # is not an agreement test. It is meaningless for an ABSOLUTE tolerance
        # measured in mode numbers -- clipping there would silently TIGHTEN the
        # criterion below what the uncertainties support (0.53 modes -> 0.25),
        # which is the opposite of a safety cap. R4 defines G5 as
        # max(floor, K*sqrt(U_o^2+U_p^2)) with no ceiling, and that is what is
        # applied here.
        tol = max(raw, floor) if absolute else min(max(raw, floor), MAX_TOL)
        entries[cid] = DerivedTolerance(
            cid, c.name, float(tol), "DERIVED",
            "absolute" if absolute else "relative",
            u_o, u_p, COVERAGE_FACTOR, floor,
            f"clip({COVERAGE_FACTOR}*sqrt(U_o^2+U_p^2)={raw:.6g}, "
            f"floor={floor:g}, hi={MAX_TOL:g})",
            src_o, src_p)

    fp = _fingerprint(entries)
    prov = {
        "criteria_version": CRITERIA_VERSION,
        "coverage_factor": COVERAGE_FACTOR,
        "floor_tolerances_relative": dict(FLOOR_TOL_RELATIVE),
        "floor_tolerances_absolute_modes": dict(FLOOR_TOL_ABSOLUTE_MODES),
        "max_tolerance": MAX_TOL,
        "hard_thresholds": dict(HARD_THRESHOLDS),
        "sources": {
            "ours": {"path": str(ours_path), "sha256": _sha_file(ours_path),
                     "level_key": ours_level},
            "pylle": {"path": str(pylle_path), "sha256": _sha_file(pylle_path),
                      "tag": pylle_tag},
        },
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "argv": list(argv if argv is not None else sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    return ToleranceSet(entries, fp, prov)


# ==========================================================================
# Evaluation
# ==========================================================================
def _result(cid: str, verdict: Verdict, **kw) -> dict:
    c = CRITERIA_BY_ID[cid]
    out = {"cid": cid, "name": c.name, "class": c.cls.value,
           "quantity": c.quantity, "verdict": verdict.value}
    out.update(kw)
    return out


def evaluate_hard(values: dict) -> list[dict]:
    """Evaluate the HARD checks from a flat dict of measured values."""
    res = []

    def thresh(cid, key):
        v = values.get(key)
        lim = HARD_THRESHOLDS[cid] if cid in HARD_THRESHOLDS else 0.0
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return _result(cid, Verdict.NOT_MEASURED, value=v, threshold=lim)
        ok = float(v) <= lim
        return _result(cid, Verdict.PASS if ok else Verdict.FAIL,
                       value=float(v), threshold=lim)

    res.append(thresh("H1", "parameter_round_trip_max_rel"))
    res.append(thresh("H2", "dispersion_refit_max_rel"))

    for cid, key in (("H3", "dispersion_mirror_applied"),
                     ("H7", "seed_arrays_identical")):
        v = values.get(key)
        if v is None:
            res.append(_result(cid, Verdict.NOT_MEASURED, value=None))
        else:
            res.append(_result(cid, Verdict.PASS if bool(v) else Verdict.FAIL,
                               value=bool(v)))

    # H4: an exact identity -- the pump used must BE the CSV mu=0 resonance.
    v = values.get("pump_mode_reference_delta_hz")
    if v is None:
        res.append(_result("H4", Verdict.NOT_MEASURED, value=None, threshold=0.0))
    else:
        res.append(_result("H4", Verdict.PASS if float(v) == 0.0 else Verdict.FAIL,
                           value=float(v), threshold=0.0))

    res.append(thresh("H5", "detuning_endpoint_match_rel"))
    res.append(thresh("H6", "delta_omega_eff_max_rel"))
    return res


def evaluate_gated(values: dict, tolset: ToleranceSet) -> list[dict]:
    """Evaluate the GATED checks. UNDERIVED tolerances demote to DIAGNOSTIC."""
    res = []
    for cid in DERIVED_GATED_IDS:
        c = CRITERIA_BY_ID[cid]
        ent = tolset.entries[cid]
        o = values.get(f"{c.ours_key}__ours")
        p = values.get(f"{c.pylle_key}__pylle")
        common = {"ours": o, "pylle": p,
                  "u_ours": ent.u_ours, "u_pylle": ent.u_pylle,
                  "coverage_factor": ent.coverage_factor,
                  "tolerance": ent.tolerance, "tolerance_kind": ent.kind,
                  "tolerance_status": ent.status,
                  "source_ours": ent.source_ours, "source_pylle": ent.source_pylle}

        if ent.status == "UNDERIVED":
            d = None if (o is None or p is None) else (
                abs(float(o) - float(p)) if ent.kind == "absolute"
                else rel(float(o), float(p)))
            res.append(_result(cid, Verdict.UNDERIVED, diff=d,
                               demoted_to="DIAGNOSTIC", reason=ent.reason, **common))
            continue

        if o is None or p is None or (isinstance(o, float) and math.isnan(o)) \
                or (isinstance(p, float) and math.isnan(p)):
            res.append(_result(cid, Verdict.NOT_MEASURED, diff=None, **common))
            continue

        d = (abs(float(o) - float(p)) if ent.kind == "absolute"
             else rel(float(o), float(p)))
        res.append(_result(cid, Verdict.PASS if d <= ent.tolerance else Verdict.FAIL,
                           diff=d, **common))
    return res


def evaluate_existence_edges(values: dict, tolset: ToleranceSet) -> list[dict]:
    """G7: brackets, not points.

    PASSES iff the two brackets OVERLAP, or their midpoints agree within
    ``K*sqrt(hw_o^2 + hw_p^2 + U_o^2 + U_p^2)``. v1 compared ``br[1]`` -- the
    tightest surviving point -- on both sides, which is a biased estimator: it
    reports the edge as the last point that happened to survive rather than as
    the interval the bisection actually resolved.
    """
    out = []
    for edge in ("lower", "upper"):
        bo = values.get(f"existence_{edge}_bracket_ours")
        bp = values.get(f"existence_{edge}_bracket_pylle")
        u_o = values.get(f"existence_{edge}_u_ours", 0.0) or 0.0
        u_p = values.get(f"existence_{edge}_u_pylle", 0.0) or 0.0
        base = {"edge": edge, "bracket_ours": bo, "bracket_pylle": bp,
                "u_ours": u_o, "u_pylle": u_p,
                "coverage_factor": COVERAGE_FACTOR}
        if not bo or not bp or len(bo) != 2 or len(bp) != 2:
            out.append(_result("G7", Verdict.NOT_MEASURED, **base))
            continue
        lo_o, hi_o = sorted(float(x) for x in bo)
        lo_p, hi_p = sorted(float(x) for x in bp)
        mid_o, mid_p = 0.5 * (lo_o + hi_o), 0.5 * (lo_p + hi_p)
        hw_o, hw_p = 0.5 * (hi_o - lo_o), 0.5 * (hi_p - lo_p)
        overlap = (lo_o <= hi_p) and (lo_p <= hi_o)
        tol = COVERAGE_FACTOR * math.sqrt(hw_o ** 2 + hw_p ** 2 + u_o ** 2 + u_p ** 2)
        d = abs(mid_o - mid_p)
        ok = bool(overlap or d <= tol)
        out.append(_result("G7", Verdict.PASS if ok else Verdict.FAIL,
                           estimate_ours=mid_o, estimate_pylle=mid_p,
                           uncertainty_ours=hw_o, uncertainty_pylle=hw_p,
                           brackets_overlap=overlap, abs_diff=d, tolerance=tol,
                           **base))
    return out


def evaluate_diagnostics(values: dict) -> list[dict]:
    """DIAGNOSTIC results carry data only and can never change ``overall``."""
    out = []
    for cid in DIAGNOSTIC_IDS:
        c = CRITERIA_BY_ID[cid]
        out.append(_result(cid, Verdict.DIAGNOSTIC,
                           data=values.get(f"diag_{c.name}"), note=c.note))
    return out


def overall_verdict(checks: list[dict]) -> tuple[str, bool, list[str]]:
    """``overall`` from HARD + GATED only; DIAGNOSTIC is structurally excluded.

    Returns ``(overall, qualified, reasons)``. ``qualified`` is True when a GATED
    check was demoted for want of a derivable tolerance -- the run is then not a
    clean pass even if nothing failed, and says so.
    """
    reasons: list[str] = []
    failed = False
    qualified = False
    for c in checks:
        if c["class"] == CriterionClass.DIAGNOSTIC.value:
            continue
        v = c["verdict"]
        if v in (Verdict.FAIL.value, Verdict.NOT_MEASURED.value):
            failed = True
            reasons.append(f"{c['cid']} {c['name']}: {v}")
        elif v == Verdict.UNDERIVED.value:
            qualified = True
            reasons.append(
                f"{c['cid']} {c['name']}: tolerance UNDERIVED, demoted to "
                f"DIAGNOSTIC ({c.get('reason', '')})")
    return ("FAIL" if failed else "PASS"), qualified, reasons


def build_report(values: dict, tolset: ToleranceSet, *,
                 extra: dict | None = None) -> dict:
    """Assemble the full v2 report and enforce the fingerprint guard.

    The guard is the mechanical version of "no post-hoc tolerance tuning": the
    fingerprint is taken at derivation, before any comparison value exists, and
    recomputed here. If a tolerance was touched in between, this raises.
    """
    live = _fingerprint(tolset.entries)
    if live != tolset.fingerprint:
        raise CriteriaError(
            "tolerance_fingerprint changed between derivation and evaluation: "
            f"derived {tolset.fingerprint}, now {live}. A tolerance was modified "
            "after derivation, which is exactly what the fingerprint exists to "
            "prevent.")

    checks = (evaluate_hard(values) + evaluate_gated(values, tolset)
              + evaluate_existence_edges(values, tolset)
              + evaluate_diagnostics(values))
    overall, qualified, reasons = overall_verdict(checks)

    report = {
        "schema_version": SCHEMA_VERSION,
        "criteria_version": CRITERIA_VERSION,
        "tolerance_fingerprint": tolset.fingerprint,
        "criteria_provenance": tolset.provenance,
        "tolerances": {cid: asdict(e) for cid, e in tolset.entries.items()},
        "checks": checks,
        "overall": overall,
        "overall_qualified": qualified,
        "overall_qualification_reasons": reasons,
    }
    if extra:
        report.update(extra)
    return report


# ==========================================================================
# Schema validation
# ==========================================================================
def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def validate_report(payload: dict) -> None:
    """Validate a v2 report. Raises :class:`CriteriaError` on any failure.

    Uses ``jsonschema`` when available for full draft-2020-12 checking, and
    always applies the structural invariants that matter most and that a schema
    cannot express: every HARD criterion present, every GATED check carrying
    both uncertainties, a self-consistent fingerprint, and only known verdicts.
    """
    if not isinstance(payload, dict):
        raise CriteriaError("report must be a JSON object")

    try:
        import jsonschema
    except Exception:
        jsonschema = None
    if jsonschema is not None:
        try:
            jsonschema.validate(payload, _load_schema())
        except jsonschema.ValidationError as exc:      # pragma: no cover - message path
            raise CriteriaError(f"schema validation failed: {exc.message}") from exc

    for key in ("schema_version", "criteria_version", "tolerance_fingerprint",
                "tolerances", "checks", "overall"):
        if key not in payload:
            raise CriteriaError(f"report is missing required key {key!r}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise CriteriaError(f"unsupported schema_version {payload['schema_version']!r}")

    checks = payload["checks"]
    if not isinstance(checks, list):
        raise CriteriaError("checks must be a list")
    seen = {c.get("cid") for c in checks}

    missing_hard = [cid for cid in HARD_IDS if cid not in seen]
    if missing_hard:
        raise CriteriaError(f"report is missing HARD criteria: {missing_hard}")

    for c in checks:
        v = c.get("verdict")
        if v not in VERDICTS:
            raise CriteriaError(f"unknown verdict {v!r} on {c.get('cid')}")
        if c.get("class") == CriterionClass.GATED.value and c.get("cid") in DERIVED_GATED_IDS:
            if "u_ours" not in c or "u_pylle" not in c:
                raise CriteriaError(
                    f"GATED check {c.get('cid')} must carry u_ours and u_pylle")

    tols = payload["tolerances"]
    rebuilt = {
        cid: DerivedTolerance(
            cid, e.get("name", ""), e.get("tolerance"), e.get("status", ""),
            e.get("kind", ""))
        for cid, e in tols.items()
    }
    if _fingerprint(rebuilt) != payload["tolerance_fingerprint"]:
        raise CriteriaError(
            "tolerance_fingerprint does not match the tolerances block; the "
            "tolerances were altered after derivation")

    # DIAGNOSTIC results must be structurally incapable of changing the verdict.
    recomputed, qualified, _ = overall_verdict(checks)
    if recomputed != payload["overall"]:
        raise CriteriaError(
            f"overall {payload['overall']!r} disagrees with the checks "
            f"(recomputed {recomputed!r})")
    if "overall_qualified" in payload and bool(payload["overall_qualified"]) != qualified:
        raise CriteriaError("overall_qualified disagrees with the checks")


__all__ = [
    "CRITERIA", "CRITERIA_BY_ID", "CriterionClass", "Verdict", "CriteriaError",
    "COVERAGE_FACTOR", "FLOOR_TOL_RELATIVE", "FLOOR_TOL_ABSOLUTE_MODES",
    "HARD_THRESHOLDS", "HARD_IDS", "GATED_IDS", "DIAGNOSTIC_IDS",
    "DERIVED_GATED_IDS", "TOLERANCES_V1_HISTORICAL",
    "derive_tolerances", "DerivedTolerance", "ToleranceSet",
    "evaluate_hard", "evaluate_gated", "evaluate_existence_edges",
    "evaluate_diagnostics", "overall_verdict", "build_report",
    "validate_report", "rel", "combine",
]
