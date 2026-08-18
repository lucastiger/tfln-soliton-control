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
    """How much authority a criterion's verdict carries.

    Attributes
    ----------
    HARD
        Must hold for ANY correct solver, independent of the other code.
        Its threshold is a property of the mathematics, not of an agreement.
    GATED
        A cross-code AGREEMENT claim, and therefore meaningful only to the
        precision both codes actually resolve. Its tolerance is DERIVED from
        the two measured uncertainty studies, never chosen.
    DIAGNOSTIC
        Reported, never decisive. Structurally excluded from ``overall``.

    Notes
    -----
    Subclasses ``str`` so a member serialises straight into JSON as its own
    value and a report can be read without the enum being importable.

    Examples
    --------
    >>> CriterionClass.HARD.value
    'HARD'
    >>> CriterionClass.GATED == "GATED"
    True
    """

    HARD = "HARD"
    GATED = "GATED"
    DIAGNOSTIC = "DIAGNOSTIC"


class Verdict(str, Enum):
    """The outcome of evaluating one criterion.

    Attributes
    ----------
    PASS
        Measured and within threshold or tolerance.
    FAIL
        Measured and outside it.
    NOT_MEASURED
        The value was absent or NaN. Treated as a failure of the run, not as a
        silent skip: an unmeasured criterion is not a satisfied one.
    UNDERIVED
        A GATED criterion whose tolerance could not be derived from the
        uncertainty studies. Demoted to diagnostic and reported as a
        QUALIFICATION on the overall verdict rather than given a guessed
        number.
    DIAGNOSTIC
        Carries data only and can never change ``overall``.

    Notes
    -----
    Subclasses ``str`` for the same JSON-serialisation reason as
    :class:`CriterionClass`.

    Examples
    --------
    >>> Verdict.PASS.value, Verdict.UNDERIVED.value
    ('PASS', 'UNDERIVED')
    >>> sorted(VERDICTS)
    ['DIAGNOSTIC', 'FAIL', 'NOT_MEASURED', 'PASS', 'UNDERIVED']
    """

    PASS = "PASS"
    FAIL = "FAIL"
    NOT_MEASURED = "NOT_MEASURED"
    UNDERIVED = "UNDERIVED"
    DIAGNOSTIC = "DIAGNOSTIC"


VERDICTS = {v.value for v in Verdict}


class CriteriaError(RuntimeError):
    """Raised on a tolerance-fingerprint mismatch or a malformed report.

    Notes
    -----
    A distinct exception type rather than a bare ``RuntimeError`` because the
    two conditions it signals -- a tolerance edited after derivation, and a
    report that does not satisfy its own schema -- are process failures rather
    than physics failures, and a caller may reasonably want to catch exactly
    those.

    Examples
    --------
    >>> issubclass(CriteriaError, RuntimeError)
    True
    """


# ==========================================================================
# Criterion table -- DATA, not code paths (acceptance criterion A1)
# ==========================================================================
@dataclass(frozen=True)
class Criterion:
    """One acceptance criterion, declared as DATA rather than as a code path.

    Parameters
    ----------
    cid : str
        Stable identifier, e.g. ``"H1"``, ``"G4"``, ``"D7"``.
    name : str
        Machine-readable name, matching the key the measured value arrives
        under.
    cls : CriterionClass
        HARD, GATED or DIAGNOSTIC.
    quantity : str
        Human-readable statement of what is being compared.
    comparison : str
        How it is compared: ``"threshold"``, ``"bool"``, ``"relative"``,
        ``"absolute"``, ``"bracket"`` or ``"report"``.
    ours_key : str or None, optional
        Key of this repository's measured value.
    pylle_key : str or None, optional
        Key of pyLLE's measured value.
    conv_key : str or None, optional
        Observable name in BOTH convergence studies, from which a GATED
        tolerance is derived. A GATED criterion without one derives its own
        scale (G7 does, from the bracket half-widths).
    units : str, optional
        Units of the compared quantity, e.g. ``"relative"``, ``"modes"``,
        ``"dB"``, ``"W"``. Default ``"-"``.
    note : str, optional
        Why the criterion is classified as it is, carried into the report.

    Raises
    ------
    dataclasses.FrozenInstanceError
        On any attempt to mutate an instance.

    Notes
    -----
    Declaring the criteria as a frozen table rather than as a sequence of
    ``if`` statements is acceptance criterion A1 of the cross-check: the set of
    things being checked is then inspectable, diffable and countable, and a
    criterion cannot be quietly weakened by editing the branch that evaluates
    it.

    Examples
    --------
    >>> h1 = CRITERIA_BY_ID["H1"]
    >>> h1.cls.value, h1.comparison, h1.units
    ('HARD', 'threshold', 'relative')
    >>> len(HARD_IDS), len(GATED_IDS), len(DIAGNOSTIC_IDS)
    (7, 7, 7)
    >>> set(CRITERIA_BY_ID) == set(HARD_IDS + GATED_IDS + DIAGNOSTIC_IDS)
    True
    """

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
    """Return the symmetric relative difference between two values.

    Parameters
    ----------
    a, b : float
        The two values, in any consistent unit.

    Returns
    -------
    float
        ``|a - b| / max(|a|, |b|)`` [dimensionless], and exactly 0.0 when both
        are zero.

    Raises
    ------
    TypeError
        If either argument cannot be coerced to ``float``.

    Notes
    -----
    Symmetric by construction -- neither value is privileged as "the
    reference", which is the right choice for a cross-code comparison where
    both codes are on trial. The both-zero case returns 0.0 rather than
    ``nan``, so an observable that is legitimately zero in both codes reads as
    agreement.

    Examples
    --------
    >>> rel(1.0, 1.0), rel(0.0, 0.0)
    (0.0, 0.0)
    >>> rel(2.0, 1.0)
    0.5
    >>> rel(1.0, 2.0) == rel(2.0, 1.0)          # symmetric
    True
    """
    a, b = float(a), float(b)
    denom = max(abs(a), abs(b))
    return 0.0 if denom == 0.0 else abs(a - b) / denom


def combine(u_a: float, u_b: float, k: float = COVERAGE_FACTOR) -> float:
    """Combine two independent uncertainties into a coverage-scaled tolerance.

    Parameters
    ----------
    u_a, u_b : float
        The two numerical uncertainties, in the same unit (relative or
        absolute, but not mixed).
    k : float, optional
        Coverage factor [dimensionless], default :data:`COVERAGE_FACTOR`
        (2.0) -- a two-sigma-style coverage on the combined uncertainty.

    Returns
    -------
    float
        ``k * sqrt(u_a**2 + u_b**2)``, in the unit of the inputs.

    Raises
    ------
    TypeError
        If any argument cannot be coerced to ``float``.

    Notes
    -----
    Quadrature because the two discretization uncertainties are independent:
    they come from two codes whose truncation errors have no reason to be
    correlated. This is the whole basis of the GATED tolerances -- they are
    computed from what each code measured about itself, so a tolerance can
    never be tightened past the precision either code actually resolves.

    Examples
    --------
    >>> round(combine(3e-3, 4e-3), 6)          # 2 * sqrt(9 + 16) * 1e-3
    0.01
    >>> combine(0.0, 0.0)
    0.0
    >>> bool(combine(1e-3, 1e-3, k=1.0) < combine(1e-3, 1e-3, k=2.0))
    True
    """
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
    """One GATED criterion's tolerance, together with how it was derived.

    Parameters
    ----------
    cid : str
        Criterion identifier.
    name : str
        Criterion name.
    tolerance : float or None
        The derived tolerance, relative [dimensionless] or absolute (units of
        the observable, e.g. modes for G5). ``None`` when ``status`` is
        ``"UNDERIVED"``.
    status : {'DERIVED', 'UNDERIVED'}
        Whether a tolerance could be computed from the two uncertainty
        studies.
    kind : {'relative', 'absolute'}
        How ``tolerance`` is to be applied.
    u_ours : float or None, optional
        This repository's measured numerical uncertainty for the observable.
    u_pylle : float or None, optional
        pyLLE's, from its own refinement study.
    coverage_factor : float, optional
        The ``k`` used in :func:`combine`, default :data:`COVERAGE_FACTOR`.
    floor : float or None, optional
        Lower bound applied to the derived tolerance, so a study that happened
        to measure an implausibly small uncertainty cannot produce a tolerance
        no correct code could meet.
    reason : str, optional
        Why the tolerance is UNDERIVED, when it is.
    source_ours, source_pylle : dict, optional
        The provenance of each uncertainty: which study, which level, which
        status it carried.

    Raises
    ------
    TypeError
        From dataclass construction if ``cid``, ``name``, ``tolerance``,
        ``status`` or ``kind`` is omitted.

    Notes
    -----
    Mutable by design, unlike :class:`Criterion`: it is built during
    derivation. What must not change afterwards is guarded by the fingerprint
    rather than by immutability, because the fingerprint covers exactly the
    numbers verdicts are measured against and nothing else.

    Examples
    --------
    >>> t = DerivedTolerance(cid="G1", name="peak_power", tolerance=0.01,
    ...                      status="DERIVED", kind="relative",
    ...                      u_ours=3e-3, u_pylle=4e-3)
    >>> t.tolerance, t.kind, t.coverage_factor
    (0.01, 'relative', 2.0)

    An underivable tolerance is recorded as such, never guessed:

    >>> u = DerivedTolerance(cid="G6", name="comb_frac", tolerance=None,
    ...                      status="UNDERIVED", kind="relative",
    ...                      reason="pyLLE study has no such observable")
    >>> u.tolerance is None and u.status == "UNDERIVED"
    True
    """

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
    """Every derived tolerance, fingerprinted before any comparison value is read.

    Parameters
    ----------
    entries : dict of str to DerivedTolerance
        One entry per GATED criterion, keyed by ``cid``.
    fingerprint : str
        ``sha256`` over the ``(criterion -> tolerance)`` mapping ONLY, taken at
        derivation time.
    provenance : dict
        Which study files the uncertainties came from, their hashes, the git
        state and the argv.

    Raises
    ------
    TypeError
        From dataclass construction if a field is omitted.

    Notes
    -----
    The fingerprint is the mechanical form of "no post-hoc tolerance tuning".
    It is computed BEFORE any comparison value exists and re-computed in
    :func:`build_report`; if a tolerance moved in between, the report refuses
    to build. It deliberately covers only the tolerances -- not the
    uncertainties or the provenance -- because those may legitimately be
    annotated afterwards, while the numbers verdicts are measured against may
    not.

    Examples
    --------
    >>> entries = {"G1": DerivedTolerance(cid="G1", name="peak_power",
    ...                                   tolerance=0.01, status="DERIVED",
    ...                                   kind="relative")}
    >>> tolset = ToleranceSet(entries=entries,
    ...                       fingerprint=_fingerprint(entries),
    ...                       provenance={})
    >>> tolset.tol("G1"), tolset.is_derived("G1")
    (0.01, True)
    >>> len(tolset.fingerprint)
    64
    """

    entries: dict[str, DerivedTolerance]
    fingerprint: str
    provenance: dict[str, Any]

    def tol(self, cid: str) -> float | None:
        """Return the tolerance for one criterion.

        Parameters
        ----------
        cid : str
            Criterion identifier.

        Returns
        -------
        float or None
            The derived tolerance, or ``None`` if it is UNDERIVED.

        Raises
        ------
        KeyError
            If ``cid`` is not in this set -- a missing tolerance is an error,
            never a permissive default.

        Examples
        --------
        >>> entries = {"G1": DerivedTolerance(cid="G1", name="peak_power",
        ...                                   tolerance=0.01, status="DERIVED",
        ...                                   kind="relative")}
        >>> ToleranceSet(entries, _fingerprint(entries), {}).tol("G1")
        0.01
        """
        return self.entries[cid].tolerance

    def is_derived(self, cid: str) -> bool:
        """Whether a criterion's tolerance was successfully derived.

        Parameters
        ----------
        cid : str
            Criterion identifier.

        Returns
        -------
        bool
            ``True`` iff the entry's ``status`` is ``"DERIVED"``.

        Raises
        ------
        KeyError
            If ``cid`` is not in this set.

        Notes
        -----
        :func:`evaluate_gated` uses this to demote an underivable criterion to
        DIAGNOSTIC rather than evaluating it against a guessed number.

        Examples
        --------
        >>> entries = {"G6": DerivedTolerance(cid="G6", name="comb_frac",
        ...                                   tolerance=None,
        ...                                   status="UNDERIVED",
        ...                                   kind="relative")}
        >>> ToleranceSet(entries, _fingerprint(entries), {}).is_derived("G6")
        False
        """
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

    Parameters
    ----------
    ours_path : pathlib.Path or str or None, optional
        This repository's convergence study; ``None`` (default) uses
        ``validation/results/convergence_lle_dw30k.json``.
    pylle_path : pathlib.Path or str or None, optional
        pyLLE's refinement study; ``None`` (default) uses
        ``validation/results/pylle_refinement_dw30k.json``.
    ours_level : str, optional
        Which uncertainty to read from our study, default
        ``"numerical_uncertainty_at_n1"`` -- the band at the level the
        comparison was actually run at. Keyword-only.
    pylle_tag : str, optional
        Which refinement tag to read from the reference study, default
        ``"tight"``. Keyword-only.
    argv : list of str or None, optional
        Command line recorded in the provenance. Keyword-only.

    Returns
    -------
    ToleranceSet
        One :class:`DerivedTolerance` per criterion in
        :data:`DERIVED_GATED_IDS`, the fingerprint taken over them, and the
        provenance of both source studies.

    Raises
    ------
    FileNotFoundError
        If either study file is absent.
    json.JSONDecodeError
        If either is not valid JSON.
    KeyError
        If a criterion in :data:`DERIVED_GATED_IDS` has no floor declared in
        :data:`FLOOR_TOL_RELATIVE` or :data:`FLOOR_TOL_ABSOLUTE_MODES`.

    Notes
    -----
    Must be called BEFORE any comparison value is read; the returned
    fingerprint is what :func:`build_report` re-checks. That ordering is the
    whole guarantee: a tolerance derived after seeing a result is not a
    tolerance, it is a decision.

    A criterion whose uncertainty could not be measured in BOTH studies gets no
    tolerance at all and is marked UNDERIVED, carrying the reason from each
    side. It is then demoted rather than given a plausible number -- an
    agreement claim at a tolerance nobody measured is not evidence.

    No ``Examples`` section: it reads the two committed study artifacts.
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
        if absolute:
            tol = max(raw, floor)
            # State the mode-converted uncertainties explicitly: the entry's
            # u_ours/u_pylle fields stay RELATIVE (that is what the studies
            # measured), so quoting them here without the conversion would not
            # reproduce `raw`.
            why = (f"max({COVERAGE_FACTOR}*sqrt(U_o^2+U_p^2)={raw:.6g} modes "
                   f"[U_o={u_o_abs:.6g}, U_p={u_p_abs:.6g} modes], "
                   f"floor={floor:g}); no relative ceiling applied to an "
                   f"absolute tolerance")
        else:
            tol = min(max(raw, floor), MAX_TOL)
            why = (f"clip({COVERAGE_FACTOR}*sqrt(U_o^2+U_p^2)={raw:.6g}, "
                   f"floor={floor:g}, hi={MAX_TOL:g})")
        entries[cid] = DerivedTolerance(
            cid, c.name, float(tol), "DERIVED",
            "absolute" if absolute else "relative",
            u_o, u_p, COVERAGE_FACTOR, floor, why, src_o, src_p)

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
    """Evaluate the HARD checks from a flat dict of measured values.

    Parameters
    ----------
    values : dict
        Measured values keyed by criterion ``name``. Units follow the
        criterion: ``parameter_round_trip_max_rel`` and
        ``dispersion_refit_max_rel`` are relative [dimensionless],
        ``pump_mode_reference_delta_hz`` is in Hz, and
        ``dispersion_mirror_applied`` / ``seed_arrays_identical`` are booleans.

    Returns
    -------
    list of dict
        One result per HARD criterion, in :data:`HARD_IDS` order, each with
        ``cid``, ``name``, ``class``, ``quantity``, ``verdict`` and the
        measured ``value`` (plus ``threshold`` where one applies).

    Raises
    ------
    KeyError
        If a criterion id is missing from :data:`CRITERIA_BY_ID`.
    TypeError
        If a supplied value is not numeric where a threshold comparison
        expects one.

    Notes
    -----
    HARD criteria are gated on thresholds that are properties of the
    MATHEMATICS, not of an agreement: an exact algebraic round trip must invert
    to 1e-12, and the pump-mode reference must match the CSV resonance
    EXACTLY (0.0 Hz), because there is no tolerance at which being on the wrong
    mode is acceptable.

    A missing or NaN value yields ``NOT_MEASURED``, which
    :func:`overall_verdict` counts as a failure. Silence is not a pass.

    Examples
    --------
    >>> results = evaluate_hard({"parameter_round_trip_max_rel": 0.0})
    >>> len(results) == len(HARD_IDS)
    True
    >>> results[0]["cid"], results[0]["verdict"], results[0]["threshold"]
    ('H1', 'PASS', 1e-12)

    An absent measurement is reported, not skipped:

    >>> evaluate_hard({})[0]["verdict"]
    'NOT_MEASURED'
    """
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
    """Evaluate the GATED cross-code checks against their derived tolerances.

    Parameters
    ----------
    values : dict
        Measured values keyed ``"<ours_key>__ours"`` and
        ``"<pylle_key>__pylle"``. Units follow each criterion -- W for peak
        power, modes for spans and centroids, dB for relative powers.
    tolset : ToleranceSet
        Tolerances from :func:`derive_tolerances`.

    Returns
    -------
    list of dict
        One result per criterion in :data:`DERIVED_GATED_IDS`, carrying both
        measured values, both uncertainties, the coverage factor, the tolerance
        and its kind and status, the provenance of each uncertainty, and the
        verdict.

    Raises
    ------
    KeyError
        If ``tolset`` has no entry for a criterion in
        :data:`DERIVED_GATED_IDS`.

    Notes
    -----
    A criterion whose tolerance is UNDERIVED is reported with verdict
    ``UNDERIVED`` and does not fail the run; it QUALIFIES it instead, via
    :func:`overall_verdict`. That is the deliberate middle path between
    failing a run for a tolerance nobody could derive and quietly passing it
    against a guessed one.

    No ``Examples`` section: a meaningful call needs a full
    :class:`ToleranceSet` derived from the two committed study artifacts.
    """
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


def evaluate_existence_edges(values: dict, tolset: ToleranceSet,
                             *, discretization_u: dict | None = None) -> list[dict]:
    """G7: brackets, not points.

    PASSES iff the two brackets OVERLAP, or their midpoints agree within
    ``K*sqrt(hw_o^2 + hw_p^2 + U_o^2 + U_p^2)``. v1 compared ``br[1]`` -- the
    tightest surviving point -- on both sides, which is a biased estimator: it
    reports the edge as the last point that happened to survive rather than as
    the interval the bisection actually resolved.

    ``discretization_u`` is an OPTIONAL ``{edge: U_disc}`` mapping in the same
    units as the brackets (kappa), measured by refining the edge bisection over
    a substep ladder -- see ``validation/existence_convergence.py``. G7 is the
    only GATED criterion whose band was built from bracket half-widths alone,
    i.e. with no discretization term at all, because the v2 existence bisection
    ran at ``n_substeps = 1`` only. When supplied, the term enters in
    quadrature exactly as the bracket half-widths do::

        band = K * sqrt(hw_o^2 + hw_p^2 + u_o^2 + u_p^2 + U_disc^2)

    **When it is None the result is bit-identical to the original**, which is
    what keeps the frozen v2 artifact and the tolerance fingerprint valid; a
    test asserts that. Supplying it is a measurement being added to the band,
    never a tolerance being widened to obtain a pass: the value comes from
    ``existence_convergence_ours.json`` and nowhere else.

    Parameters
    ----------
    values : dict
        Measured values, read as ``existence_<edge>_bracket_ours`` and
        ``existence_<edge>_bracket_pylle`` for ``edge`` in ``lower``,
        ``upper``. Brackets are in units of ``kappa`` [dimensionless].
    tolset : ToleranceSet
        The derived tolerances, for the coverage factor and provenance.
    discretization_u : dict or None, optional
        Optional ``{edge: U_disc}`` mapping [same units as the brackets], from
        ``validation/existence_convergence.py``. ``None`` (default) reproduces
        the original result bit-for-bit. Keyword-only.

    Returns
    -------
    list of dict
        One result per edge, carrying both brackets, their half-widths, the
        combined band, the midpoint separation and the verdict.

    Raises
    ------
    KeyError
        If ``tolset`` lacks the entry this criterion reads.
    TypeError
        If a supplied bracket is not a two-element sequence.

    Notes
    -----
    Brackets, not points -- and that distinction is the whole criterion. A
    bisection resolves an INTERVAL; reporting its tightest surviving point as
    "the edge" is a biased estimator, and comparing two such points across
    codes compares two biases.

    Examples
    --------
    The bracket algebra this criterion rests on:

    >>> lower_ours, lower_pylle = (2.1, 2.3), (2.2, 2.5)
    >>> overlap = (lower_ours[0] <= lower_pylle[1]
    ...            and lower_pylle[0] <= lower_ours[1])
    >>> overlap                       # overlapping brackets agree outright
    True
    >>> half_width = (lower_ours[1] - lower_ours[0]) / 2
    >>> round(half_width, 4)
    0.1
    """
    out = []
    for edge in ("lower", "upper"):
        bo = values.get(f"existence_{edge}_bracket_ours")
        bp = values.get(f"existence_{edge}_bracket_pylle")
        u_o = values.get(f"existence_{edge}_u_ours", 0.0) or 0.0
        u_p = values.get(f"existence_{edge}_u_pylle", 0.0) or 0.0
        u_disc = float((discretization_u or {}).get(edge) or 0.0)
        base = {"edge": edge, "bracket_ours": bo, "bracket_pylle": bp,
                "u_ours": u_o, "u_pylle": u_p,
                "coverage_factor": COVERAGE_FACTOR}
        if u_disc:
            base["u_discretization_ours"] = u_disc
        if not bo or not bp or len(bo) != 2 or len(bp) != 2:
            out.append(_result("G7", Verdict.NOT_MEASURED, **base))
            continue
        lo_o, hi_o = sorted(float(x) for x in bo)
        lo_p, hi_p = sorted(float(x) for x in bp)
        mid_o, mid_p = 0.5 * (lo_o + hi_o), 0.5 * (lo_p + hi_p)
        hw_o, hw_p = 0.5 * (hi_o - lo_o), 0.5 * (hi_p - lo_p)
        overlap = (lo_o <= hi_p) and (lo_p <= hi_o)
        tol = COVERAGE_FACTOR * math.sqrt(
            hw_o ** 2 + hw_p ** 2 + u_o ** 2 + u_p ** 2 + u_disc ** 2)
        d = abs(mid_o - mid_p)
        ok = bool(overlap or d <= tol)
        out.append(_result("G7", Verdict.PASS if ok else Verdict.FAIL,
                           estimate_ours=mid_o, estimate_pylle=mid_p,
                           uncertainty_ours=hw_o, uncertainty_pylle=hw_p,
                           brackets_overlap=overlap, abs_diff=d, tolerance=tol,
                           **base))
    return out


def evaluate_diagnostics(values: dict) -> list[dict]:
    """Collect the DIAGNOSTIC results, which carry data only.

    Parameters
    ----------
    values : dict
        Measured values, read as ``diag_<criterion name>``. Units follow each
        criterion -- counts, modes, radians.

    Returns
    -------
    list of dict
        One result per criterion in :data:`DIAGNOSTIC_IDS`, each with verdict
        ``'DIAGNOSTIC'``, the raw ``data`` (``None`` when absent) and the
        criterion's ``note``.

    Raises
    ------
    KeyError
        If a criterion id is missing from :data:`CRITERIA_BY_ID`.

    Notes
    -----
    These never change ``overall`` -- :func:`overall_verdict` excludes them by
    CLASS, before reading any verdict. That is what makes it safe to publish
    ill-conditioned quantities such as the -60 dBc line count: they inform a
    reader without ever deciding anything.

    A missing diagnostic is recorded as ``None`` rather than omitted, so the
    report's shape does not depend on which measurements happened to be
    available.

    Examples
    --------
    >>> results = evaluate_diagnostics({})
    >>> len(results) == len(DIAGNOSTIC_IDS)
    True
    >>> {r["verdict"] for r in results}
    {'DIAGNOSTIC'}
    >>> results[0]["data"] is None
    True
    """
    out = []
    for cid in DIAGNOSTIC_IDS:
        c = CRITERIA_BY_ID[cid]
        out.append(_result(cid, Verdict.DIAGNOSTIC,
                           data=values.get(f"diag_{c.name}"), note=c.note))
    return out


def overall_verdict(checks: list[dict]) -> tuple[str, bool, list[str]]:
    """Reduce a list of checks to a single verdict, its qualification and reasons.

    Parameters
    ----------
    checks : list of dict
        Results from :func:`evaluate_hard`, :func:`evaluate_gated`,
        :func:`evaluate_existence_edges` and :func:`evaluate_diagnostics`.

    Returns
    -------
    overall : str
        ``'PASS'`` or ``'FAIL'``.
    qualified : bool
        ``True`` when at least one GATED check was demoted for want of a
        derivable tolerance -- the run is then not a clean pass even though
        nothing failed, and says so.
    reasons : list of str
        One line per failing or qualifying check, naming its id and name.

    Raises
    ------
    KeyError
        If a check dict is missing ``class`` or ``verdict``.

    Notes
    -----
    DIAGNOSTIC results are excluded STRUCTURALLY -- by class, before the
    verdict is even read -- rather than by their verdict value. A diagnostic
    can therefore never change the outcome no matter what it reports, which is
    the property that lets ill-conditioned observables be published alongside
    the decisive ones without contaminating them.

    ``NOT_MEASURED`` counts as a failure, deliberately: a criterion that was
    never evaluated has not been satisfied.

    Examples
    --------
    >>> overall_verdict([{"class": "HARD", "verdict": "PASS",
    ...                   "cid": "H1", "name": "round_trip"}])
    ('PASS', False, [])
    >>> verdict, qualified, reasons = overall_verdict(
    ...     [{"class": "HARD", "verdict": "FAIL",
    ...       "cid": "H1", "name": "round_trip"}])
    >>> verdict, qualified, reasons
    ('FAIL', False, ['H1 round_trip: FAIL'])

    A diagnostic cannot move the verdict:

    >>> overall_verdict([{"class": "DIAGNOSTIC", "verdict": "DIAGNOSTIC",
    ...                   "cid": "D1", "name": "line_count"}])
    ('PASS', False, [])
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
    """Assemble the full v2 report and enforce the tolerance-fingerprint guard.

    Parameters
    ----------
    values : dict
        Every measured value, keyed as the evaluators expect.
    tolset : ToleranceSet
        The tolerances derived BEFORE any of ``values`` was read.
    extra : dict or None, optional
        Additional top-level keys merged into the report. Keyword-only.

    Returns
    -------
    dict
        ``schema_version``, ``criteria_version``, ``tolerance_fingerprint``,
        ``criteria_provenance``, ``tolerances``, ``checks``, ``overall``,
        ``overall_qualified`` and ``overall_qualification_reasons``, plus
        anything in ``extra``.

    Raises
    ------
    CriteriaError
        If the fingerprint recomputed here differs from the one recorded at
        derivation -- meaning a tolerance was modified after derivation, which
        is exactly what the fingerprint exists to prevent.
    KeyError
        Propagated from the evaluators for a missing tolerance entry.

    Notes
    -----
    The guard is the mechanical version of "no post-hoc tolerance tuning": the
    fingerprint is taken at derivation, before any comparison value exists, and
    recomputed here. A reviewer does not have to trust that nobody adjusted a
    number after seeing a result; the report simply cannot be built if anyone
    did.

    No ``Examples`` section: a meaningful call needs the full measured-value
    set and a derived :class:`ToleranceSet`.
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
    """Validate a v2 report against the schema and the structural invariants.

    Parameters
    ----------
    payload : dict
        A report as produced by :func:`build_report`.

    Returns
    -------
    None
        Returns normally iff the report is valid.

    Raises
    ------
    CriteriaError
        If ``payload`` is not a mapping, fails the JSON schema, omits a HARD
        criterion, carries a GATED check without both uncertainties, has a
        fingerprint inconsistent with its own tolerances, or contains a verdict
        outside :data:`VERDICTS`.
    FileNotFoundError
        If ``validation/schemas/crosscheck_v2.schema.json`` is absent.

    Notes
    -----
    Uses ``jsonschema`` when available for full draft-2020-12 checking, and
    ALWAYS applies the structural invariants a schema cannot express: every
    HARD criterion present, every GATED check carrying both uncertainties, a
    self-consistent fingerprint, and only known verdicts. Those checks are the
    ones that matter, so they must not be contingent on an optional dependency
    being installed.

    No ``Examples`` section: a valid payload is a full report.
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
