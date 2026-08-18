"""How far does OUR OWN single-soliton existence edge move under refinement?

THE GAP THIS CLOSES
-------------------
G7 (existence edge, lower and upper) is the only GATED criterion in the suite
whose tolerance contains **no discretization-uncertainty term**. Every other
GATED criterion draws its uncertainty from a convergence ladder; G7's band is
built from bisection bracket half-widths alone, because the existence bisection
in ``validation/pylle_crosscheck_v2.py`` runs at ``n_substeps = 1`` only. Until
our own edge is refined we cannot say whether the 0.281 kappa cross-code gap at
the lower edge is a disagreement between the codes or a resolution limit of the
measurement.

This module runs **our solver only**. It does not run pyLLE, does not need
Julia, and does not re-run the v2 cross-check. pyLLE's v2 edge values are used
strictly as read-only reference points.

PRE-COMMITTED HYPOTHESES
------------------------
Written before the experiment was run and not revised afterwards. The
thresholds, not the results, select the outcome::

    H-A  "under-resolved criterion": the ours-side edge moves by >= 0.28 kappa
         across n = 1 -> 4, i.e. comparably to or more than the cross-code gap.
         => the v2 G7 FAIL is a resolution limit, not a code disagreement.
    H-B  "real disagreement": the edge moves by < 0.05 kappa across n = 1 -> 4,
         i.e. well inside the 0.0469 kappa bracket half-width.
         => the disagreement is not our discretization. Record it as a standing
         open item with the classifier as the leading remaining suspect, and do
         NOT widen G7 to make it pass.
    H-C  "intermediate": movement between 0.05 and 0.28 kappa. Report the
         measured value, recompute G7's band with it, and state the resulting
         verdict whatever it is.

**A null result (H-B) is a valid outcome and is reported as prominently as any
other.**

WHAT IS MEASURED
----------------
The existence edge definition is UNCHANGED from v2 (a change would require a new
versioned observable; see the module's ``EDGE_OBSERVABLE_VERSION``): seed a
deterministic sech on the lower CW branch at detuning ``dw``, hold ``dw``
constant for ``n_roundtrips_scan`` round trips, and classify the final field
with ``is_single_soliton``. The lower edge is the smallest surviving ``dw``, the
upper the largest; both are bracketed on the coarse scan grid and bisected.

New here is the discretization uncertainty of that edge. With ``e(n)`` the edge
midpoint at ``n_substeps = n``::

    drift_1_to_4 = |e(4) - e(1)|
    drift_2_to_4 = |e(4) - e(2)|
    U_disc       = max(drift_2_to_4, bracket_halfwidth(4))

``U_disc`` is floored at the finest bracket half-width because the bisection
cannot resolve the edge better than that no matter how small the drift is.
Richardson extrapolation is attempted on ``(e(1), e(2), e(4))`` and its status
reported, but a non-monotone or out-of-range sequence falls back to ``U_disc``
above rather than being forced into a fit.

The recomputed G7 band is::

    band = K * sqrt(hw_ours^2 + hw_pylle^2 + U_disc_ours^2)

**This band is asymmetric and that is a known limitation**: pyLLE's own edge was
not refined by this task, so it contributes no discretization term. The JSON
records ``pylle_u_disc_not_measured: true`` and the report says so explicitly.

REUSE
-----
Everything here is imported. The seed, the classifier, the bracket primitives
and the Richardson routine are the same objects the v2 cross-check uses --
``tests/test_existence_convergence.py::test_reuses_v2_objects`` asserts object
identity, so a second implementation cannot drift into existence unnoticed. The
regression gate (R1) additionally proves that this module at ``n_substeps = 1``
reproduces the v2 JSON's ours-side existence block exactly; if it does not, the
run stops, because nothing downstream would be interpretable.

RUNNING IT
----------
    python -m validation.existence_convergence

Writes ``validation/results/existence_convergence_ours.json`` and
``..._fields.npz``. The v2 artifacts are never touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "validation" / "results"
V2_JSON = RESULTS_DIR / "pylle_crosscheck_v2.json"
V2_FIELDS = RESULTS_DIR / "pylle_crosscheck_v2_fields.npz"

SCHEMA_VERSION = "1.0"

# The edge definition is carried over from v2 unchanged. R8: any change to the
# classifier or the edge definition must be introduced as a NEW versioned
# observable with both definitions computed side by side. This constant exists so
# that such a change cannot be made silently.
EDGE_OBSERVABLE_VERSION = "edge_v1"
EDGE_DEFINITION = (
    "lower/upper edge = smallest/largest delta_omega at which is_single_soliton "
    "(contrast >= 5 AND exactly one circular connected component above the "
    "half-way level between peak and median power) returns True for the final "
    "field of a constant-detuning hold of n_roundtrips_scan round trips, seeded "
    "with the deterministic sech ansatz on the lower CW branch at that detuning; "
    "bracketed on the coarse scan grid and bisected bisect_iters times; edge "
    "estimate = bracket midpoint, bracket uncertainty = half-width"
)

# Pre-committed decision thresholds (kappa). Fixed before the run.
H_A_THRESHOLD = 0.28      # >= this: under-resolved criterion
H_B_THRESHOLD = 0.05      # <  this: real disagreement (null result)

DEFAULT_LEVELS = "1,2,4"
# Same offsets the v2 run used, so the peak-power spreads are directly
# comparable to v2's (including pyLLE's 22.6% at its lower-edge midpoint).
STATIONARITY_OFFSETS = (1, 2, 5, 10)

VOLATILE_KEYS = frozenset({
    "wall_s", "wall_clock_total_s", "timestamp_utc", "hostname", "platform",
    "work_dir", "python", "level_wall_s",
})


# ==========================================================================
# helpers
# ==========================================================================
def _sha_arr(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def _strip_volatile(obj):
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items() if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


def numerical_digest(payload: dict) -> str:
    """Hash every non-volatile field of a report, so two runs can be compared.

    Parameters
    ----------
    payload : dict
        The report. Its own ``numerical_digest`` key, if present, is excluded.

    Returns
    -------
    str
        64 lowercase hex characters over the payload with every key in
        :data:`VOLATILE_KEYS` removed recursively.

    Raises
    ------
    TypeError
        If a value is neither JSON-serialisable nor ``str``-representable; the
        encoder passes ``default=str``, so this is close to unreachable.

    Notes
    -----
    Same style as the v2 report. Volatile keys -- timestamps, wall-clock times,
    hostname, platform, work directory -- are stripped RECURSIVELY, so the
    digest answers "did the numbers change?" rather than "was this the same
    run?". Without that, every rerun would differ and the digest would carry no
    information.

    Examples
    --------
    >>> len(numerical_digest({"a": 1}))
    64
    >>> numerical_digest({"a": 1}) == numerical_digest({"a": 1,
    ...                                                 "timestamp_utc": "x"})
    True
    >>> numerical_digest({"a": 1}) == numerical_digest({"a": 2})
    False
    """
    stripped = _strip_volatile({k: v for k, v in payload.items()
                                if k != "numerical_digest"})
    return hashlib.sha256(
        json.dumps(stripped, sort_keys=True, separators=(",", ":"),
                   default=str).encode()).hexdigest()


# ==========================================================================
# the edge measurement
# ==========================================================================
def edge_uncertainty(e1: float | None, e2: float | None, e4: float | None,
                     halfwidth_finest: float | None) -> dict:
    """Measure the discretization uncertainty of one existence edge.

    Parameters
    ----------
    e1, e2, e4 : float or None
        Edge position [units of kappa] at ``n_substeps`` = 1, 2 and 4. ``None``
        for a level that did not produce one.
    halfwidth_finest : float or None
        Bracket half-width [units of kappa] at the finest level.

    Returns
    -------
    dict
        ``e1``, ``e2``, ``e4`` echoed; ``drift_1_to_4`` and ``drift_2_to_4``
        [kappa]; ``U_disc`` [kappa]; ``halfwidth_finest``; the
        :func:`~validation.convergence_lle.richardson` study over the three
        levels; and ``u_disc_source``, naming which term set ``U_disc``
        (``"drift_2_to_4"``, ``"bracket_halfwidth"``,
        ``"bracket_halfwidth_only"`` or ``"INSUFFICIENT_LEVELS"``).

    Raises
    ------
    TypeError
        If a supplied level is not numeric.

    Notes
    -----
    ``U_disc`` is FLOORED at the finest bracket half-width: however small the
    drift between levels, the bisection never resolved the edge better than one
    half-width, and claiming otherwise would report a precision that was never
    measured.

    ``u_disc_source`` is returned precisely so a reader can tell which of the
    two the number came from -- a ``U_disc`` set by the half-width says the
    ladder converged below the bisection's resolution, which is a different
    statement from one set by the drift.

    Examples
    --------
    A ladder that settles well inside the bracket half-width:

    >>> u = edge_uncertainty(2.10, 2.05, 2.04, halfwidth_finest=0.01)
    >>> round(u["drift_1_to_4"], 6), round(u["drift_2_to_4"], 6)
    (0.06, 0.01)
    >>> u["U_disc"], u["u_disc_source"]
    (0.01, 'bracket_halfwidth')
    >>> u["richardson"]["status"]
    'OK'

    With no levels at all the half-width still bounds the edge:

    >>> edge_uncertainty(None, None, None, 0.02)["u_disc_source"]
    'bracket_halfwidth_only'
    >>> edge_uncertainty(None, None, None, None)["U_disc"] is None
    True
    """
    from validation.convergence_lle import richardson

    out = {"e1": e1, "e2": e2, "e4": e4,
           "drift_1_to_4": None, "drift_2_to_4": None, "U_disc": None,
           "halfwidth_finest": halfwidth_finest,
           "richardson": None, "u_disc_source": None}
    if e1 is None or e2 is None or e4 is None:
        out["u_disc_source"] = "INSUFFICIENT_LEVELS"
        if halfwidth_finest is not None:
            out["U_disc"] = float(halfwidth_finest)
            out["u_disc_source"] = "bracket_halfwidth_only"
        return out

    d14 = abs(float(e4) - float(e1))
    d24 = abs(float(e4) - float(e2))
    hw = float(halfwidth_finest or 0.0)
    out["drift_1_to_4"] = d14
    out["drift_2_to_4"] = d24
    out["U_disc"] = max(d24, hw)
    out["u_disc_source"] = ("drift_2_to_4" if d24 >= hw else "bracket_halfwidth")

    # h = 1/n_substeps, coarse -> fine.
    out["richardson"] = richardson([e1, e2, e4], [1.0, 0.5, 0.25])
    return out


def select_hypothesis(drift_lower: float | None, drift_upper: float | None) -> dict:
    """Classify the outcome against the pre-committed drift thresholds.

    Parameters
    ----------
    drift_lower, drift_upper : float or None
        ``drift_1_to_4`` [units of kappa] for each edge, from
        :func:`edge_uncertainty`. ``None`` for an edge that produced none.

    Returns
    -------
    dict
        ``outcome`` (``'H-A'``, ``'H-B'``, ``'H-C'`` or ``None``),
        ``deciding_drift`` [kappa], both input drifts, the ``thresholds`` that
        were applied, and a ``reason`` naming the comparison in full.

    Raises
    ------
    TypeError
        If a supplied drift is not numeric.

    Notes
    -----
    The thresholds :data:`H_A_THRESHOLD` (0.28 kappa, the cross-code lower-edge
    gap) and :data:`H_B_THRESHOLD` (0.05 kappa, the bracket half-width) are
    PRE-COMMITTED -- fixed before the study ran. H-A means the criterion was
    under-resolved; H-B means the disagreement is real and the result is a null;
    H-C means neither, and says so rather than rounding to whichever is more
    convenient.

    Deciding on the worst of the two edges is the conservative choice: one
    under-resolved edge is enough to make the criterion under-resolved.

    Examples
    --------
    >>> select_hypothesis(0.30, 0.01)["outcome"]      # >= 0.28 kappa
    'H-A'
    >>> select_hypothesis(0.01, 0.02)["outcome"]      # < 0.05 kappa
    'H-B'
    >>> select_hypothesis(0.10, None)["outcome"]      # between the two
    'H-C'
    >>> select_hypothesis(None, None)["outcome"] is None
    True
    """
    drifts = [d for d in (drift_lower, drift_upper) if d is not None]
    if not drifts:
        return {"outcome": None, "deciding_drift": None,
                "reason": "no edge produced a drift_1_to_4"}
    worst = max(drifts)
    if worst >= H_A_THRESHOLD:
        outcome, why = "H-A", (
            f"max drift_1_to_4 = {worst:.4f} kappa >= the pre-committed H-A "
            f"threshold of {H_A_THRESHOLD} kappa (the cross-code lower-edge gap)")
    elif worst < H_B_THRESHOLD:
        outcome, why = "H-B", (
            f"max drift_1_to_4 = {worst:.4f} kappa < the pre-committed H-B "
            f"threshold of {H_B_THRESHOLD} kappa (inside the bracket half-width)")
    else:
        outcome, why = "H-C", (
            f"max drift_1_to_4 = {worst:.4f} kappa lies between the "
            f"pre-committed thresholds {H_B_THRESHOLD} and {H_A_THRESHOLD} kappa")
    return {"outcome": outcome, "deciding_drift": worst,
            "drift_lower": drift_lower, "drift_upper": drift_upper,
            "thresholds": {"H_A_at_or_above": H_A_THRESHOLD,
                           "H_B_below": H_B_THRESHOLD},
            "reason": why}


def _g7_variant(ours_bracket, pylle_bracket, u_disc: float, coverage: float) -> dict:
    """One G7 evaluation: band, gap, overlap, verdict."""
    lo_o, hi_o = sorted(float(x) for x in ours_bracket)
    lo_p, hi_p = sorted(float(x) for x in pylle_bracket)
    mid_o, mid_p = 0.5 * (lo_o + hi_o), 0.5 * (lo_p + hi_p)
    hw_o, hw_p = 0.5 * (hi_o - lo_o), 0.5 * (hi_p - lo_p)
    overlap = (lo_o <= hi_p) and (lo_p <= hi_o)
    d = abs(mid_o - mid_p)
    band = coverage * math.sqrt(hw_o ** 2 + hw_p ** 2 + float(u_disc) ** 2)
    return {
        "ours_bracket": [lo_o, hi_o], "pylle_bracket": [lo_p, hi_p],
        "ours_midpoint": mid_o, "pylle_midpoint": mid_p,
        "hw_ours": hw_o, "hw_pylle": hw_p, "U_disc_ours": float(u_disc),
        "abs_diff": d, "band": band,
        "brackets_overlap": bool(overlap),
        "passes_by_overlap": bool(overlap),
        "passes_by_band": bool(d <= band),
        "verdict": "PASS" if (overlap or d <= band) else "FAIL",
    }


def recompute_g7(edge: str, ours_bracket_v2, ours_bracket_refined, pylle_bracket,
                 u_disc: float | None, coverage: float) -> dict:
    """G7 decomposed into THREE clearly separated evaluations.

    Reporting only "before and after" would conflate two entirely different
    causes of a verdict change -- the edge MOVING under refinement, and the band
    WIDENING because a new uncertainty term was added. Those have opposite
    epistemic standing: the first is a better measurement, the second is a
    weaker test. So all three are reported:

      ``v2_as_measured``       the frozen v2 situation: our n=1 bracket, no
                               discretization term. This must reproduce the v2
                               JSON's G7 verdict exactly.
      ``refined_no_u_disc``    our n=4 bracket, still no discretization term.
                               Isolates the effect of refining our own edge.
      ``refined_with_u_disc``  our n=4 bracket plus the measured U_disc. This is
                               the recomputed criterion.

    If the verdict improves between the first and the second, that is a
    measurement improving. If it improves only between the second and the third,
    that is the band widening, and it should be read far more sceptically.

    Parameters
    ----------
    edge : {'lower', 'upper'}
        Which existence edge is being recomputed.
    ours_bracket_v2 : sequence of float or None
        Our frozen v2 bracket [units of kappa], measured at
        ``n_substeps = 1``.
    ours_bracket_refined : sequence of float or None
        Our refined bracket [kappa], from the finest level of the ladder.
    pylle_bracket : sequence of float or None
        The reference code's bracket [kappa].
    u_disc : float or None
        Measured discretization uncertainty [kappa], or ``None`` to omit the
        term entirely.
    coverage : float
        Coverage factor K [dimensionless] applied to the combined band.

    Returns
    -------
    dict
        ``edge``, ``coverage_factor``, the three variants
        (``v2_as_measured``, ``refined_no_u_disc``, ``refined_with_u_disc``,
        each with both brackets, midpoints, half-widths, the band, the midpoint
        separation and a verdict), ``verdict_change_attributed_to``, and
        ``u_disc_double_counts_halfwidth``. ``{'edge': ..., 'status':
        'NOT_MEASURED'}`` when any bracket is missing.

    Raises
    ------
    TypeError
        If a supplied bracket is not a two-element numeric sequence.

    Notes
    -----
    ``u_disc_double_counts_halfwidth`` names a real double count: when the
    drift is smaller than the bracket half-width, ``U_disc`` is floored AT that
    half-width, which then appears both as ``hw_ours`` and inside ``U_disc``.
    The band is then wider than strictly justified, so a FAIL under it is
    conservative and a PASS is not -- which is exactly why the flag is
    reported rather than the double count quietly removed.

    Examples
    --------
    >>> g7 = recompute_g7("lower", (2.1, 2.3), (2.15, 2.25), (2.2, 2.5),
    ...                   u_disc=0.02, coverage=2.0)
    >>> sorted(k for k in g7 if k.startswith(("v2_", "refined_")))
    ['refined_no_u_disc', 'refined_with_u_disc', 'v2_as_measured']
    >>> g7["v2_as_measured"]["verdict"], g7["refined_with_u_disc"]["verdict"]
    ('PASS', 'PASS')
    >>> g7["verdict_change_attributed_to"]
    'no change'

    A missing bracket is reported, never treated as agreement:

    >>> recompute_g7("lower", None, (2.15, 2.25), (2.2, 2.5), 0.02, 2.0)
    {'edge': 'lower', 'status': 'NOT_MEASURED'}
    """
    if not ours_bracket_v2 or not ours_bracket_refined or not pylle_bracket:
        return {"edge": edge, "status": "NOT_MEASURED"}
    u = float(u_disc or 0.0)
    out = {
        "edge": edge,
        "coverage_factor": coverage,
        "v2_as_measured": _g7_variant(ours_bracket_v2, pylle_bracket, 0.0, coverage),
        "refined_no_u_disc": _g7_variant(ours_bracket_refined, pylle_bracket,
                                         0.0, coverage),
        "refined_with_u_disc": _g7_variant(ours_bracket_refined, pylle_bracket,
                                           u, coverage),
    }
    a, b, c = (out["v2_as_measured"], out["refined_no_u_disc"],
               out["refined_with_u_disc"])
    if a["verdict"] == c["verdict"]:
        attribution = "no change"
    elif b["verdict"] != a["verdict"]:
        attribution = ("the refined edge measurement -- our own edge moved; the "
                       "band term is not what changed the verdict")
    else:
        attribution = ("the added discretization term alone -- the band widened "
                       "while the edges stayed put; treat with scepticism")
    out["verdict_change_attributed_to"] = attribution
    # A double-count worth naming: when the drift is smaller than the bracket
    # half-width, U_disc is floored AT that half-width, which then appears both
    # as hw_ours and inside U_disc. That makes the band wider than strictly
    # justified, so a FAIL under it is conservative and a PASS is not.
    out["u_disc_double_counts_halfwidth"] = bool(
        u > 0 and abs(u - c["hw_ours"]) < 1e-12)
    return out


# ==========================================================================
# the run
# ==========================================================================
def run_level(n_substeps: int, *, ctx: dict, scan_dws, bisect_iters: int,
              n_roundtrips: int, deadline: float | None) -> dict:
    """Run the coarse scan and both edge bisections at one refinement level.

    Parameters
    ----------
    n_substeps : int
        Strang sub-steps per round trip [dimensionless] -- the refinement knob.
    ctx : dict
        Run context: ``kappa`` [rad/s], ``t_r`` [s], ``gamma``
        [J^-1 s^-1], ``dev``, ``cfg``, ``d_int_fftorder`` [rad/s] and a
        ``seed_at(dw)`` callable. Keyword-only.
    scan_dws : sequence of float
        Detunings [rad/s] of the coarse survival scan. Keyword-only.
    bisect_iters : int
        Bisection steps per edge [dimensionless count]. Keyword-only.
    n_roundtrips : int
        Hold length per point [round trips]. Keyword-only.
    deadline : float or None
        Wall-clock deadline [s, ``time.time`` epoch] after which the level is
        abandoned; ``None`` for no limit. Keyword-only.

    Returns
    -------
    dict
        ``survival`` per scanned detuning, the ``lower`` and ``upper`` edge
        records (bracket, midpoint, half-width and the full bisection trace,
        all in units of kappa), the level ``status``, and the final fields.

    Raises
    ------
    AssertionError
        If any hold produces a non-finite field, naming the level and detuning.

    Notes
    -----
    Uses the v2 bracket primitives
    (:func:`~validation.pylle_crosscheck_v2.initial_brackets` and
    :func:`~validation.pylle_crosscheck_v2.update_bracket`) rather than
    restating the bisection rule, so the two harnesses cannot drift apart. If
    they could, a difference between this study's edge and the v2 edge would be
    unattributable -- exactly the ambiguity the study exists to remove.

    No ``Examples`` section: one level runs a coarse scan plus two bisections,
    each a full solve.
    """
    from validation.pylle_crosscheck_v2 import (bracket_halfwidth, bracket_midpoint,
                                                classify_field, initial_brackets,
                                                run_ours, update_bracket)

    kappa, t_r, gamma = ctx["kappa"], ctx["t_r"], ctx["gamma"]
    fields: dict[str, np.ndarray] = {}
    t0 = time.time()

    def hold(dw: float, tscan: int) -> np.ndarray:
        r = run_ours(np.full(int(tscan), dw), ctx["seed_at"](dw),
                     ctx["d_int_fftorder"], ctx["dev"], ctx["cfg"],
                     n_substeps=n_substeps)
        f = r["field"]
        if not np.all(np.isfinite(f)):
            raise AssertionError(
                f"non-finite field at n_substeps={n_substeps}, dw/kappa={dw / kappa:.6f} (F4)")
        return f

    def evaluate(dw: float, tag: str, *, stationarity: bool = False) -> dict:
        """Classify a hold; optionally measure how much the state is still moving.

        The spread is taken by TRUNCATING the hold, which at constant detuning is
        exactly the state at that round trip: the trajectory is deterministic and
        the detuning array is constant, so ``run(N-k)`` IS the state at round
        trip ``N-k`` of ``run(N)``. The offsets are the same ones the v2 run used
        (0, 1, 2, 5, 10) so the resulting spreads are directly comparable to
        v2's -- a two-point endpoint difference would be a lower bound on the
        same quantity and could not be put on the same axis.

        It is measured at the bisection midpoints, which is where it matters: a
        marginal state near an edge is exactly what would make a threshold
        classifier flip, and this is what makes such a state visible.
        """
        f = hold(dw, n_roundtrips)
        fields[tag] = f
        c = classify_field(f, t_r, kappa, gamma)
        c["delta_omega_over_kappa"] = dw / kappa
        if not stationarity:
            return c
        peaks, labels = [c["peak_power_w"]], {c["single"]}
        for off in STATIONARITY_OFFSETS:
            ft = hold(dw, n_roundtrips - off)
            fields[f"{tag}_m{off}"] = ft
            ct = classify_field(ft, t_r, kappa, gamma)
            peaks.append(ct["peak_power_w"])
            labels.add(ct["single"])
        c["peak_power_spread_last10"] = (
            (max(peaks) - min(peaks)) / max(peaks) if max(peaks) > 0 else 0.0)
        c["label_changed_over_tail"] = len(labels) > 1
        c["stationarity_offsets"] = list(STATIONARITY_OFFSETS)
        return c

    # ---- coarse scan ----
    survival, scan_records = [], []
    for dw in scan_dws:
        c = evaluate(dw, f"n{n_substeps}_scan_{dw / kappa:.4f}")
        survival.append(c["single"])
        scan_records.append(c)
        print(f"   [n={n_substeps}] scan dw/k={dw / kappa:7.4f} single={c['single']} "
              f"peaks={c['n_peaks']} contrast={c['contrast']:9.1f}", flush=True)
        if deadline and time.time() > deadline:
            return {"n_substeps": n_substeps, "status": "TIMEOUT",
                    "wall_s": time.time() - t0, "survival": survival,
                    "scan": scan_records, "_fields": fields}

    br_lo, br_hi = initial_brackets(survival, scan_dws)
    traces = {"lower": [], "upper": []}
    for it in range(bisect_iters):
        for edge, br in (("lower", br_lo), ("upper", br_hi)):
            if br is None:
                continue
            m = bracket_midpoint(br)
            c = evaluate(m, f"n{n_substeps}_bis{it}_{edge}", stationarity=True)
            traces[edge].append({"iter": it, "dw_over_kappa": m / kappa,
                                 "alive": c["single"], **c})
            if edge == "lower":
                br_lo = update_bracket(br_lo, m, c["single"], "lower")
            else:
                br_hi = update_bracket(br_hi, m, c["single"], "upper")
        print(f"   [n={n_substeps}] bisect {it}: "
              f"lower={_fmt_br(br_lo, kappa)} upper={_fmt_br(br_hi, kappa)}", flush=True)
        if deadline and time.time() > deadline:
            break

    def block(br, trace):
        return {
            "bracket": None if br is None else sorted([br[0] / kappa, br[1] / kappa]),
            "midpoint": None if br is None else bracket_midpoint(br) / kappa,
            "halfwidth": None if br is None else bracket_halfwidth(br) / kappa,
            "trace": trace,
        }

    return {"n_substeps": n_substeps, "status": "OK", "wall_s": time.time() - t0,
            "survival": survival, "scan": scan_records,
            "lower": block(br_lo, traces["lower"]),
            "upper": block(br_hi, traces["upper"]),
            "_fields": fields}


def _fmt_br(br, kappa):
    return "-" if br is None else f"[{br[0] / kappa:.4f},{br[1] / kappa:.4f}]"


def check_regression(level_n1: dict, v2_existence: dict) -> dict:
    """Check that the ``n_substeps = 1`` level reproduces the frozen v2 block exactly.

    Parameters
    ----------
    level_n1 : dict
        The ``n_substeps = 1`` level record from :func:`run_level`.
    v2_existence : dict
        The ours-side existence block of the frozen v2 report.

    Returns
    -------
    dict
        ``reproduces_v2_n1`` (bool) and ``per_field``, a mapping from each
        compared field -- survival vector, both brackets, midpoints,
        half-widths and both bisection traces -- to its two values and whether
        they match.

    Raises
    ------
    KeyError
        If either input is missing a compared field.

    Notes
    -----
    EXACT equality, no tolerance. If ``n_substeps = 1`` does not reproduce the
    v2 block bit-for-bit, this module is not driving the same problem the v2 run
    drove, and nothing downstream -- the drift, ``U_disc``, the recomputed G7
    band -- would be interpretable. A tolerance here would let the study report
    a refinement effect that was really a setup difference.

    The bisection TRACE is compared too, not just the final bracket: two
    different traces that happen to end on the same bracket did not solve the
    same sequence of problems.

    No ``Examples`` section: it compares two full run records.
    """
    checks = {}
    checks["survival"] = {
        "ours": level_n1["survival"],
        "v2": v2_existence["survival_ours"],
        "match": list(level_n1["survival"]) == list(v2_existence["survival_ours"]),
    }
    for edge in ("lower", "upper"):
        got = level_n1[edge]
        for field, v2key in (("bracket", f"{edge}_bracket_ours"),
                             ("midpoint", f"{edge}_midpoint_ours"),
                             ("halfwidth", f"{edge}_halfwidth_ours")):
            a, b = got[field], v2_existence[v2key]
            checks[f"{edge}_{field}"] = {"ours": a, "v2": b, "match": a == b}
    # the bisection trace: same midpoints, same alive/dead decisions, in order
    v2_trace = {"lower": [], "upper": []}
    for step in v2_existence["bisect_trace"]:
        for key, edge in (("lo", "lower"), ("hi", "upper")):
            if key in step["ours"]:
                v2_trace[edge].append((step["ours"][key]["midpoint_over_kappa"],
                                       step["ours"][key]["single"]))
    for edge in ("lower", "upper"):
        mine = [(t["dw_over_kappa"], t["alive"]) for t in level_n1[edge]["trace"]]
        checks[f"{edge}_trace"] = {"ours": mine, "v2": v2_trace[edge],
                                   "match": mine == v2_trace[edge]}
    return {"reproduces_v2_n1": all(c["match"] for c in checks.values()),
            "per_field": checks}


def main(argv: list[str] | None = None) -> int:
    """Run the existence-edge refinement study from the command line.

    Parameters
    ----------
    argv : list of str or None, optional
        Command-line arguments; ``None`` (default) reads ``sys.argv[1:]``.
        Supports ``--levels``, ``--max-seconds``, ``--hold-block`` and
        ``--out-tag``.

    Returns
    -------
    int
        The process exit status: 0 when the study completed and its
        ``n_substeps = 1`` regression check reproduced the frozen v2 block,
        non-zero otherwise.

    Raises
    ------
    SystemExit
        From ``argparse`` on a malformed command line or ``--help``.
    AssertionError
        Propagated from :func:`run_level` for a non-finite field.
    FileNotFoundError
        If the frozen v2 artifacts are absent.

    Notes
    -----
    Writes ``validation/results/existence_convergence_<tag>.json`` and its
    fields archive. ``--hold-block`` adds the secondary axis: repeating the
    finest bisections at twice the hold time, which separates residual settle
    time from genuine discretization -- two causes that would otherwise both
    show up as edge drift.

    No ``Examples`` section: every path runs the full ladder.
    """
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--levels", default=DEFAULT_LEVELS,
                    help="n_substeps ladder for the edge bisection")
    ap.add_argument("--max-seconds", type=float, default=3600.0)
    ap.add_argument("--hold-block", action="store_true",
                    help="secondary axis: repeat the n=4 bisections at 2x the hold, "
                         "to separate residual settle time from discretization")
    ap.add_argument("--out-tag", default="ours")
    args = ap.parse_args(raw_argv)

    t_start = time.time()
    deadline = t_start + args.max_seconds

    from validation import criteria as CR
    from validation.pylle_crosscheck import (derived_config, load_device,
                                             soliton_seed)
    from validation.pylle_crosscheck_v2 import bracket_halfwidth

    if not V2_JSON.exists() or not V2_FIELDS.exists():
        print("ERROR: the v2 artifacts are required (read-only) and are missing.",
              file=sys.stderr)
        return 2
    v2 = json.loads(V2_JSON.read_text())
    v2_ex = v2["existence"]

    # ---- fixed setup, taken from the v2 run rather than re-derived -------
    dev = load_device()
    kappa = dev["kappa_i_rad_per_s"] + dev["kappa_c_rad_per_s"]
    t_r = float(v2["frame_and_timebase"]["t_r_s"])
    fsr_used = float(v2["frame_and_timebase"]["fsr_hz_used"])
    d2_local = float(v2["initialization"]["local_D2_rad_per_s2"])
    mu_half = int(v2["discretization"]["n_modes"] - 1) // 2
    n_modes = 2 * mu_half + 1
    n_roundtrips = int(v2["discretization"]["n_roundtrips_scan"])
    bisect_iters = int(v2_ex["bisect_iters"])
    scan_dws = [float(k) * kappa for k in v2_ex["scan_over_kappa"]]

    # R-fixed: the SAME dispersion array the v2 run integrated, read from the
    # committed npz rather than re-derived, so the ladder cannot drift onto a
    # different problem through a refit.
    with np.load(V2_FIELDS) as z:
        d_int_natural = np.asarray(z["d_int_rad_per_s"], dtype=float)
        mu = np.asarray(z["mu"])
        seed_v2 = np.asarray(z["seed_ours"], dtype=np.complex128)
    d_int_fftorder = np.fft.ifftshift(d_int_natural)

    import tempfile
    work = Path(tempfile.mkdtemp(prefix="edgeconv_"))
    cfg = derived_config(fsr_used, work)

    def seed_at(dw: float) -> np.ndarray:
        return soliton_seed(n_modes, dw, kappa, dev["kappa_c_rad_per_s"],
                            dev["gamma_LLE_per_J_per_s"], dev["pin_w"], d2_local)

    ctx = {"kappa": kappa, "t_r": t_r, "gamma": dev["gamma_LLE_per_J_per_s"],
           "dev": dev, "cfg": cfg, "d_int_fftorder": d_int_fftorder,
           "seed_at": seed_at}

    print("=" * 92)
    print("EXISTENCE-EDGE DISCRETIZATION UNCERTAINTY  (our solver only; pyLLE not run)")
    print("=" * 92)
    print(f"grid {n_modes} modes, hold {n_roundtrips} round trips, "
          f"bisect_iters {bisect_iters}, scan {v2_ex['scan_over_kappa']}")
    print(f"pre-committed thresholds: H-A at >= {H_A_THRESHOLD} kappa, "
          f"H-B at < {H_B_THRESHOLD} kappa")

    import jax

    levels_req = [int(x) for x in args.levels.split(",")]
    levels, all_fields = [], {}
    for n in levels_req:
        if time.time() > deadline:
            levels.append({"n_substeps": n, "status": "TIMEOUT", "wall_s": 0.0})
            print(f"[n={n}] TIMEOUT before start", flush=True)
            continue
        try:
            lv = run_level(n, ctx=ctx, scan_dws=scan_dws, bisect_iters=bisect_iters,
                           n_roundtrips=n_roundtrips, deadline=deadline)
        except Exception as exc:                      # noqa: BLE001
            # A level that dies must not take the completed levels with it. The
            # failure is recorded as a level status, never swallowed.
            levels.append({"n_substeps": n, "status": f"ERROR: {type(exc).__name__}",
                           "wall_s": 0.0, "detail": str(exc)[:600]})
            print(f"[n={n}] ERROR {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            continue
        all_fields.update(lv.pop("_fields"))
        levels.append(lv)
        print(f"[n={n}] {lv['status']} in {lv['wall_s']:.1f}s  "
              f"lower={lv.get('lower', {}).get('midpoint')}  "
              f"upper={lv.get('upper', {}).get('midpoint')}", flush=True)
        # Every distinct hold length compiles its own executable; over a full
        # ladder that is dozens of large programs, and the JIT can run out of
        # room to materialize them. Dropping the caches between levels keeps the
        # ladder from failing for a reason that has nothing to do with physics.
        jax.clear_caches()

    by_n = {lv["n_substeps"]: lv for lv in levels if lv.get("status") == "OK"}

    # ---- R1 regression gate ---------------------------------------------
    gate = {"reproduces_v2_n1": False, "per_field": {},
            "note": "n_substeps=1 must reproduce the v2 ours-side existence block exactly"}
    if 1 in by_n:
        gate = check_regression(by_n[1], v2_ex)
        gate["note"] = ("n_substeps=1 must reproduce the v2 ours-side existence "
                        "block exactly")
    print(f"\n[R1 regression gate] reproduces v2 at n=1: {gate['reproduces_v2_n1']}")
    if not gate["reproduces_v2_n1"]:
        for k, v in gate["per_field"].items():
            if not v["match"]:
                print(f"   MISMATCH {k}: ours={v['ours']} v2={v['v2']}")

    # ---- edge convergence -------------------------------------------------
    edge_conv = {}
    for edge in ("lower", "upper"):
        def mid(n):
            lv = by_n.get(n)
            return None if not lv else lv[edge]["midpoint"]
        finest = max(by_n) if by_n else None
        hw = None if finest is None else by_n[finest][edge]["halfwidth"]
        edge_conv[edge] = edge_uncertainty(mid(1), mid(2), mid(4), hw)

    hypothesis = select_hypothesis(edge_conv["lower"]["drift_1_to_4"],
                                   edge_conv["upper"]["drift_1_to_4"])

    # ---- recomputed G7 ----------------------------------------------------
    g7 = {"pylle_u_disc_not_measured": True,
          "asymmetry_note": (
              "U_disc is measured for OUR edge only. This task does not refine "
              "pyLLE's existence bisection (it runs no Julia), so pyLLE "
              "contributes no discretization term and the recomputed band is "
              "asymmetric. A pyLLE-side term could only widen the band further, "
              "so a FAIL under this band is conservative and a PASS is not.")}
    for edge in ("lower", "upper"):
        g7[edge] = recompute_g7(
            edge, v2_ex[f"{edge}_bracket_ours"],
            by_n[max(by_n)][edge]["bracket"] if by_n else None,
            v2_ex[f"{edge}_bracket_pylle"], edge_conv[edge]["U_disc"],
            CR.COVERAGE_FACTOR)
        g7[edge]["refined_bracket_source"] = (
            f"this task, n_substeps={max(by_n)}" if by_n else None)
        g7[edge]["v2_bracket_source"] = "pylle_crosscheck_v2.json, n_substeps=1"

    # ---- secondary axis: hold length at the finest level -------------------
    hold_block = None
    if args.hold_block and by_n and time.time() < deadline:
        n_fine = max(by_n)
        print(f"\n[secondary] repeating the n={n_fine} bisections at 2x the hold",
              flush=True)
        jax.clear_caches()
        try:
            lv2 = run_level(n_fine, ctx=ctx, scan_dws=scan_dws,
                            bisect_iters=bisect_iters, n_roundtrips=2 * n_roundtrips,
                            deadline=deadline)
            all_fields.update({f"hold2x_{k}": v for k, v in lv2.pop("_fields").items()})
            hold_block = {
                "n_substeps": n_fine, "n_roundtrips_scan": 2 * n_roundtrips,
                "status": lv2["status"],
                "lower": lv2.get("lower"), "upper": lv2.get("upper"),
            }
        except Exception as exc:                      # noqa: BLE001
            # The secondary axis is explicitly optional; losing it must not lose
            # the primary measurement, and its failure is recorded rather than
            # hidden.
            hold_block = {"n_substeps": n_fine,
                          "n_roundtrips_scan": 2 * n_roundtrips,
                          "status": f"ERROR: {type(exc).__name__}",
                          "detail": str(exc)[:600]}
            print(f"[secondary] ERROR {type(exc).__name__}: {str(exc)[:200]}",
                  flush=True)
        hold_block["note"] = (
            "separates residual settle time from discretization; kept OUT of "
            "U_disc, which is a discretization quantity only")

    # ---- provenance -------------------------------------------------------
    import scipy
    prov = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "argv": raw_argv,
        "command": "python -m validation.existence_convergence " + " ".join(raw_argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "scipy": scipy.__version__, "jax": jax.__version__},
        "physical_parameters": dev,
        "derived_config": {
            "filename": Path(cfg).name, "sha256": _sha_file(Path(cfg)),
            "overrides": {"dn_dT_per_k": 0.0, "fsr_hz": fsr_used},
            "fsr_hz_config_leaf": dev["fsr_hz"],
        },
        "solver_flags": {
            "fine_cadence_M": 1, "symmetric_drive": False,
            "dealias_two_thirds": False, "edge_absorber": False,
            "dispersion_validity_mask": False,
            "noise_config": "NoiseConfig.all_off()",
            "dn_dT_per_k": 0.0,
            "rng_key": "jax.random.PRNGKey(0), unused -- every noise channel off",
            "n_substeps_ladder": levels_req,
        },
        "d_int_source": {"path": str(V2_FIELDS), "sha256": _sha_file(V2_FIELDS),
                         "npz_key": "d_int_rad_per_s",
                         "note": "read from the frozen v2 artifact, not re-derived"},
        "v2_reference": {"path": str(V2_JSON), "sha256": _sha_file(V2_JSON),
                         "note": "read-only; pyLLE values are reference points only"},
        "seed_rule": "deterministic sech on the lower CW branch, "
                     "A^2 = 2*delta_omega/gamma_LLE, w = sqrt(|D2|/(2*delta_omega))",
        "seed_sha256_per_detuning": {
            f"{dw / kappa:.4f}": _sha_arr(seed_at(dw)) for dw in scan_dws},
        "seed_sha256_v2_main_run": _sha_arr(seed_v2),
        "classifier": EDGE_DEFINITION,
        "edge_observable_version": EDGE_OBSERVABLE_VERSION,
        "wall_clock_total_s": time.time() - t_start,
        "level_wall_s": {str(lv["n_substeps"]): lv.get("wall_s") for lv in levels},
    }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "provenance": prov,
        "regression_gate": gate,
        "fixed_setup": {
            "mu_half": mu_half, "n_modes": n_modes,
            "n_roundtrips_scan": n_roundtrips, "bisect_iters": bisect_iters,
            "scan_over_kappa": v2_ex["scan_over_kappa"],
            "kappa_rad_per_s": kappa, "t_r_s": t_r,
            "local_D2_rad_per_s2": d2_local,
            "edge_definition": EDGE_DEFINITION,
            "edge_observable_version": EDGE_OBSERVABLE_VERSION,
        },
        "hypotheses_pre_committed": {
            "H-A": f"edge moves >= {H_A_THRESHOLD} kappa over n=1..4 -> the v2 G7 "
                   f"FAIL is a resolution limit, not a code disagreement",
            "H-B": f"edge moves < {H_B_THRESHOLD} kappa -> the disagreement is not "
                   f"our discretization; standing open item, do NOT widen G7",
            "H-C": "intermediate -> report the measured value and the recomputed "
                   "verdict whatever it is",
        },
        "levels": [{k: v for k, v in lv.items() if not k.startswith("_")}
                   for lv in levels],
        "edge_convergence": edge_conv,
        "hold_time_block": hold_block,
        "g7_recomputed": g7,
        "hypothesis_outcome": hypothesis,
    }

    npz_path = RESULTS_DIR / f"existence_convergence_{args.out_tag}_fields.npz"
    arrays = dict(all_fields)
    arrays.update({"mu": mu, "d_int_rad_per_s": d_int_natural,
                   "t_r_s": np.array([t_r]), "kappa_rad_per_s": np.array([kappa]),
                   "scan_delta_omega": np.asarray(scan_dws),
                   "seed_v2_main": seed_v2})
    for dw in scan_dws:
        arrays[f"seed_{dw / kappa:.4f}"] = seed_at(dw)
    np.savez_compressed(npz_path, **arrays)
    payload["fields_npz"] = {"path": str(npz_path), "sha256": _sha_file(npz_path)}
    payload["numerical_digest"] = numerical_digest(payload)

    json_path = RESULTS_DIR / f"existence_convergence_{args.out_tag}.json"
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {json_path}\nwrote {npz_path}")

    _print_summary(payload)
    if not gate["reproduces_v2_n1"]:
        print("\nF1: the regression gate FAILED. Nothing downstream is "
              "interpretable; no other change was made.", file=sys.stderr)
        return 1
    n_ok = sum(1 for lv in levels if lv.get("status") == "OK")
    if n_ok < 2:
        print(f"\nF3: only {n_ok} level(s) completed.", file=sys.stderr)
        return 1
    return 0


def _print_summary(p: dict) -> None:
    print("\n" + "=" * 92)
    print("EDGE MIDPOINTS BY REFINEMENT LEVEL  (kappa)")
    print("-" * 92)
    print(f"{'n_substeps':<12}{'lower mid':>14}{'lower hw':>12}"
          f"{'upper mid':>14}{'upper hw':>12}   status")
    for lv in p["levels"]:
        if lv.get("status") != "OK":
            print(f"{lv['n_substeps']:<12}{'-':>14}{'-':>12}{'-':>14}{'-':>12}   "
                  f"{lv.get('status')}")
            continue
        lo, hi = lv["lower"], lv["upper"]
        print(f"{lv['n_substeps']:<12}{lo['midpoint']:>14.6f}{lo['halfwidth']:>12.6f}"
              f"{hi['midpoint']:>14.6f}{hi['halfwidth']:>12.6f}   {lv['status']}")

    print("\nEDGE DISCRETIZATION UNCERTAINTY")
    print("-" * 92)
    for edge in ("lower", "upper"):
        e = p["edge_convergence"][edge]
        r = e.get("richardson") or {}
        print(f"  {edge}: e(1)={e['e1']} e(2)={e['e2']} e(4)={e['e4']}")
        print(f"     drift 1->4 = {e['drift_1_to_4']}   drift 2->4 = {e['drift_2_to_4']}")
        print(f"     U_disc = {e['U_disc']}  (from {e['u_disc_source']}; "
              f"richardson status {r.get('status')})")

    print("\nG7 RECOMPUTED  (pyLLE side NOT refined -- asymmetric band)")
    print("-" * 92)
    print(f"{'edge':<8}{'variant':<24}{'|ours-pylle|':>14}{'band':>11}"
          f"{'overlap':>9}  verdict")
    for edge in ("lower", "upper"):
        g = p["g7_recomputed"][edge]
        if g.get("status") == "NOT_MEASURED":
            print(f"{edge:<8}NOT_MEASURED")
            continue
        for name in ("v2_as_measured", "refined_no_u_disc", "refined_with_u_disc"):
            v = g[name]
            print(f"{edge:<8}{name:<24}{v['abs_diff']:>14.6f}{v['band']:>11.6f}"
                  f"{str(v['brackets_overlap']):>9}  {v['verdict']}")
        print(f"        -> change attributed to: {g['verdict_change_attributed_to']}")
        if g.get("u_disc_double_counts_halfwidth"):
            print("        -> NOTE: U_disc is floored at the bracket half-width, "
                  "which therefore enters twice (conservative)")

    h = p["hypothesis_outcome"]
    print(f"\nPRE-COMMITTED HYPOTHESIS SELECTED: {h['outcome']}")
    print(f"   {h['reason']}")
    print("=" * 92)


if __name__ == "__main__":
    sys.exit(main())
