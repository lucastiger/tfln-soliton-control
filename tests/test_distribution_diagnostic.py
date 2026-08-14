"""Per-seed distribution diagnostic: catch a mean that no realization produced.

``mean +- std`` is a faithful summary only of a unimodal sample. The 24-seed W1
run produced a per-seed 3 dB span distribution that splits into two tight,
widely separated clusters, and its mean landed in the empty gap between them --
so the committed summary "ON 7023.05 +/- 65.94 GHz" describes a value that NOT
ONE of the 24 realizations came within 10 GHz of, and the std measures the
cluster separation rather than any run-to-run uncertainty.

That is a wrong number which looks entirely normal in a report, so
``analysis.noise_validation_campaign._distribution_diagnostic`` runs on every
per-seed sample and flags it. These tests pin that it fires on the real observed
data, stays quiet on ordinary unimodal samples, and does not divide by zero on
the degenerate inputs a small ensemble can produce.
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.noise_validation_campaign import (
    CLUSTER_GAP_RANGE_FRAC,
    CLUSTER_GAP_RATIO,
    CLUSTER_MIN_N,
    _distribution_diagnostic,
    _format_cluster_warning,
)

#: The 24 per-seed 3 dB spans [GHz] from the production W1 run that exposed this
#: (seeds 1..24, in order). OFF reference: 6986.75 GHz.
W1_SPANS_GHZ = [
    6982.00, 6986.27, 7134.14, 6988.30, 7134.73, 6985.73, 6987.76, 6989.26,
    6981.81, 6985.13, 6981.80, 7133.75, 6986.72, 7137.42, 7134.22, 6987.18,
    6985.26, 6986.31, 6987.24, 6988.50, 6986.79, 7134.49, 6982.62, 6985.69,
]
W1_SPAN_OFF_GHZ = 6986.747130284056


def test_flags_the_observed_w1_span_clustering():
    """The real 24-seed span sample is flagged, and split 18/6 as observed."""
    d = _distribution_diagnostic(W1_SPANS_GHZ, "three_db_span_ghz_on")

    assert d["clustered"] is True
    assert d["n"] == 24
    lo, hi = d["clusters"]
    assert (lo["n"], hi["n"]) == (18, 6)
    assert lo["mean"] == pytest.approx(6985.80, abs=0.05)
    assert hi["mean"] == pytest.approx(7134.79, abs=0.05)

    # The separation is overwhelming: a ~144 GHz gap against a ~2 GHz spread.
    assert d["cluster_gap"] == pytest.approx(144.49, abs=0.05)
    assert d["cluster_gap_over_within_std"] > 50.0
    assert max(lo["std"], hi["std"]) < 3.0

    # The headline pathology: the mean is in the gap, far from every sample.
    assert d["mean"] == pytest.approx(7023.05, abs=0.05)
    assert d["closest_seed_to_mean_abs"] > 30.0
    assert min(abs(s - d["mean"]) for s in W1_SPANS_GHZ) > 30.0

    # The median is a faithful summary here and sits on the OFF value, i.e. the
    # large cluster shows no broadening at all.
    assert d["median"] == pytest.approx(6986.99, abs=0.05)
    assert abs(d["median"] - W1_SPAN_OFF_GHZ) < 1.0
    # ...whereas the mean would be quoted as a +36 GHz broadening.
    assert d["mean"] - W1_SPAN_OFF_GHZ == pytest.approx(36.3, abs=0.1)


def test_false_positive_rate_on_unimodal_samples():
    """Measured false-positive rate on unimodal samples, not a spot check.

    A tripwire that cried wolf would be worse than none -- it would train the
    reader to ignore it. This measures the rate over many trials for the three
    shapes a per-seed sample plausibly takes, at the ensemble sizes the campaign
    actually runs, and holds it under 1%. (Simulated at authoring time: 0/4000
    for Gaussian and uniform at n = 24, <= 0.05% exponential.)
    """
    rng = np.random.default_rng(20260814)
    trials = 500
    for n in (CLUSTER_MIN_N, 24, 64):
        for name, draw in (
            ("gaussian", lambda: rng.normal(7000.0, 60.0, size=n)),
            ("uniform", lambda: rng.uniform(6900.0, 7100.0, size=n)),
            ("exponential", lambda: rng.exponential(60.0, size=n)),
        ):
            fp = sum(_distribution_diagnostic(draw())["clustered"]
                     for _ in range(trials))
            assert fp / trials < 0.01, (n, name, fp, trials)


def test_never_fires_below_the_minimum_sample_size():
    """Under CLUSTER_MIN_N the test is not run at all, however split the data.

    Quick runs use 2-3 seeds; the gap statistic is not usable there and a
    confident-looking warning off 3 points would be noise.
    """
    obviously_split = [0.0, 0.1, 0.2, 1000.0, 1000.1, 1000.2]
    assert len(obviously_split) < CLUSTER_MIN_N
    assert _distribution_diagnostic(obviously_split)["clustered"] is False


def test_detects_a_grossly_separated_planted_split():
    """A wide planted split is caught, a narrow one is not."""
    rng = np.random.default_rng(11)
    wide = np.concatenate([rng.normal(0.0, 1.0, 16), rng.normal(40.0, 1.0, 8)])
    d = _distribution_diagnostic(wide)
    assert d["clustered"] is True
    assert d["cluster_gap_over_within_std"] >= CLUSTER_GAP_RATIO
    assert d["cluster_gap_over_range"] >= CLUSTER_GAP_RANGE_FRAC
    assert sorted(c["n"] for c in d["clusters"]) == [8, 16]

    narrow = np.concatenate([rng.normal(0.0, 1.0, 16), rng.normal(2.0, 1.0, 8)])
    assert _distribution_diagnostic(narrow)["clustered"] is False


def test_degenerate_and_small_inputs_do_not_raise():
    """Small, constant and near-constant samples return cleanly.

    A quick run uses as few as 2-3 seeds, and an all-identical sample makes the
    within-cluster std exactly 0 -- the division that would otherwise blow up.
    """
    for values in ([], [1.0], [1.0, 2.0], [1.0, 2.0, 3.0]):
        d = _distribution_diagnostic(values)
        assert d["clustered"] is False
        assert d["n"] == len(values)
        assert d["clusters"] is None

    const = _distribution_diagnostic([5.0] * 12)
    assert const["clustered"] is False          # zero gap, nothing to split
    assert const["std"] == 0.0
    assert const["closest_seed_to_mean_abs"] == pytest.approx(0.0)

    # Constant except one outlier: within-std is 0 on one side, so the guard
    # must keep the ratio finite-or-inf without raising, and the minimum
    # cluster size (>= 2) must prevent a single point being called a cluster.
    one_out = _distribution_diagnostic([5.0] * 11 + [500.0])
    assert one_out["clustered"] is False
    assert np.isfinite(one_out["mean"])


def test_cluster_warning_renders_on_the_real_data():
    """The gate warning formats cleanly, and says the thing that matters.

    This branch only executes on a genuinely clustered ensemble, i.e. in the
    middle of a production run. Rendering it here means a bad format field fails
    in CI rather than after a 5-minute GPU job.
    """
    d = _distribution_diagnostic(W1_SPANS_GHZ, "three_db_span_ghz_on")
    lines = _format_cluster_warning(d, "GHz")
    assert lines, "the real clustered sample must produce a warning"
    text = "\n".join(lines)

    assert "DO NOT QUOTE mean +/- std" in text
    assert "three_db_span_ghz_on" in text
    assert "18/24" in text and "6/24" in text          # the observed split
    assert "6986.99" in text                            # the median it points to
    assert "GHz" in text
    assert "{" not in text and "}" not in text          # no unformatted fields

    # Quiet path returns nothing at all, so the gate prints nothing.
    assert _format_cluster_warning(
        _distribution_diagnostic(np.linspace(0.0, 1.0, 24))) == []
    assert _format_cluster_warning({"clustered": False}) == []
    assert _format_cluster_warning({}) == []

    # Unitless rendering (the comb-fraction case) must not leave a dangling space.
    unitless = "\n".join(_format_cluster_warning(d))
    assert "  GHz" not in unitless


def test_summary_fields_are_json_ready():
    """Every value is a plain Python scalar/list, so json.dumps cannot fail."""
    import json

    d = _distribution_diagnostic(W1_SPANS_GHZ, "spans")
    json.dumps(d)                                    # must not raise
    for key in ("n", "mean", "std", "median", "iqr", "min", "max",
                "clustered", "cluster_gap", "cluster_gap_over_within_std",
                "closest_seed_to_mean_abs", "label"):
        assert key in d
    assert isinstance(d["n"], int) and isinstance(d["clustered"], bool)
