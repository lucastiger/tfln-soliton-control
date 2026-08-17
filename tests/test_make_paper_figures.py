"""Unit tests for the manuscript figure pipeline (``scripts/make_paper_figures.py``).

Covers the registry contract and the driver's failure behaviour -- the parts
that decide whether ``--all`` is safe to put in a Makefile -- without building
the expensive figures. The end-to-end build is exercised by
``python scripts/make_paper_figures.py --all`` (gate G5).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.make_paper_figures import (
    FIGURES,
    PNG_DPI,
    REPO_ROOT,
    BuildContext,
    FigureSpec,
    _artifact_record,
    _budget_headline_point,
    _sha256_file,
    build_figure,
    main,
)


# ---------------------------------------------------------------------------
# Registry contract
# ---------------------------------------------------------------------------
def test_all_eight_manuscript_ids_are_registered():
    """The ids are fixed and are the ones used in the manuscript."""
    assert set(FIGURES) == {
        "fig1_verification", "fig2_numerics", "fig3_pylle", "fig4_budget",
        "fig5_srep", "fig6_runtime", "fig7_experiment", "fig8_quietpoint",
    }


def test_registry_key_matches_its_spec_id():
    for fig_id, spec in FIGURES.items():
        assert spec.fig_id == fig_id


def test_every_figure_emits_both_pdf_and_png(tmp_path):
    """Vector for the manuscript, raster for drafts -- always both."""
    for spec in FIGURES.values():
        pdf, png = spec.outputs(tmp_path)
        assert pdf.suffix == ".pdf"
        assert png.suffix == ".png"
        assert pdf.stem == png.stem == spec.fig_id


def test_png_is_300_dpi():
    assert PNG_DPI == 300


def test_every_figure_has_a_sidecar_path(tmp_path):
    for spec in FIGURES.values():
        assert spec.sidecar(tmp_path).name.endswith(".provenance.json")


def test_every_builder_is_callable_and_described():
    for spec in FIGURES.values():
        assert callable(spec.builder)
        assert spec.description.strip()


def test_recompute_figures_declare_a_budget():
    """The 60 s bar has to stay visible, not be remembered."""
    for spec in FIGURES.values():
        if spec.recompute:
            assert spec.recompute_budget_s is not None
            assert spec.recompute_budget_s < 60.0, spec.fig_id


def test_non_recompute_figures_read_artifacts_only():
    """Anything not marked recompute must not claim a solver budget."""
    for spec in FIGURES.values():
        if not spec.recompute:
            assert spec.recompute_budget_s is None, spec.fig_id


def test_inputs_are_absolute_paths_inside_the_repo():
    for spec in FIGURES.values():
        for path in list(spec.inputs) + list(spec.optional_inputs):
            assert path.is_absolute(), spec.fig_id
            assert REPO_ROOT in path.parents, spec.fig_id


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def test_artifact_record_hashes_a_present_file(tmp_path):
    path = tmp_path / "a.json"
    path.write_text('{"x": 1}', encoding="utf-8")
    # _artifact_record reports paths relative to the repo root.
    rec = _artifact_record(REPO_ROOT / "config" / "sin_params.yaml")
    assert rec["present"] is True
    assert len(rec["sha256"]) == 64
    assert rec["bytes"] > 0
    assert rec["mtime_utc"].endswith("Z")


def test_artifact_record_marks_a_missing_file_without_raising():
    rec = _artifact_record(REPO_ROOT / "does" / "not" / "exist.npz")
    assert rec["present"] is False
    assert "sha256" not in rec


def test_sha256_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "blob.bin"
    payload = b"soliton" * 1000
    path.write_bytes(payload)
    assert _sha256_file(path) == hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Driver behaviour
# ---------------------------------------------------------------------------
def _spec(builder, inputs=(), optional=()):
    return FigureSpec(fig_id="figX", builder=builder, inputs=tuple(inputs),
                      optional_inputs=tuple(optional), description="test")


def test_missing_required_input_skips_and_never_raises(tmp_path):
    spec = _spec(lambda ctx: (_ for _ in ()).throw(AssertionError("built!")),
                 inputs=[tmp_path / "absent.npz"])
    result = build_figure(spec, tmp_path)
    assert result.status == "skipped"
    assert "absent.npz" in result.message
    assert result.outputs == []


def test_builder_exception_is_reported_as_failed_not_propagated(tmp_path):
    def boom(ctx):
        raise RuntimeError("kaboom")

    result = build_figure(_spec(boom), tmp_path)
    assert result.status == "failed"
    assert "kaboom" in result.message


def test_builder_filenotfound_is_a_skip_not_a_failure(tmp_path):
    """An optional dataset that does not exist is a fact, not a bug."""
    def missing(ctx):
        raise FileNotFoundError("no measured dataset is committed")

    result = build_figure(_spec(missing), tmp_path)
    assert result.status == "skipped"
    assert "measured dataset" in result.message


def test_successful_build_writes_pdf_png_and_sidecar(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def ok(ctx):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        return fig, {"hello": "world"}

    spec = _spec(ok, inputs=[REPO_ROOT / "config" / "sin_params.yaml"])
    result = build_figure(spec, tmp_path)
    assert result.status == "built"
    assert (tmp_path / "figX.pdf").exists()
    assert (tmp_path / "figX.png").exists()
    sidecar = json.loads((tmp_path / "figX.provenance.json").read_text())
    assert sidecar["figure_id"] == "figX"
    assert sidecar["notes"]["hello"] == "world"
    assert sidecar["input_artifacts"][0]["present"] is True
    assert len(sidecar["input_artifacts"][0]["sha256"]) == 64


def test_missing_panels_downgrade_to_partial_not_failure(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def partial(ctx):
        ctx.missing_panels["panel_b"] = "study not committed"
        fig, ax = plt.subplots()
        return fig, {}

    result = build_figure(_spec(partial), tmp_path)
    assert result.status == "partial"
    assert result.missing_panels == {"panel_b": "study not committed"}
    # A partial figure is still WRITTEN -- a manuscript figure must not vanish
    # because one panel's input was absent.
    assert (tmp_path / "figX.pdf").exists()
    sidecar = json.loads((tmp_path / "figX.provenance.json").read_text())
    assert sidecar["status"] == "partial"


def test_check_only_does_not_write_anything(tmp_path):
    def boom(ctx):
        raise AssertionError("builder must not run under --check")

    spec = _spec(boom, inputs=[REPO_ROOT / "config" / "sin_params.yaml"])
    result = build_figure(spec, tmp_path, check_only=True)
    assert result.status == "ok"
    assert not list(tmp_path.iterdir())


def test_check_reports_missing_optional_inputs(tmp_path):
    spec = _spec(lambda ctx: None,
                 inputs=[REPO_ROOT / "config" / "sin_params.yaml"],
                 optional=[tmp_path / "nope.json"])
    result = build_figure(spec, tmp_path, check_only=True)
    assert result.status == "ok"
    assert "optional absent" in result.message


# ---------------------------------------------------------------------------
# Budget headline-point selection (fig4/fig5)
# ---------------------------------------------------------------------------
def _cells(**by_set):
    return {
        name: {"obs": {"points": {
            point: {"statistics": {"mean": value}}
            for point, value in points.items()}}}
        for name, points in by_set.items()
    }


def test_headline_point_prefers_the_most_plottable():
    cells = _cells(a={"p1": 1.0, "p2": None}, b={"p1": 2.0, "p2": 3.0})
    assert _budget_headline_point(cells, "obs") == "p1"


def test_headline_point_rejects_all_zero_points():
    """Zero is valid data but invisible on the log axis this figure uses."""
    cells = _cells(a={"p1": 0.0}, b={"p1": 0.0})
    assert _budget_headline_point(cells, "obs") is None


def test_headline_point_rejects_all_null_points():
    cells = _cells(a={"p1": None}, b={"p1": None})
    assert _budget_headline_point(cells, "obs") is None


def test_headline_point_ignores_negative_and_non_finite():
    cells = _cells(a={"p1": -1.0, "p2": 5.0}, b={"p1": float("nan"), "p2": 6.0})
    assert _budget_headline_point(cells, "obs") == "p2"


def test_headline_point_none_for_unknown_observable():
    assert _budget_headline_point(_cells(a={"p1": 1.0}), "missing") is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_list_exits_zero(capsys):
    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    for fig_id in FIGURES:
        assert fig_id in out


def test_no_action_is_an_error():
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code != 0


def test_check_exits_zero_even_with_a_skipped_figure(tmp_path, capsys):
    """A skip is a stated fact about the checkout, not a failure."""
    assert main(["--check", "--out", str(tmp_path)]) == 0
    assert "fig7_experiment" in capsys.readouterr().out


def test_unknown_figure_id_is_rejected():
    with pytest.raises(SystemExit):
        main(["--figure", "fig99_nonexistent"])
