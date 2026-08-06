"""Tests for :mod:`simulator.equation_map`.

The map is only worth citing in a paper if it cannot silently rot, so every
claim it makes is checked mechanically:

* the channel set matches :class:`NoiseConfig`'s switch fields EXACTLY, via
  ``dataclasses.fields`` -- adding a channel without mapping it fails here;
* every ``code_symbol`` actually imports;
* every ``code_file`` actually exists;
* every name in ``validated_by`` is a real test function (found by parsing the
  test suite with :mod:`ast` -- never by importing it, which would execute
  module-level solver calls);
* ``shares_random_source_with`` is symmetric.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
from pathlib import Path

import pytest

from simulator.equation_map import (
    ADDITIVITY,
    CHANNEL_EQUATIONS,
    ChannelEquation,
    ENTERS_AS,
    render_latex,
    render_markdown,
)
from simulator.equation_map import _COLUMNS as _COLUMNS_IN_TABLE
from simulator.noise_config import PARAMETER_BOOL_DEFAULTS, SWITCH_FIELDS, NoiseConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"

ENTRIES = tuple(CHANNEL_EQUATIONS.values())
IDS = tuple(CHANNEL_EQUATIONS)


def _switch_bool_fields() -> set[str]:
    """Boolean NoiseConfig fields that are SWITCHES, straight off the dataclass.

    Enumerated with ``dataclasses.fields`` rather than from a hand-written
    list, so a newly added channel shows up here automatically. The boolean
    *parameter* fields (``PARAMETER_BOOL_DEFAULTS`` -- e.g.
    ``quantum_seed_vacuum_init``) are excluded: they tune a channel that is
    already mapped, they are not channels of their own. That split is itself
    enforced by ``tests/test_noise_config.py``, so it cannot drift silently.
    """
    bool_fields = {
        f.name for f in dataclasses.fields(NoiseConfig) if f.type in ("bool", bool)
    }
    return bool_fields - set(PARAMETER_BOOL_DEFAULTS)


# ---------------------------------------------------------------------------
# channel set
# ---------------------------------------------------------------------------
def test_every_switch_field_has_exactly_one_entry() -> None:
    switches = _switch_bool_fields()

    missing = switches - set(CHANNEL_EQUATIONS)
    extra = set(CHANNEL_EQUATIONS) - switches

    assert not missing, (
        f"NoiseConfig switch field(s) with no CHANNEL_EQUATIONS entry: "
        f"{sorted(missing)}. A channel that is not mapped cannot be cited in "
        f"the paper -- add it to simulator/equation_map.py."
    )
    assert not extra, (
        f"CHANNEL_EQUATIONS entries that are not NoiseConfig switch fields: "
        f"{sorted(extra)}."
    )


def test_switch_field_enumeration_matches_declared_constant() -> None:
    """Cross-check the dataclass walk against noise_config's own SWITCH_FIELDS."""
    assert _switch_bool_fields() == set(SWITCH_FIELDS)


def test_all_eight_channels_present() -> None:
    assert len(CHANNEL_EQUATIONS) == 8, sorted(CHANNEL_EQUATIONS)


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_key_matches_channel_field(entry: ChannelEquation) -> None:
    assert CHANNEL_EQUATIONS[entry.channel] is entry
    assert hasattr(NoiseConfig(), entry.channel), (
        f"{entry.channel!r} is not a NoiseConfig field."
    )


# ---------------------------------------------------------------------------
# code references
# ---------------------------------------------------------------------------
def _resolve(dotted: str):
    """Import ``a.b.c`` by walking module boundaries then attributes.

    ``PumpNoise.sample_freq`` is a method, so the split between module path and
    attribute chain is not simply the last dot.
    """
    parts = dotted.split(".")
    for split in range(len(parts) - 1, 0, -1):
        module_path = ".".join(parts[:split])
        try:
            obj = importlib.import_module(module_path)
        except ImportError:
            continue
        for attr in parts[split:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(f"no importable module prefix in {dotted!r}")


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_code_symbol_is_importable(entry: ChannelEquation) -> None:
    obj = _resolve(entry.code_symbol)
    assert obj is not None
    assert callable(obj) or isinstance(obj, type), (
        f"{entry.code_symbol} resolved to a non-callable {type(obj)}."
    )


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_code_file_exists(entry: ChannelEquation) -> None:
    path = REPO_ROOT / entry.code_file
    assert path.is_file(), f"{entry.code_file} does not exist."


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_code_symbol_lives_in_code_file(entry: ChannelEquation) -> None:
    """The dotted path and the declared file must agree."""
    module_path = entry.code_symbol.rsplit(".", 1)[0]
    expected = entry.code_file[: -len(".py")].replace("/", ".")
    assert module_path.startswith(expected), (
        f"{entry.code_symbol} does not live in {entry.code_file}."
    )


# ---------------------------------------------------------------------------
# validated_by
# ---------------------------------------------------------------------------
def _test_function_names() -> dict[str, str]:
    """{test function name: file it lives in}, parsed with ast.

    Deliberately NOT imported: several test modules build solver fixtures at
    import time, and collecting names must stay cheap and side-effect free.
    """
    found: dict[str, str] = {}
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_"):
                    found.setdefault(node.name, path.name)
    return found


def test_ast_scan_finds_this_files_own_tests() -> None:
    """Guard the guard: a broken parser would make every check below vacuous."""
    names = _test_function_names()
    assert "test_ast_scan_finds_this_files_own_tests" in names
    assert len(names) > 100, f"only {len(names)} tests found -- scan is broken."


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_validated_by_names_exist(entry: ChannelEquation) -> None:
    names = _test_function_names()
    assert entry.validated_by, f"{entry.channel} claims no validating test."
    missing = [name for name in entry.validated_by if name not in names]
    assert not missing, (
        f"{entry.channel}: validated_by names no such test: {missing}. "
        f"Either the test was renamed or the map is wrong."
    )


# ---------------------------------------------------------------------------
# random-source sharing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_shares_random_source_is_symmetric(entry: ChannelEquation) -> None:
    for other in entry.shares_random_source_with:
        assert other in CHANNEL_EQUATIONS, (
            f"{entry.channel} shares a source with unknown channel {other!r}."
        )
        back = CHANNEL_EQUATIONS[other].shares_random_source_with
        assert entry.channel in back, (
            f"asymmetric sharing: {entry.channel} -> {other}, but "
            f"{other} -> {sorted(back)}."
        )


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_channel_does_not_share_with_itself(entry: ChannelEquation) -> None:
    assert entry.channel not in entry.shares_random_source_with


def test_delta_t_channels_form_one_sharing_group() -> None:
    """trn, pyro_eo and fsr all consume the SAME delta_T realization.

    TRN and pyro_eo share it by direct reuse of one array in
    ``TotalNoise.sample_with_delta_t``; fsr shares it by deterministic
    regeneration from the same ``noise_keys``. TCCR takes the independent
    second subkey and must NOT appear in the group.
    """
    group = {"trn", "pyro_eo", "fsr"}
    for name in group:
        assert set(CHANNEL_EQUATIONS[name].shares_random_source_with) == group - {
            name
        }, f"{name} sharing group is wrong."

    assert CHANNEL_EQUATIONS["tccr"].shares_random_source_with == ()
    for independent in ("quantum_vacuum", "pump_freq_noise", "pump_rin"):
        assert CHANNEL_EQUATIONS[independent].shares_random_source_with == ()


# ---------------------------------------------------------------------------
# field hygiene
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_controlled_vocabularies(entry: ChannelEquation) -> None:
    assert entry.enters_as in ENTERS_AS, entry.enters_as
    assert entry.additive_or_multiplicative in ADDITIVITY
    assert entry.arxiv_id == "2604.05897"
    assert entry.arxiv_version == "v1", (
        "the arXiv version pin is mandatory: equation and section numbers "
        "shift between versions."
    )


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_required_prose_fields_are_populated(entry: ChannelEquation) -> None:
    """paper_eq/paper_section MAY be empty (unrecorded); the rest may not."""
    for field in ("equation_latex", "discrete_latex", "colour", "units"):
        assert getattr(entry, field).strip(), f"{entry.channel}.{field} is empty."


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def test_render_markdown_contains_every_channel() -> None:
    md = render_markdown()
    for name in CHANNEL_EQUATIONS:
        assert name in md, f"{name} missing from the Markdown table."
    # One header row + one separator + one row per channel.
    table_rows = [ln for ln in md.splitlines() if ln.startswith("| ")]
    assert len(table_rows) == len(CHANNEL_EQUATIONS) + 1


def _delimiter_count(line: str) -> int:
    r"""Cell delimiters in a Markdown table row.

    Escaped pipes (``\|``) are literal content INSIDE a cell -- e.g. the
    ``lorentzian(tau_th) | kondratiev_gorodetsky`` colour -- so they are
    removed before counting, otherwise a correctly escaped row reads as ragged.
    """
    return line.replace(r"\|", "").count("|")


def test_render_markdown_cells_do_not_break_the_table() -> None:
    """Every row must have the same number of unescaped cell delimiters."""
    md = render_markdown()
    header = next(ln for ln in md.splitlines() if ln.startswith("| Channel"))
    expected = _delimiter_count(header)
    assert expected == len(_COLUMNS_IN_TABLE) + 1

    for line in md.splitlines():
        if line.startswith("| ") and not line.startswith("| Channel"):
            assert _delimiter_count(line) == expected, f"ragged row: {line}"


def test_escaped_pipes_are_present_where_content_needs_them() -> None:
    r"""Guard the guard: a renderer that stopped escaping would go unnoticed.

    The ``colour`` field of the delta_T channels genuinely contains ``|``, so
    the rendered table must carry ``\|`` -- if it ever emits a bare pipe there,
    the row silently gains a column in every Markdown viewer.
    """
    md = render_markdown()
    assert "|" in CHANNEL_EQUATIONS["trn"].colour, (
        "test premise gone: trn.colour no longer contains a pipe."
    )
    assert r"\|" in md


def test_render_latex_is_a_booktabs_table() -> None:
    tex = render_latex()
    for token in (r"\begin{table}", r"\toprule", r"\midrule", r"\bottomrule",
                  r"\end{tabular}", r"\end{table}"):
        assert token in tex, f"{token} missing from the LaTeX table."
    body_rows = [ln for ln in tex.splitlines() if ln.strip().endswith(r"\\")]
    # header + one row per channel
    assert len(body_rows) == len(CHANNEL_EQUATIONS) + 1


def test_render_latex_escapes_cell_text() -> None:
    r"""Underscores and pipes must not reach the typesetter raw.

    A bare ``_`` is a maths-mode subscript (compile error in a text cell) and a
    bare ``|`` typesets as an em dash under OT1 -- both would corrupt the table
    silently or loudly. Channel names are full of underscores and the colour
    column is full of pipes, so this is not hypothetical.
    """
    tex = render_latex()
    body = tex.split(r"\midrule", 1)[1]

    assert "quantum\\_vacuum" in body
    assert r"\textbar{}" in body, "pipes in the colour column are not escaped."

    for line in body.splitlines():
        stripped = line.strip()
        if not stripped.endswith(r"\\"):
            continue
        # Every underscore in a rendered row must be backslash-escaped.
        for idx, char in enumerate(stripped):
            if char == "_":
                assert idx > 0 and stripped[idx - 1] == "\\", (
                    f"unescaped underscore in LaTeX row: {stripped}"
                )


def test_committed_doc_matches_render_markdown() -> None:
    """docs/EQUATION_MAP.md must be the current render, not a stale copy."""
    doc = REPO_ROOT / "docs" / "EQUATION_MAP.md"
    assert doc.is_file(), "docs/EQUATION_MAP.md is missing."
    assert doc.read_text(encoding="utf-8") == render_markdown(), (
        "docs/EQUATION_MAP.md is stale. Regenerate with:\n"
        "  python -m simulator.equation_map > docs/EQUATION_MAP.md"
    )
