"""Guards that make the validation freeze real rather than aspirational.

A freeze recorded only in prose is a freeze that erodes. These tests are what
stop that:

* every artifact under ``validation/results/`` is hash-pinned against
  ``FROZEN_MANIFEST.md``, so a regeneration cannot happen quietly;
* the expected artifact set is pinned, so a deletion is caught;
* the test-suite split is checked in BOTH directions -- the cheap invariants are
  still in the default run, and nothing moved behind ``pylle_full`` vanished;
* ``docs/VALIDATION_STATUS.md`` must actually carry an impact line for every
  open item and an evidence pointer for every stopping-rule item, so the status
  document cannot decay into a list of headings.

None of this needs Julia, pyLLE or a solver run.
"""

from __future__ import annotations

import functools
import hashlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "validation" / "results"
MANIFEST = RESULTS / "FROZEN_MANIFEST.md"
STATUS = REPO_ROOT / "docs" / "VALIDATION_STATUS.md"
V2_DOC = REPO_ROOT / "docs" / "PYLLE_STATUS_V2.md"

# The manifest describes the artifacts; it is not itself an artifact of a run.
MANIFEST_EXEMPT = {"FROZEN_MANIFEST.md"}

EXPECTED_ARTIFACTS = {
    "convergence_lle_dw30k.json",
    "convergence_lle_dw30k_fields.npz",
    "existence_convergence_ours.json",
    "existence_convergence_ours_fields.npz",
    "pylle_crosscheck.json",
    "pylle_crosscheck.png",
    "pylle_crosscheck_fields.npz",
    "pylle_crosscheck_v2.json",
    "pylle_crosscheck_v2.png",
    "pylle_crosscheck_v2_fields.npz",
    "pylle_refinement_dw30k.json",
    "pylle_refinement_dw30k_fields.npz",
    "v1/MANIFEST.md",
    "v1/pylle_crosscheck.json",
    "v1/pylle_crosscheck.png",
    "v1/pylle_crosscheck_fields.npz",
}

# R3: these must stay in the DEFAULT suite. Each guards an invariant of the
# code, not the outcome of a run, and each is cheap.
CHEAP_CHECKS_REQUIRED_IN_DEFAULT = [
    "test_v1_artifacts_untouched",                       # frozen sha256, v1
    "test_v1_frozen_copy_matches_original",              # frozen sha256, v1 copies
    "test_convention_functions_are_the_v1_objects",      # v2 uses v1's objects
    "test_convention_functions_bit_identical_to_v1",
    "test_dispersion_mirror_is_applied_to_the_input",    # mirror on the INPUT
    "test_detuning_endpoint_gate_catches_probe_offset",  # H5 synthetic failure
    "test_criteria_fingerprint_unchanged_when_u_disc_is_none",  # fingerprint
    "test_committed_report_validates_against_the_v2_schema",    # schema
    "test_committed_report_records_argv_and_defaults_flag",     # defaults repro
    "test_committed_report_hard_checks_all_pass",               # 7/7 HARD
]

MARKED_TESTS = [
    "test_committed_report_has_a_containment_verdict_for_every_gated_observable",
    "test_committed_report_ladders_have_at_least_three_levels",
    "test_committed_report_records_picard_iterations_for_every_pylle_level",
]

# R2(d): every open item that must appear in the status document.
OPEN_ITEMS = [
    "Existence edges",
    "SEPARATED",
    "UNDERIVED",
    "grid-truncated",
    "non-stationary",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_rows() -> dict[str, str]:
    """{artifact path relative to results/: sha256} parsed from the manifest."""
    rows = {}
    for line in MANIFEST.read_text().splitlines():
        m = re.match(r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|", line)
        if m:
            rows[m.group(1)] = m.group(2)
    return rows


@functools.lru_cache(maxsize=8)
def _collect(*args: str) -> str:
    """Collect-only output, memoized.

    Each call spawns a pytest that imports the whole suite (jax included), which
    costs several seconds. The parametrized split checks below ask the same two
    questions a dozen times each, so without the cache this file alone would add
    ~100 s to every default run -- which is exactly the kind of friction the
    marker split exists to remove.
    """
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *args],
        cwd=REPO_ROOT, capture_output=True, text=True)
    return out.stdout + out.stderr


# ==========================================================================
# A1 / F1 -- the freeze
# ==========================================================================
def test_frozen_manifest_exists_and_parses():
    assert MANIFEST.exists(), "validation/results/FROZEN_MANIFEST.md is missing"
    rows = _manifest_rows()
    assert rows, "no artifact rows parsed out of FROZEN_MANIFEST.md"


def test_frozen_manifest_hashes_match():
    """The guard that makes the freeze real."""
    rows = _manifest_rows()
    mismatched = []
    for rel, sha in rows.items():
        path = RESULTS / rel
        assert path.exists(), f"{rel} is in the manifest but missing from disk"
        actual = _sha(path)
        if actual != sha:
            mismatched.append(f"{rel}: manifest {sha[:12]}… != actual {actual[:12]}…")
    assert not mismatched, (
        "FROZEN artifacts changed:\n  " + "\n  ".join(mismatched) +
        "\nThese are evidence of runs that happened. If a re-run trigger fired "
        "(docs/VALIDATION_STATUS.md section 5), update the manifest in the same "
        "commit and say which trigger it was.")


def test_manifest_covers_every_artifact_on_disk():
    """A new artifact must be added to the manifest, not left unpinned."""
    on_disk = set()
    for p in RESULTS.rglob("*"):
        if p.is_file():
            rel = p.relative_to(RESULTS).as_posix()
            if rel not in MANIFEST_EXEMPT:
                on_disk.add(rel)
    listed = set(_manifest_rows())
    assert on_disk == listed, (
        f"manifest and disk disagree; unlisted: {sorted(on_disk - listed)}, "
        f"listed but absent: {sorted(listed - on_disk)}")


def test_no_result_artifact_deleted():
    on_disk = {p.relative_to(RESULTS).as_posix()
               for p in RESULTS.rglob("*") if p.is_file()}
    missing = EXPECTED_ARTIFACTS - on_disk
    assert not missing, f"validation artifacts were deleted: {sorted(missing)}"


def test_every_artifact_is_marked_frozen():
    text = MANIFEST.read_text()
    for rel in _manifest_rows():
        row = next(l for l in text.splitlines() if f"`{rel}`" in l)
        assert row.rstrip().endswith("FROZEN |"), f"{rel} is not marked FROZEN"


# ==========================================================================
# A3 / A4 / R3 -- the test-suite split, checked in both directions
# ==========================================================================
@pytest.mark.parametrize("name", CHEAP_CHECKS_REQUIRED_IN_DEFAULT)
def test_cheap_checks_still_in_default_suite(name):
    """The named invariants must run WITHOUT -m pylle_full."""
    out = _collect("tests/")
    assert name in out, (
        f"{name} is no longer collected by the default suite. The split must "
        f"not move a cheap invariant behind the marker.")


@pytest.mark.parametrize("name", MARKED_TESTS)
def test_marked_tests_still_exist(name):
    """Nothing moved behind the marker may have been deleted."""
    out = _collect("tests/", "-m", "pylle_full")
    assert name in out, (
        f"{name} was marked pylle_full but is no longer collected with the "
        f"marker enabled -- the test was deleted rather than deferred.")


def test_marked_tests_are_skipped_not_run_by_default():
    out = _collect("tests/")
    # They are still COLLECTED by default (so nothing hides), just skipped.
    for name in MARKED_TESTS:
        assert name in out, f"{name} vanished from default collection entirely"


def test_marker_is_registered():
    conftest = (REPO_ROOT / "conftest.py").read_text()
    assert "pylle_full" in conftest
    out = subprocess.run([sys.executable, "-m", "pytest", "--markers"],
                         cwd=REPO_ROOT, capture_output=True, text=True).stdout
    assert "pylle_full" in out, "the marker is not registered; pytest will warn"


# ==========================================================================
# A2 / R2 / R4 -- the status document must carry real content
# ==========================================================================
def test_validation_status_exists_with_required_sections():
    assert STATUS.exists()
    text = STATUS.read_text()
    for heading in ("What is validated", "Configuration coverage",
                    "Stopping rule", "Known open items", "re-run triggers",
                    "Next validation priority"):
        assert heading.lower() in text.lower(), f"missing section: {heading}"


def test_validation_status_has_the_configuration_coverage_table():
    """The table is the most important content; check it names the real values."""
    text = STATUS.read_text()
    from analysis.dks_access import N_TAU, OPERATING_DW_KAPPA, PRODUCTION_NUMERICS

    assert str(N_TAU) in text, "the production n_tau is not in the coverage table"
    assert f"{OPERATING_DW_KAPPA:g}" in text
    assert str(PRODUCTION_NUMERICS["n_substeps"]) in text
    assert "6601" in text, "the validated n_tau is not in the coverage table"
    assert "30" in text
    # and it must say plainly that they differ
    assert "not the production configuration" in text


def test_validation_status_lists_all_open_items():
    """Each open item must appear WITH a non-empty impact line."""
    text = STATUS.read_text()
    for item in OPEN_ITEMS:
        assert item.lower() in text.lower(), f"open item not documented: {item}"
    impacts = re.findall(r"\*\*Impact on current claims:\s*([^*]+)\*\*", text)
    assert len(impacts) >= len(OPEN_ITEMS), (
        f"found {len(impacts)} impact lines for {len(OPEN_ITEMS)} open items")
    for i in impacts:
        assert i.strip(" :"), "an impact line is empty"


def test_stopping_rule_items_have_evidence_pointers():
    """All eight stopping-rule rows must point at evidence, not just assert."""
    text = STATUS.read_text()
    rows = [l for l in text.splitlines()
            if re.match(r"^\|\s*[1-8]\s*\|", l)]
    assert len(rows) == 8, f"expected 8 stopping-rule rows, found {len(rows)}"
    for row in rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert len(cells) >= 4, f"malformed stopping-rule row: {row}"
        assert cells[2], f"stopping-rule item {cells[0]} has no met/not-met verdict"
        evidence = cells[3]
        assert evidence, f"stopping-rule item {cells[0]} has no evidence pointer"
        assert ("." in evidence or "`" in evidence), (
            f"stopping-rule item {cells[0]} evidence is not a pointer: {evidence}")


def test_validation_status_does_not_claim_a_pass():
    """F3 / PROHIBITED: the freeze must not be presented as a PASS."""
    text = STATUS.read_text()
    assert "FAIL (QUALIFIED)" in text or "FAIL" in text
    assert "Freezing is not a pass" in text or "freeze is a judgement" in text


def test_rerun_triggers_named():
    text = STATUS.read_text()
    for trigger in ("_fine_step", "load_dint_grid", "dispersion sign",
                    "mode-index origin", "new device"):
        assert trigger in text, f"re-run trigger not named: {trigger}"


def test_solver_carries_trigger_comments():
    """R4: the trigger must be visible where the code would be edited."""
    src = (REPO_ROOT / "simulator" / "lle_solver.py").read_text()
    assert src.count("VALIDATION RE-RUN TRIGGER") >= 2, (
        "simulator/lle_solver.py must point at docs/VALIDATION_STATUS.md from "
        "both _fine_step and load_dint_grid")
    assert "docs/VALIDATION_STATUS.md" in src


# ==========================================================================
# A5 / A6 -- the freeze changed no result
# ==========================================================================
def test_v2_doc_has_section_10_and_states_the_verdict_unchanged():
    text = V2_DOC.read_text()
    assert "## 10. Status: FROZEN" in text
    tail = text.split("## 10. Status: FROZEN", 1)[1]
    assert "FAIL" in tail
    assert "does not change it" in tail
    assert "docs/VALIDATION_STATUS.md" in tail


def test_v2_verdict_is_still_fail_qualified():
    """F3: this task must not improve any result."""
    import json
    payload = json.loads((RESULTS / "pylle_crosscheck_v2.json").read_text())
    assert payload["overall"] == "FAIL"
    assert payload["overall_qualified"] is True
    g7 = [c for c in payload["checks"] if c["cid"] == "G7"]
    assert all(c["verdict"] == "FAIL" for c in g7), (
        "the frozen G7 verdicts changed; the freeze must not alter a result")
