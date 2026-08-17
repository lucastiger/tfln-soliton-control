"""Keeps the root conftest's optional-extra ignore list honest.

``torch`` and ``einops`` live in the ``ml`` extra rather than the base
dependency set, so a base-only environment -- the Docker image, the ``fast`` CI
job, anyone who ran ``pip install soliton-control`` to reproduce a benchmark
number -- does not have them. Two test modules import torch at module scope, and
a module-scope import of a missing package is a COLLECTION ERROR: it aborts the
entire run, not just that file. ``conftest._OPTIONAL_MODULE_REQUIREMENTS`` names
those modules so they can be dropped from collection when, and only when, the
extra is genuinely absent.

A hand-maintained list of that kind rots silently and in the dangerous
direction: someone adds a new torch-importing test, does not know the mapping
exists, and the base-only CI job goes red for a reason that looks nothing like
the cause. So this module re-derives the answer from the source and compares.

The scan is static (``ast``) and MODULE-SCOPE ONLY, which is exactly the
property that matters -- a lazy import inside a function body, or a
``pytest.importorskip`` guard like the one in test_mms_convergence.py, degrades
to a skip rather than an error and needs no entry here.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"

# Packages that a base-only install does not provide. Mirrors the
# optional-dependencies table in pyproject.toml.
OPTIONAL_PACKAGES = frozenset({"torch", "einops", "pyLLE"})

# First-party packages a test module may import, and through which a torch
# dependency can arrive transitively (e.g. `from data.dataloader import ...`).
FIRST_PARTY = frozenset(
    {
        "analysis",
        "benchmarks",
        "control",
        "data",
        "model",
        "scripts",
        "simulator",
        "third_party",
        "validation",
    }
)


def _module_scope_imports(path: Path) -> set[str]:
    """Top-level import targets of ``path``, as dotted names.

    Only ``tree.body`` is walked, never function or class bodies: an import
    nested inside a function is evaluated at call time, where a ModuleNotFound
    surfaces as a normal test failure or a skip, not as a collection error.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    targets: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            targets.add(node.module)
    return targets


def _source_for(dotted: str) -> Path | None:
    """Repo file backing a first-party dotted name, module or package."""
    as_module = REPO_ROOT / (dotted.replace(".", "/") + ".py")
    if as_module.is_file():
        return as_module
    as_package = REPO_ROOT / dotted.replace(".", "/") / "__init__.py"
    if as_package.is_file():
        return as_package
    return None


def _required_optional_packages(path: Path, _seen: set[Path] | None = None) -> set[str]:
    """Optional packages ``path`` pulls in at import time, following first-party edges."""
    seen = _seen if _seen is not None else set()
    resolved = path.resolve()
    if resolved in seen:
        return set()
    seen.add(resolved)

    required: set[str] = set()
    for dotted in _module_scope_imports(path):
        root = dotted.split(".")[0]
        if root in OPTIONAL_PACKAGES:
            required.add(root)
        elif root in FIRST_PARTY:
            source = _source_for(dotted)
            if source is not None:
                required |= _required_optional_packages(source, seen)
    return required


def _conftest_mapping() -> dict[str, str]:
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import conftest  # the repo-root one; pytest has already imported it
    finally:
        sys.path.remove(str(REPO_ROOT))
    return dict(conftest._OPTIONAL_MODULE_REQUIREMENTS)


def test_ignore_mapping_matches_the_source() -> None:
    """Every test module that hard-imports an optional package is declared, and vice versa."""
    derived = {}
    for module in sorted(TESTS_DIR.glob("test_*.py")):
        required = _required_optional_packages(module)
        if required:
            rel = module.relative_to(REPO_ROOT).as_posix()
            # One entry per module; if a module ever needs two extras this
            # assertion is the right place to find out.
            assert len(required) == 1, (
                f"{rel} hard-imports several optional packages {sorted(required)}; "
                "conftest._OPTIONAL_MODULE_REQUIREMENTS stores one per module and "
                "needs widening before this can be expressed."
            )
            derived[rel] = next(iter(required))

    declared = _conftest_mapping()

    undeclared = {k: v for k, v in derived.items() if k not in declared}
    assert not undeclared, (
        "these test modules import an optional package at module scope but are "
        f"not in conftest._OPTIONAL_MODULE_REQUIREMENTS: {undeclared}. Without an "
        "entry they turn a base-only run (the Docker image, the `fast` CI job) "
        "into a collection ERROR that aborts the whole suite. Add them."
    )

    stale = {k: v for k, v in declared.items() if k not in derived}
    assert not stale, (
        "conftest._OPTIONAL_MODULE_REQUIREMENTS lists modules that no longer "
        f"import their optional package at module scope: {stale}. Remove the "
        "entries so the tests are collected again."
    )

    mismatched = {
        k: (declared[k], derived[k]) for k in declared.keys() & derived.keys()
        if declared[k] != derived[k]
    }
    assert not mismatched, (
        f"declared vs actual optional package disagree (declared, actual): {mismatched}"
    )


def test_declared_paths_exist() -> None:
    """A typo in the mapping would silently protect nothing."""
    for rel in _conftest_mapping():
        assert (REPO_ROOT / rel).is_file(), (
            f"conftest._OPTIONAL_MODULE_REQUIREMENTS names {rel!r}, which does not "
            "exist. A path that matches no file ignores no file."
        )


def test_nothing_is_ignored_when_the_extra_is_installed() -> None:
    """The ignore list is driven by availability, not by a hard-coded opinion.

    With the ``ml`` extra present, collection must be byte-for-byte what it was
    before the packaging work -- that is the constraint the whole mechanism is
    built around.
    """
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import conftest
    finally:
        sys.path.remove(str(REPO_ROOT))

    for rel, requirement in conftest._OPTIONAL_MODULE_REQUIREMENTS.items():
        installed = importlib.util.find_spec(requirement) is not None
        assert (rel in conftest.collect_ignore) == (not installed), (
            f"{rel} ignored={rel in conftest.collect_ignore} while "
            f"{requirement} installed={installed}; the ignore list must track "
            "availability exactly."
        )


def test_toolchain_banner_is_visible_under_dash_q() -> None:
    """The banner must survive ``-q``, because ``-q`` is what CI runs.

    This is not a theoretical concern: the obvious implementations both fail
    here. ``pytest_report_header`` is not called at all under ``-q``, and a
    print inside a session fixture goes through capture. Only writing to the
    terminal reporter works, and nothing but this test would notice a
    regression back to either.

    Runs pytest in a subprocess -- the banner is a property of a session, so it
    cannot be observed from inside one. The target is a single cheap test from
    THIS module rather than a synthetic file in a temp directory: the root
    conftest is only loaded for paths beneath it, so a probe outside the repo
    would prove nothing (and quietly pass a broken implementation).
    """
    import subprocess

    target = f"{Path(__file__).relative_to(REPO_ROOT).as_posix()}::test_declared_paths_exist"
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", target,
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"probe run failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    assert "toolchain:" in result.stdout, (
        "the toolchain banner did not appear under -q.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    # The versions that decide bit-identity must all be named.
    for dist in ("python=", "jax=", "jaxlib=", "numpy="):
        assert dist in result.stdout, f"banner omits {dist!r}"


def test_toolchain_versions_fixture_reports_the_real_versions(
    toolchain_versions: dict[str, str],
) -> None:
    """The fixture and the installed distributions agree."""
    import numpy

    assert toolchain_versions["numpy"] == numpy.__version__
    assert toolchain_versions["python"] == ".".join(map(str, sys.version_info[:3]))


def test_base_only_suite_has_no_torch_collection_error() -> None:
    """The invariant in one line, from the other direction.

    If torch is absent, no COLLECTED test module may import it at module scope.
    Skipped when torch is installed, because then there is nothing to prove.
    """
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed; the base-only path is not under test here")

    declared = set(_conftest_mapping())
    for module in sorted(TESTS_DIR.glob("test_*.py")):
        rel = module.relative_to(REPO_ROOT).as_posix()
        if rel in declared:
            continue
        assert "torch" not in _required_optional_packages(module), (
            f"{rel} is collected in a torch-free environment but imports torch "
            "at module scope"
        )
