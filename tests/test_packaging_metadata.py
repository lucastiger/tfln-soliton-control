"""Guards on the packaging metadata.

Packaging files rot quietly. A dependency added to pyproject.toml but not to
requirements.in produces an image that works on the machine that built it and
fails in CI; a regenerated lock silently drops the note explaining why jaxlib's
pin is load-bearing, and six months later somebody bumps it as routine
maintenance. None of that shows up in a physics test, so it is checked here.

Everything in this module is static file inspection -- no solver runs, no
network, no imports of the packaged code.
"""

from __future__ import annotations

import ast
import fnmatch
import json
import re
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - 3.10 runs this branch in CI's `fast` job
    tomllib = pytest.importorskip(
        "tomli", reason="TOML parsing on 3.10 needs the tomli backport"
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"
PYPROJECT = REPO_ROOT / "pyproject.toml"
REQUIREMENTS_IN = REPO_ROOT / "requirements.in"
REQUIREMENTS_TXT = REPO_ROOT / "requirements.txt"
LOCK = REPO_ROOT / "requirements.lock.txt"
CITATION = REPO_ROOT / "CITATION.cff"
CODEMETA = REPO_ROOT / "codemeta.json"
ZENODO = REPO_ROOT / ".zenodo.json"
DOCKERFILE = REPO_ROOT / "Dockerfile"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

# The keyword set the task specified for the archival metadata. Pinned here so
# a drive-by edit to one file does not desynchronise the three.
REQUIRED_KEYWORDS = {
    "Lugiato-Lefever",
    "Kerr soliton",
    "microcomb",
    "stochastic simulation",
    "JAX",
    "thermorefractive noise",
    "quantum noise",
    "verification",
}


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def yaml_safe_load(path: Path) -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _requirement_name(line: str) -> str | None:
    """Distribution name from a requirement line, or None for blanks/comments."""
    line = line.split("#")[0].strip()
    if not line or line.startswith("-"):
        return None
    return re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0].strip().lower()


def _names(path: Path) -> set[str]:
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        name = _requirement_name(line)
        if name:
            names.add(name)
    return names


def _locked_versions() -> dict[str, str]:
    """name -> pinned version, from the lock's `name==version \\` lines."""
    pattern = re.compile(r"^([A-Za-z0-9._-]+)==([^\s\\]+)")
    pins = {}
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


# ---------------------------------------------------------------------------
# Dependency sets agree with each other
# ---------------------------------------------------------------------------

def test_base_dependencies_are_the_expected_set() -> None:
    """The base install is the solver's needs and nothing heavier."""
    declared = {_requirement_name(d) for d in _pyproject()["project"]["dependencies"]}
    assert declared == {
        "jax", "jaxlib", "numpy", "scipy", "matplotlib", "pyyaml", "tqdm", "h5py",
    }


def test_torch_and_einops_are_not_base_dependencies() -> None:
    """The reason the `ml` extra exists at all.

    A benchmark user installs this to integrate an LLE. torch is ~1 GB and the
    PI-RNN is not on the path from a config file to a number, so it must stay
    behind an extra -- and the fast CI job and the Docker image must keep
    exercising the configuration that omits it.
    """
    project = _pyproject()["project"]
    base = {_requirement_name(d) for d in project["dependencies"]}
    assert not (base & {"torch", "einops"}), (
        "torch/einops are back in the base dependency set; they belong in the "
        "`ml` extra."
    )
    ml = {_requirement_name(d) for d in project["optional-dependencies"]["ml"]}
    assert ml == {"torch", "einops"}


def test_optional_extras_are_declared() -> None:
    extras = _pyproject()["project"]["optional-dependencies"]
    assert {"ml", "pylle", "dev"} <= extras.keys()
    dev = {_requirement_name(d) for d in extras["dev"]}
    assert {"pytest", "pytest-cov", "ruff", "jupyter", "nbconvert", "pip-tools"} <= dev
    assert {_requirement_name(d) for d in extras["pylle"]} == {"pylle"}


def test_requirements_txt_mirrors_the_base_dependencies() -> None:
    """The convenience file and the authority must not disagree."""
    declared = {_requirement_name(d) for d in _pyproject()["project"]["dependencies"]}
    assert _names(REQUIREMENTS_TXT) == declared


def test_requirements_in_covers_every_base_dependency() -> None:
    """A base dependency missing from the lock source is missing from the image."""
    declared = {_requirement_name(d) for d in _pyproject()["project"]["dependencies"]}
    missing = declared - _names(REQUIREMENTS_IN)
    assert not missing, (
        f"base dependencies absent from requirements.in: {sorted(missing)}. The "
        "Docker image and the `identity` CI job install from the lock compiled "
        "out of that file, so they would run without them."
    )


def test_lock_does_not_carry_the_ml_extra() -> None:
    """torch in the lock would put a ~1 GB wheel in the benchmark image."""
    assert not ({"torch", "einops"} & set(_locked_versions())), (
        "the ml extra leaked into requirements.lock.txt"
    )


# ---------------------------------------------------------------------------
# The lock file's load-bearing pin
# ---------------------------------------------------------------------------

def test_jaxlib_is_pinned_exactly() -> None:
    pins = _locked_versions()
    assert "jaxlib" in pins, "jaxlib is not pinned in requirements.lock.txt"
    assert re.fullmatch(r"\d+\.\d+\.\d+.*", pins["jaxlib"]), (
        f"jaxlib pin {pins['jaxlib']!r} is not an exact version"
    )


def test_every_locked_requirement_carries_hashes() -> None:
    """--require-hashes fails the install otherwise, but late and cryptically."""
    text = LOCK.read_text(encoding="utf-8")
    for name in _locked_versions():
        block = re.search(
            rf"^{re.escape(name)}==.*?(?=^\S|\Z)", text, re.MULTILINE | re.DOTALL
        )
        assert block and "--hash=sha256:" in block.group(0), (
            f"{name} has no sha256 hash in the lock; `pip install --require-hashes` "
            "will refuse the whole file."
        )


def test_lock_explains_why_jaxlib_is_pinned() -> None:
    """pip-compile does not preserve hand-written notes; this is the tripwire.

    The note is the only place a reader learns that bumping jaxlib invalidates
    the goldens rather than merely refreshing a dependency. If a regeneration
    drops it, this fails and CONTRIBUTING.md tells you to put it back.
    """
    text = LOCK.read_text(encoding="utf-8")
    assert "ULP-STABILITY DEPENDENCY" in text, (
        "the jaxlib note is missing from requirements.lock.txt -- most likely the "
        "lock was regenerated with pip-compile, which does not preserve it. "
        "Restore it (see CONTRIBUTING.md)."
    )
    assert "SOLITON_STRICT_ULP" in text


def test_locked_toolchain_matches_the_golden_provenance() -> None:
    """The pinned versions ARE the ones the goldens were produced with.

    This is the assertion that makes the `identity` CI job worth running. If the
    lock drifts off the sidecars, that job is comparing against a toolchain the
    goldens never saw, and a pass means nothing.
    """
    sidecars = sorted((REPO_ROOT / "tests" / "data" / "golden").glob("*.provenance.json"))
    assert sidecars, "no golden provenance sidecars found"

    pins = _locked_versions()
    for sidecar in sidecars:
        versions = json.loads(sidecar.read_text(encoding="utf-8")).get("versions", {})
        for dist in ("jax", "jaxlib", "numpy"):
            recorded = versions.get(dist)
            if recorded is None:
                continue
            assert pins.get(dist) == recorded, (
                f"{sidecar.name} records {dist}=={recorded} but the lock pins "
                f"{dist}=={pins.get(dist)}. Either regenerate the goldens on the "
                "locked toolchain or restore the pin -- do not leave them "
                "disagreeing, because SOLITON_STRICT_ULP=1 then tests nothing."
            )


# ---------------------------------------------------------------------------
# Archival metadata
# ---------------------------------------------------------------------------

def test_citation_cff_is_valid_yaml_with_the_required_fields() -> None:
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(CITATION.read_text(encoding="utf-8"))

    assert data["cff-version"] == "1.2.0"
    for field in ("title", "authors", "version", "date-released", "license",
                  "repository-code", "preferred-citation"):
        assert field in data, f"CITATION.cff is missing {field!r}"
    assert data["license"] == "MIT"
    assert data["authors"], "CITATION.cff lists no authors"


def test_zenodo_doi_placeholder_is_present_and_not_invented() -> None:
    """The DOI is minted at first release; a plausible-looking guess is worse than a TODO."""
    text = CITATION.read_text(encoding="utf-8")
    assert "TODO-after-first-release" in text
    doi_entries = [
        identifier for identifier in yaml_safe_load(CITATION).get("identifiers", [])
        if identifier.get("type") == "doi"
    ]
    assert doi_entries, "CITATION.cff has no DOI identifier entry"
    assert doi_entries[0]["value"] == "TODO-after-first-release"


def test_orcids_are_left_as_todo_rather_than_invented() -> None:
    """An invented ORCID resolves to a real, uninvolved human being.

    Checked against the PARSED document, not the raw text: the file carries an
    example iD inside an explanatory comment, and a comment cannot be mistaken
    for an author's identity by any consumer. What must not happen is a
    syntactically valid ORCID appearing as an actual `orcid:` field while the
    TODO markers are still standing -- that combination is what a pattern-filled
    placeholder looks like.
    """
    text = CITATION.read_text(encoding="utf-8")
    assert "TODO" in text, "CITATION.cff no longer flags the ORCID as outstanding"

    data = yaml_safe_load(CITATION)
    authors = list(data.get("authors", []))
    preferred = data.get("preferred-citation") or {}
    authors += list(preferred.get("authors", []))

    valid_orcid = re.compile(r"^https://orcid\.org/\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")
    for author in authors:
        orcid = author.get("orcid")
        if orcid is None:
            continue
        assert valid_orcid.match(str(orcid)), (
            f"author {author.get('family-names')!r} carries a malformed ORCID "
            f"{orcid!r}; either give the real iD or leave the field out."
        )
        pytest.fail(
            f"author {author.get('family-names')!r} has an ORCID field while "
            "CITATION.cff still contains TODO markers. If this iD is genuinely "
            "the author's, clear the TODOs; if it was filled in from the example "
            "in the comment, remove it -- that iD belongs to someone else."
        )


@pytest.mark.parametrize("path", [CODEMETA, ZENODO])
def test_metadata_json_parses_and_carries_the_keywords(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    keywords = set(data["keywords"])
    missing = REQUIRED_KEYWORDS - keywords
    assert not missing, f"{path.name} is missing keywords: {sorted(missing)}"


def test_versions_agree_across_metadata() -> None:
    """One version number, four files."""
    version = _pyproject()["project"]["version"]
    assert json.loads(CODEMETA.read_text(encoding="utf-8"))["version"] == version
    assert json.loads(ZENODO.read_text(encoding="utf-8"))["version"] == version
    assert str(yaml_safe_load(CITATION)["version"]) == version
    assert f"## [{version}]" in (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Container and CI
# ---------------------------------------------------------------------------

def test_dockerfile_installs_from_the_lock_with_hashes() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "python:3.11-slim" in text
    assert "--require-hashes -r requirements.lock.txt" in text, (
        "the image must install the hash-verified lock, not floating versions"
    )
    assert "-e ." in text, "the image must install the project editable"


def test_dockerfile_installs_the_system_tools_the_suite_shells_out_to() -> None:
    """python:3.11-slim has neither, and both failures are quiet ones.

    `git` missing does not raise: simulator/provenance.py:70 falls back to
    "unknown", so artifacts build fine and are simply untraceable forever after.
    `patch` missing surfaces as a bare FileNotFoundError from subprocess, which
    reads like a broken test rather than a missing package.
    """
    text = DOCKERFILE.read_text(encoding="utf-8")
    for tool in ("git", "patch"):
        assert re.search(rf"apt-get install[^\n]*\b{tool}\b", text), (
            f"the image does not install {tool!r}; see the comment above the "
            "apt-get layer in the Dockerfile for what breaks without it"
        )


def test_dockerignore_keeps_the_git_directory() -> None:
    """Provenance stamping needs real git history, not just the git binary."""
    assert not _dockerignore_excludes(".git", _dockerignore_patterns()), (
        ".dockerignore excludes .git, so simulator/provenance.py:70 stamps every "
        "artifact produced in the image with git_commit='unknown'"
    )


def _dockerignore_patterns() -> list[str]:
    return [
        stripped
        for line in (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if (stripped := line.split("#")[0].strip().rstrip("/"))
    ]


def _repo_relative_paths_read_by_tests() -> dict[str, str]:
    """Repo-relative directories the test suite opens, derived from the AST.

    Returns {"analysis/results": "tests/test_provenance.py", ...}.

    Derived rather than listed, and derived from the syntax tree rather than by
    grepping for path strings, because grepping is what fails here: not one test
    contains the substring "analysis/results" -- they all build it as
    ``REPO_ROOT / "analysis" / "results"``. A text search says the directory is
    unused, excluding it from the image looks safe, and the suite then fails
    inside the container with six errors that point nowhere near the cause.
    """
    root_names = {"REPO_ROOT", "REPO", "ROOT", "PROJECT_ROOT"}
    found: dict[str, str] = {}

    def leftmost_is_repo_root(node: ast.AST) -> bool:
        while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            node = node.left
        return isinstance(node, ast.Name) and node.id in root_names

    def segments(node: ast.AST) -> list[str] | None:
        """String segments of a ``REPO_ROOT / "a" / "b"`` chain, in order."""
        parts: list[str] = []
        while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if not isinstance(node.right, ast.Constant) or not isinstance(
                node.right.value, str
            ):
                return None  # an f-string or variable component; stop guessing
            parts.append(node.right.value)
            node = node.left
        return list(reversed(parts))

    for module in sorted(TESTS_DIR.glob("test_*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
                continue
            if not leftmost_is_repo_root(node):
                continue
            parts = segments(node)
            if not parts:
                continue
            # Record the directory prefix, not the file: excluding a parent is
            # what actually happens in a .dockerignore.
            prefix = "/".join(parts[:2] if len(parts) > 1 else parts)
            found.setdefault(prefix, module.relative_to(REPO_ROOT).as_posix())
    return found


def _dockerignore_excludes(path: str, patterns: list[str]) -> bool:
    """Approximate Docker's matching: a matched ancestor excludes the subtree."""
    parts = path.split("/")
    for pattern in patterns:
        bare = pattern.replace("**/", "")
        for i in range(1, len(parts) + 1):
            if fnmatch.fnmatch("/".join(parts[:i]), pattern):
                return True
            if fnmatch.fnmatch(parts[i - 1], bare):
                return True
    return False


def test_dockerignore_keeps_what_the_fast_suite_reads() -> None:
    """The exclusions that would turn green into meaningless green.

    Two ways an exclusion hurts. It can break the suite outright -- that is what
    dropping analysis/results did, six failures inside the image. Worse, it can
    leave the suite green while silently disabling a check: validation/results/
    is hash-pinned by tests/test_validation_freeze.py, so excluding it turns the
    freeze guard into a no-op that still reports success.

    So the protected set is computed from what the tests actually reference,
    rather than maintained by hand next to a file that is edited for size.
    """
    patterns = _dockerignore_patterns()

    offenders = {
        path: module
        for path, module in _repo_relative_paths_read_by_tests().items()
        if (REPO_ROOT / path).exists() and _dockerignore_excludes(path, patterns)
    }
    assert not offenders, (
        ".dockerignore excludes paths the test suite reads, so `docker run --rm sc` "
        f"would fail on them: {offenders}. Either keep the path in the image or "
        "confirm the test no longer needs it."
    )

    # Belt and braces for the two that matter most, in case the derivation above
    # ever stops seeing them.
    for required in ("validation/results", "analysis/results", "tests/data",
                     "tests/data/golden", "config", "simulator"):
        assert not _dockerignore_excludes(required, patterns), (
            f".dockerignore excludes {required!r}, which the fast suite reads"
        )


def test_ci_workflow_has_the_four_jobs() -> None:
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert set(workflow["jobs"]) == {"lint", "fast", "identity", "slow"}

    fast = workflow["jobs"]["fast"]
    versions = fast["strategy"]["matrix"]["python-version"]
    assert set(versions) == {"3.10", "3.11", "3.12"}

    # The identity job must NOT be a matrix: bit-identity is version-specific, so
    # running it across interpreters manufactures failures that look like physics.
    identity = workflow["jobs"]["identity"]
    assert "strategy" not in identity, (
        "the identity job must run on ONE pinned interpreter"
    )
    identity_text = yaml.dump(identity)
    assert "SOLITON_STRICT_ULP" in identity_text
    assert "--require-hashes" in identity_text

    # The weekly slow job is the only scheduled one.
    assert "schedule" in workflow[True] or "schedule" in workflow.get("on", {}), (
        "no schedule trigger declared"
    )
    assert "--runslow" in yaml.dump(workflow["jobs"]["slow"])


def _conftest_module():
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import conftest
    finally:
        sys.path.remove(str(REPO_ROOT))
    return conftest


def test_every_hardware_locked_id_still_names_a_real_test() -> None:
    """A rename must not silently empty the skip list.

    ``--skip-hardware-locked`` matches node IDs by prefix. A stale entry matches
    nothing and fails silently -- the CI job would go on passing while the test
    it was meant to exclude either ran (and failed) or, worse, had been renamed
    and was quietly excluded by a different stale prefix. So each entry is
    checked against the source: the file exists, and it defines that function.
    """
    for node_id in _conftest_module().HARDWARE_LOCKED_NODE_IDS:
        rel_path, _, func = node_id.partition("::")
        path = REPO_ROOT / rel_path
        assert path.is_file(), f"{node_id} names a file that does not exist: {rel_path}"

        tree = ast.parse(path.read_text(encoding="utf-8"))
        defined = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert func in defined, (
            f"{node_id} names a test that {rel_path} no longer defines. Update "
            "conftest.HARDWARE_LOCKED_NODE_IDS -- a stale prefix silently "
            "excludes nothing, and the `fast` CI job would keep reporting green."
        )


def test_hardware_locked_skipping_is_off_by_default() -> None:
    """The reference machine must still run these; only CI opts out.

    This is the constraint the whole mechanism hangs on. If the default ever
    flips, byte-identity stops being checked anywhere at all and the repo's
    central claim quietly becomes unverified.
    """
    workflow_text = WORKFLOW.read_text(encoding="utf-8")
    assert "--skip-hardware-locked" in workflow_text, (
        "the fast job no longer skips the hardware-locked comparisons; it will "
        "go red on any runner whose CPU differs from the goldens' machine"
    )

    conftest_text = (REPO_ROOT / "conftest.py").read_text(encoding="utf-8")
    option_block = conftest_text.split('"--skip-hardware-locked"')[1][:400]
    assert "default=False" in option_block, (
        "--skip-hardware-locked must default to False so a plain `pytest` run on "
        "the reference machine still asserts byte-identity"
    )


def test_identity_job_does_not_claim_blocking_zero_ulp() -> None:
    """0 ULP is informational here, and the workflow must say why.

    Measured on a GitHub runner with the goldens' exact toolchain
    (version_mismatch=None): max_abs_diff 6.2e-19, max_rel_diff 1.8e-13. Same
    versions, different CPU, different bytes. If someone makes the strict step
    blocking again without pinning the hardware, the job reverts to failing
    permanently for a rounding difference.
    """
    yaml = pytest.importorskip("yaml")
    identity = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["identity"]

    strict_steps = [
        step for step in identity["steps"]
        if str(step.get("env", {}).get("SOLITON_STRICT_ULP", "")) == "1"
    ]
    assert strict_steps, "the strict 0-ULP step disappeared entirely"
    for step in strict_steps:
        assert step.get("continue-on-error") is True, (
            "the SOLITON_STRICT_ULP=1 step is blocking again. It cannot pass on a "
            "shared runner -- see the comment block on the identity job. Give the "
            "job fixed hardware before making this blocking."
        )

    # ...and something unconditional must still guard the noise-off path.
    blocking = [
        step for step in identity["steps"]
        if "test_noise_off_identity" in str(step.get("run", ""))
        and not step.get("continue-on-error")
    ]
    assert blocking, (
        "no blocking noise-off check remains; the pinned-toolchain job would "
        "assert nothing"
    )


def test_readme_carries_the_ci_badge() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "workflows/ci.yml/badge.svg" in readme
