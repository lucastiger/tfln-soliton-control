"""Provenance stamping for solver runs and generated artifacts.

Promoted from ``analysis/_provenance.py`` (which is now a deprecated shim) so
the *simulator* can stamp its own output rather than relying on each analysis
script to remember to do it. Everything the analysis helper provided is kept
byte-compatible:

* ``legacy`` mode reproduces the EXACT quantum-noise-report stamp
  ({script, git_commit (full), base_config_sha256 (whole-file), seed}), so
  ``analysis/results/quantum_noise_report.json``'s committed content is
  unchanged if that report is regenerated.
* the default (new) mode is {script, git_commit (short),
  physical_parameters_sha256 (resolved block), seed, quick, generated_utc}.

On top of that this module adds the pieces a *benchmark* artifact needs to be
auditable a year later: a content hash of the :class:`~simulator.noise_config.NoiseConfig`
that produced it, a fingerprint of the numerical environment (JAX/BLAS/device),
a version-pinned citation of the reference paper, and an atomic
artifact writer/reader pair that refuses to hand back an unstamped array file.

The git commit and hashes are computed from the repo the module lives in, so
the stamp is valid regardless of the caller's working directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from simulator.noise_config import NoiseConfig

__all__ = [
    "ARXIV_REF",
    "config_block_sha256",
    "env_fingerprint",
    "load_artifact",
    "noise_config_sha256",
    "provenance_stamp",
    "save_artifact",
]

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Reference this repository's noise physics is validated against. The
#: ``version`` pin is mandatory: equation and section numbers cited throughout
#: the solver (e.g. Sec. V.B.2, Eq. 126) are v1 numbers and may shift in a
#: later revision, so a stamp that records only the arXiv id is ambiguous.
ARXIV_REF: dict[str, Any] = {
    "id": "2604.05897",
    "version": "v1",
    "title": "Frequency combs and coherent dissipative structures in nonlinear optical microresonators",
    "authors": ["Herr", "Tikan", "Kippenberg"],
    "date": "2026-04-07",
    "accessed": "2026-08-04",
}


def _git_commit(short: bool) -> str:
    """HEAD commit hash (short = 12 chars) of the repo, or 'unknown'."""
    cmd = ["git", "-C", str(_REPO_ROOT), "rev-parse"]
    if short:
        cmd += ["--short=12"]
    cmd += ["HEAD"]
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=5.0
        ).stdout.strip()
    except Exception:
        return "unknown"


#: HEAD commit resolved ONCE at import. ``solve_lle_ssfm_jax`` is called in
#: tight batch loops (thousands of trajectories per campaign); shelling out to
#: git per call would add a process spawn to every solve and would turn a
#: broken/absent git into a per-call failure mode. Resolving here means the
#: subprocess runs at most once per process, and never inside a solve. The
#: 5 s timeout in :func:`_git_commit` bounds even this one call, so a wedged
#: git (NFS, credential prompt) cannot hang the import either.
_GIT_COMMIT_SHORT: str = _git_commit(short=True)

#: The FULL hash is needed only by ``legacy=True`` (the quantum-noise report),
#: so it is resolved on first use rather than at import: the solver's stamp
#: uses the short hash and should not pay for a second subprocess.
_GIT_COMMIT_FULL: str | None = None


def _git_commit_full_cached() -> str:
    global _GIT_COMMIT_FULL
    if _GIT_COMMIT_FULL is None:
        _GIT_COMMIT_FULL = _git_commit(short=False)
    return _GIT_COMMIT_FULL


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def config_block_sha256(physical_params: dict[str, Any]) -> str:
    """Hash a resolved ``physical_parameters`` block deterministically.

    Parameters
    ----------
    physical_params : dict
        The RESOLVED parameter block -- the values the run actually used, after
        defaults and overrides, not the raw YAML. Values may be any type
        ``json.dumps`` accepts, plus anything ``float()`` can convert.

    Returns
    -------
    str
        64 lowercase hex characters.

    Raises
    ------
    TypeError
        If a value is neither JSON-serialisable nor convertible by ``float``.

    Notes
    -----
    Canonicalized with ``json.dumps(..., sort_keys=True)``, so the digest is
    invariant to key ORDER and to float formatting round-tripping through
    Python's repr. Hashing the resolved values rather than the YAML bytes is
    deliberate: two configs that differ only in comments, key order or
    whitespace describe the same run and must hash the same, while a changed
    default that silently alters a value must not.

    Examples
    --------
    >>> a = config_block_sha256({"kappa_i_rad_per_s": 1.0, "T_k": 300.0})
    >>> b = config_block_sha256({"T_k": 300.0, "kappa_i_rad_per_s": 1.0})
    >>> len(a), a == b
    (64, True)
    >>> a == config_block_sha256({"T_k": 301.0, "kappa_i_rad_per_s": 1.0})
    False
    """
    canon = json.dumps(physical_params, sort_keys=True, default=float)
    return _sha256_bytes(canon.encode("utf-8"))


def noise_config_sha256(nc: "NoiseConfig") -> str:
    """Return the content hash of a :class:`~simulator.noise_config.NoiseConfig`.

    Parameters
    ----------
    nc : NoiseConfig
        The resolved noise configuration of a run.

    Returns
    -------
    str
        64 lowercase hex characters, identical to ``nc.sha256()``.

    Raises
    ------
    AttributeError
        If ``nc`` is not a :class:`~simulator.noise_config.NoiseConfig` (or
        anything else exposing ``sha256()``).

    Notes
    -----
    Delegates to :meth:`~simulator.noise_config.NoiseConfig.sha256` rather than
    re-canonicalizing here, so this digest cannot drift from the
    ``noise_config_sha256`` already recorded in the noise-off identity manifests
    in ``validation/noise_off_identity.py``. The digest changes iff a field
    VALUE changes and is invariant to field DECLARATION order.

    Examples
    --------
    >>> from simulator.noise_config import NoiseConfig
    >>> noise_config_sha256(NoiseConfig()) == NoiseConfig().sha256()
    True
    >>> noise_config_sha256(NoiseConfig()) == noise_config_sha256(
    ...     NoiseConfig(trn=True))
    False
    """
    return nc.sha256()


def _blas_name() -> str:
    """Best-effort BLAS/LAPACK vendor behind this numpy build.

    ``numpy.show_config`` prints to stdout on numpy < 2 and returns a dict when
    called with ``mode="dicts"`` on numpy >= 2; both spellings are tried and
    any failure degrades to ``"unknown"``.
    """
    try:
        cfg = np.show_config(mode="dicts")  # type: ignore[call-arg]
        deps = cfg.get("Build Dependencies", {})
        blas = deps.get("blas", {})
        name = blas.get("name")
        if name:
            version = blas.get("version")
            return f"{name} {version}" if version else str(name)
    except Exception:
        pass
    try:
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            np.show_config()
        text = buf.getvalue().lower()
        for candidate in ("openblas", "mkl", "accelerate", "blis", "atlas"):
            if candidate in text:
                return candidate
    except Exception:
        pass
    return "unknown"


def env_fingerprint() -> dict[str, Any]:
    """Fingerprint the numerical environment, for reproducing a run.

    Returns
    -------
    dict
        Eight string-valued keys: ``python``, ``platform``, ``numpy``, ``jax``,
        ``jaxlib``, ``backend``, ``device_kind`` and ``blas``. Any probe that
        cannot be answered reports ``"unknown"`` (or ``"none"`` when JAX
        reports no devices at all).

    Raises
    ------
    None
        By construction. Every probe is individually guarded, so a missing JAX,
        an unavailable backend, or a misbehaving ``numpy.show_config`` degrades
        that one field rather than raising. This function sits on the solver's
        RETURN path and must not be able to fail a run that has already computed
        its physics.

    Notes
    -----
    These eight fields are the ones that decide bit-level reproducibility. The
    BLAS vendor and the device kind matter as much as the library versions:
    XLA vectorizes and reassociates reductions according to the CPU it finds, so
    the same source on the same versions can produce a different -- equally
    valid -- rounding on a machine with a different SIMD width. A stamp that
    records only versions cannot explain such a difference.

    Examples
    --------
    >>> fp = env_fingerprint()
    >>> sorted(fp)
    ['backend', 'blas', 'device_kind', 'jax', 'jaxlib', 'numpy', 'platform', 'python']
    >>> all(isinstance(v, str) for v in fp.values())
    True
    """
    fp: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": getattr(np, "__version__", "unknown"),
        "jax": "unknown",
        "jaxlib": "unknown",
        "backend": "unknown",
        "device_kind": "unknown",
        "blas": "unknown",
    }
    try:
        import jax

        fp["jax"] = getattr(jax, "__version__", "unknown")
    except Exception:
        pass
    try:
        import jaxlib

        fp["jaxlib"] = getattr(jaxlib, "__version__", "unknown")
    except Exception:
        pass
    try:
        import jax

        fp["backend"] = str(jax.default_backend())
    except Exception:
        pass
    try:
        import jax

        devices = jax.devices()
        fp["device_kind"] = str(devices[0].device_kind) if devices else "none"
    except Exception:
        pass
    try:
        fp["blas"] = _blas_name()
    except Exception:
        pass
    return fp


def provenance_stamp(
    script: str,
    seed: int,
    *,
    physical_params: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    quick: bool | None = None,
    legacy: bool = False,
) -> dict[str, Any]:
    """Build a provenance dict for a generated report.

    Parameters
    ----------
    script : str
        Repo-relative path of the generating script.
    seed : int
        RNG seed used for the run (dimensionless).
    physical_params : dict or None, optional
        The resolved ``physical_parameters`` block, hashed with
        :func:`config_block_sha256` into ``physical_parameters_sha256``.
        Required in the default (new) mode. Keyword-only.
    config_path : str or pathlib.Path or None, optional
        Base YAML config. In ``legacy`` mode its WHOLE-FILE bytes are hashed
        into ``base_config_sha256``. Keyword-only.
    quick : bool or None, optional
        Value of the report's ``quick`` flag; omitted from the stamp when
        ``None``. New mode only. Keyword-only.
    legacy : bool, optional
        Reproduce the quantum-noise-report stamp exactly -- full commit hash,
        whole-file config hash, and no ``quick`` or timestamp fields. Default
        ``False``. Keyword-only.

    Returns
    -------
    dict
        Insertion-ordered and JSON-serialisable. New mode:
        ``script``, ``git_commit`` (12-char short hash),
        ``physical_parameters_sha256``, ``seed``, optionally ``quick``, then
        ``generated_utc`` (``%Y-%m-%dT%H:%M:%SZ``). Legacy mode: ``script``,
        ``git_commit`` (full hash), ``base_config_sha256``, ``seed``.

    Raises
    ------
    ValueError
        If ``legacy`` is set without ``config_path``, or if the default mode is
        used without ``physical_params``.
    OSError
        In legacy mode, if ``config_path`` cannot be read.

    Notes
    -----
    The legacy mode exists so that regenerating
    ``analysis/results/quantum_noise_report.json`` leaves its committed content
    byte-identical; it is not a deprecated path so much as a frozen one.

    ``generated_utc`` makes the new-mode stamp deliberately NOT reproducible,
    which is why the solver only attaches a stamp when explicitly asked
    (``provenance=True``): the golden whole-dict hashes assume the legacy key
    set.

    Examples
    --------
    >>> stamp = provenance_stamp("scripts/demo.py", seed=7,
    ...                          physical_params={"T_k": 300.0})
    >>> list(stamp)
    ['script', 'git_commit', 'physical_parameters_sha256', 'seed', 'generated_utc']
    >>> stamp["seed"], len(stamp["physical_parameters_sha256"])
    (7, 64)

    ``quick`` appears only when it was given:

    >>> list(provenance_stamp("scripts/demo.py", seed=7,
    ...                       physical_params={"T_k": 300.0}, quick=True))
    ['script', 'git_commit', 'physical_parameters_sha256', 'seed', 'quick', 'generated_utc']

    Neither mode will produce an under-specified stamp:

    >>> provenance_stamp("scripts/demo.py", seed=7)
    Traceback (most recent call last):
        ...
    ValueError: new-style provenance needs physical_params (resolved block hash).
    """
    if legacy:
        if config_path is None:
            raise ValueError("legacy provenance needs config_path (whole-file hash).")
        return {
            "script": script,
            "git_commit": _git_commit_full_cached(),
            "base_config_sha256": _sha256_bytes(Path(config_path).read_bytes()),
            "seed": int(seed),
        }

    if physical_params is None:
        raise ValueError(
            "new-style provenance needs physical_params (resolved block hash)."
        )
    stamp = {
        "script": script,
        "git_commit": _GIT_COMMIT_SHORT,
        "physical_parameters_sha256": config_block_sha256(physical_params),
        "seed": int(seed),
    }
    if quick is not None:
        stamp["quick"] = bool(quick)
    stamp["generated_utc"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return stamp


def _artifact_paths(path: str | Path) -> tuple[Path, Path]:
    """(npz path, sidecar path) for an artifact stem.

    A trailing ``.npz`` on ``path`` is accepted and stripped, so both
    ``save_artifact("run")`` and ``save_artifact("run.npz")`` produce
    ``run.npz`` + ``run.provenance.json``.

    The suffix is APPENDED rather than substituted via ``Path.with_suffix``:
    a dotted stem such as ``sweep_v1.2`` has ``.2`` as its suffix, and
    substituting would silently write ``sweep_v1.npz`` -- two runs colliding on
    one filename is precisely the provenance failure this module exists to
    prevent.
    """
    p = Path(path)
    if p.suffix == ".npz":
        p = p.with_suffix("")
    return (
        p.with_name(p.name + ".npz"),
        p.with_name(p.name + ".provenance.json"),
    )


def save_artifact(
    path: str | Path,
    arrays: dict[str, Any],
    meta: dict[str, Any],
    *,
    provenance: dict[str, Any],
) -> Path:
    """Write ``<path>.npz`` and ``<path>.provenance.json`` atomically.

    Parameters
    ----------
    path : str or pathlib.Path
        Artifact stem; a trailing ``.npz`` is accepted and stripped, so
        ``"run"`` and ``"run.npz"`` name the same artifact.
    arrays : dict
        Mapping of name to array, stored with ``numpy.savez``.
    meta : dict
        Run metadata, stored in the sidecar under ``"meta"``.
    provenance : dict
        The stamp from :func:`provenance_stamp`, stored in the sidecar under
        ``"provenance"``. Keyword-only.

    Returns
    -------
    pathlib.Path
        Path of the written ``.npz``.

    Raises
    ------
    OSError
        If the parent directory cannot be created, or either file cannot be
        written or moved into place.
    TypeError
        From ``json.dump`` only for a value neither serialisable nor
        ``str``-representable; the writer passes ``default=str``, so this is
        close to unreachable.

    Notes
    -----
    Both files are written to ``.tmp`` siblings and then moved into place with
    :func:`os.replace`, which is atomic within a filesystem. The ARRAYS land
    first and the SIDECAR second, so the only two interruption outcomes are:

    * interrupted before either move -- neither file appears;
    * interrupted between the moves -- arrays present, sidecar absent, and
      :func:`load_artifact` refuses to load them.

    An interrupted run can therefore never leave a half-written ``.npz`` paired
    with a valid-looking stamp, which is the failure that silently poisons a
    benchmark table.

    ``numpy.savez`` rather than ``savez_compressed`` is deliberate: DEFLATE
    output is not bit-stable across zlib builds, and these artifacts are
    compared byte-for-byte (cf. ``validation/noise_off_identity.py``).

    Examples
    --------
    >>> import numpy as np, pathlib, tempfile
    >>> stem = pathlib.Path(tempfile.mkdtemp()) / "run"
    >>> stamp = provenance_stamp("scripts/demo.py", seed=7,
    ...                          physical_params={"T_k": 300.0})
    >>> written = save_artifact(stem, {"U_int": np.arange(3.0)},
    ...                         {"note": "demo"}, provenance=stamp)
    >>> written.name
    'run.npz'
    >>> written.with_name("run.provenance.json").exists()
    True
    """
    npz_path, sidecar_path = _artifact_paths(path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    npz_tmp = npz_path.with_name(npz_path.name + ".tmp")
    sidecar_tmp = sidecar_path.with_name(sidecar_path.name + ".tmp")

    payload = {
        "provenance": dict(provenance),
        "meta": dict(meta),
        "arrays": sorted(arrays),
    }

    try:
        with npz_tmp.open("wb") as fh:
            np.savez(fh, **arrays)
        with sidecar_tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")
        os.replace(npz_tmp, npz_path)
        os.replace(sidecar_tmp, sidecar_path)
    finally:
        for tmp in (npz_tmp, sidecar_tmp):
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
    return npz_path


def load_artifact(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load ``<path>.npz`` together with its provenance sidecar.

    Parameters
    ----------
    path : str or pathlib.Path
        Artifact stem; a trailing ``.npz`` is accepted and stripped.

    Returns
    -------
    arrays : dict of str to numpy.ndarray
        Every array stored in the ``.npz``, keyed by name.
    provenance : dict
        The WHOLE sidecar record --
        ``{"provenance": <stamp>, "meta": ..., "arrays": [...]}`` -- not just
        the stamp, so the ``meta`` passed to :func:`save_artifact` is
        recoverable from the round trip.

    Raises
    ------
    FileNotFoundError
        If the ``.npz`` is missing, or -- deliberately -- if the sidecar is
        missing. An artifact without its stamp is not a usable benchmark input,
        and returning the arrays anyway would let an unattributable file flow
        into a results table.
    ValueError
        From ``numpy.load`` if the archive is corrupt, or if it contains a
        pickled object (loading is ``allow_pickle=False``).

    Notes
    -----
    The refusal to load an unstamped artifact is the whole point of the pair:
    :func:`save_artifact` guarantees the sidecar is never newer than the arrays,
    and this function guarantees the arrays are never read without it.

    Examples
    --------
    >>> import numpy as np, pathlib, tempfile
    >>> stem = pathlib.Path(tempfile.mkdtemp()) / "run"
    >>> stamp = provenance_stamp("scripts/demo.py", seed=7,
    ...                          physical_params={"T_k": 300.0})
    >>> _ = save_artifact(stem, {"U_int": np.arange(3.0)}, {"note": "demo"},
    ...                   provenance=stamp)
    >>> arrays, record = load_artifact(stem)
    >>> arrays["U_int"]
    array([0., 1., 2.])
    >>> sorted(record), record["meta"]
    (['arrays', 'meta', 'provenance'], {'note': 'demo'})

    A stem with no artifact behind it is an error, not an empty result:

    >>> load_artifact(pathlib.Path(tempfile.mkdtemp()) / "absent")
    Traceback (most recent call last):
        ...
    FileNotFoundError: artifact arrays missing: ...absent.npz
    """
    npz_path, sidecar_path = _artifact_paths(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"artifact arrays missing: {npz_path}")
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"provenance sidecar missing for {npz_path}: expected {sidecar_path}. "
            f"Refusing to load an unstamped artifact."
        )
    with np.load(npz_path, allow_pickle=False) as npz:
        arrays = {key: npz[key] for key in npz.files}
    with sidecar_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return arrays, payload
