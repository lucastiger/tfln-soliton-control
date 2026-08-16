"""Integrity check for the vendored pyLLE Julia kernel.

Three things are asserted, and any of them failing is a hard stop:

1. **Upstream has not drifted.** The sha256 of the installed
   ``pyLLE/ComputeLLE.jl`` must equal the pinned hash recorded here. A mismatch
   means the pyLLE version changed under us and the patches were authored
   against a file that no longer exists; the vendored kernel would then be
   silently stale.
2. **The vendored file is exactly the patched upstream.** The three patches are
   applied in order to the pristine ``.orig`` and the result must be
   byte-identical to ``ComputeLLE.jl``. This is what makes the vendoring
   auditable: nobody has to trust that the vendored copy contains only the
   documented changes -- it is re-derived and compared.
3. **The pristine copy is really the upstream file.** ``.orig`` must itself hash
   to the pinned value.

``assert_vendor_integrity()`` is the library entry point; the cross-check calls
it before every run so a drifted or hand-edited kernel can never silently
produce numbers.

CLI: ``python third_party/pylle/verify_vendor.py [--diff]``.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDORED = HERE / "ComputeLLE.jl"
PRISTINE = HERE / "ComputeLLE.jl.orig"
PATCH_DIR = HERE / "patches"

# Pinned upstream. Authored against pyLLE 4.1.2 as published on PyPI.
UPSTREAM_PYLLE_VERSION = "4.1.2"
UPSTREAM_SHA256 = "fc84520cff40909cfae4e68c5cc6fb0812b947d4070a73f520c854dd426d12fe"
VENDORED_SHA256 = "f400979c29ec158ed26edc27030b93e8c40b0cb4a1b8712917b1ff5c2eb599cf"

PATCH_ORDER = (
    "0001-restore-dt-cli.patch",
    "0002-restore-tol-maxiter-cli.patch",
    "0003-probe-final-step.patch",
)


class VendorIntegrityError(RuntimeError):
    """Raised when the vendored kernel is not the documented patched upstream."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def installed_kernel_path() -> Path | None:
    """Locate the installed pyLLE ComputeLLE.jl, if a pyLLE env is importable.

    Returns None when pyLLE is not installed in *this* interpreter: the vendored
    file is self-describing via ``.orig``, so the integrity check is still
    meaningful without an installed pyLLE. Drift detection simply cannot run.
    """
    try:
        import pyLLE  # noqa: F401
    except Exception:
        return None
    return Path(pyLLE.__file__).resolve().parent / "ComputeLLE.jl"


def apply_patches(base_text: str) -> str:
    """Apply the three patches in order to ``base_text`` and return the result."""
    with tempfile.TemporaryDirectory() as td:
        work = Path(td) / "ComputeLLE.jl"
        work.write_text(base_text)
        for name in PATCH_ORDER:
            patch_file = PATCH_DIR / name
            if not patch_file.exists():
                raise VendorIntegrityError(f"missing patch {name}")
            proc = subprocess.run(
                ["patch", "--silent", "-p0", str(work)],
                input=patch_file.read_text(), capture_output=True, text=True)
            if proc.returncode != 0:
                raise VendorIntegrityError(
                    f"patch {name} did not apply cleanly:\n{proc.stdout}\n{proc.stderr}")
        return work.read_text()


def assert_vendor_integrity(check_installed: bool = True) -> dict:
    """Assert the vendored kernel is exactly upstream + the three patches.

    Raises :class:`VendorIntegrityError` on any mismatch. Returns a dict of the
    hashes actually observed, suitable for dropping into a provenance block.
    """
    if not VENDORED.exists() or not PRISTINE.exists():
        raise VendorIntegrityError("vendored kernel or pristine copy is missing")

    pristine_hash = sha256_file(PRISTINE)
    vendored_hash = sha256_file(VENDORED)

    if pristine_hash != UPSTREAM_SHA256:
        raise VendorIntegrityError(
            f"ComputeLLE.jl.orig does not match the pinned upstream hash\n"
            f"  expected {UPSTREAM_SHA256}\n  got      {pristine_hash}")

    installed_hash = None
    if check_installed:
        inst = installed_kernel_path()
        if inst is not None and inst.exists():
            installed_hash = sha256_file(inst)
            if installed_hash != UPSTREAM_SHA256:
                raise VendorIntegrityError(
                    "INSTALLED pyLLE ComputeLLE.jl has drifted from the pinned "
                    f"upstream. The patches were authored against {UPSTREAM_SHA256} "
                    f"but the installed file hashes to {installed_hash}. Re-audit the "
                    "patches against the new upstream before trusting any result.")

    rebuilt = apply_patches(PRISTINE.read_text())
    rebuilt_hash = sha256_text(rebuilt)
    if rebuilt_hash != vendored_hash:
        raise VendorIntegrityError(
            "applying the patches to ComputeLLE.jl.orig does NOT reproduce the "
            f"vendored ComputeLLE.jl\n  rebuilt  {rebuilt_hash}\n  vendored {vendored_hash}\n"
            "The vendored file contains changes that are not in the patches.")
    if vendored_hash != VENDORED_SHA256:
        raise VendorIntegrityError(
            f"vendored ComputeLLE.jl hash changed\n  expected {VENDORED_SHA256}\n"
            f"  got      {vendored_hash}")

    return {
        "pylle_version": UPSTREAM_PYLLE_VERSION,
        "upstream_sha256": pristine_hash,
        "installed_sha256": installed_hash,
        "vendored_sha256": vendored_hash,
        "patches": [{"name": n, "sha256": sha256_file(PATCH_DIR / n)} for n in PATCH_ORDER],
    }


def unified_diff() -> str:
    return "".join(difflib.unified_diff(
        PRISTINE.read_text().splitlines(keepends=True),
        VENDORED.read_text().splitlines(keepends=True),
        fromfile="upstream/ComputeLLE.jl", tofile="vendored/ComputeLLE.jl"))


def revert_to_upstream(dest: Path) -> Path:
    """Write a pristine upstream kernel to ``dest`` (the documented revert path)."""
    shutil.copyfile(PRISTINE, dest)
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--diff", action="store_true", help="print vendored-vs-upstream diff")
    ap.add_argument("--no-installed-check", action="store_true",
                    help="skip the installed-pyLLE drift check")
    a = ap.parse_args(argv)
    try:
        info = assert_vendor_integrity(check_installed=not a.no_installed_check)
    except VendorIntegrityError as exc:
        print(f"VENDOR INTEGRITY FAILURE\n{exc}", file=sys.stderr)
        return 1
    print("vendor integrity: OK")
    print(f"  pyLLE            {info['pylle_version']}")
    print(f"  upstream sha256  {info['upstream_sha256']}")
    print(f"  installed sha256 {info['installed_sha256'] or '(pyLLE not importable here)'}")
    print(f"  vendored sha256  {info['vendored_sha256']}")
    for p in info["patches"]:
        print(f"  patch            {p['name']}  {p['sha256'][:16]}")
    if a.diff:
        print("\n" + unified_diff())
    return 0


if __name__ == "__main__":
    sys.exit(main())
