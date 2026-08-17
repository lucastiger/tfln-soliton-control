# Reproducible environment for the stochastic-LLE benchmark.
#
#     docker build -t sc .
#     docker run --rm sc                      # fast suite
#     docker run --rm -e SOLITON_STRICT_ULP=1 sc \
#         pytest -q tests/test_noise_off_identity.py   # 0-ULP identity check
#
# python:3.11-slim matches the interpreter recorded in
# tests/data/golden/*.provenance.json (3.11.x), which is the environment the
# committed golden trajectories were produced in. Together with the exact jaxlib
# pin in requirements.lock.txt that makes the bit-identity check meaningful
# inside this image rather than merely approximate.
FROM python:3.11-slim

# PYTHONDONTWRITEBYTECODE keeps the layer clean; PYTHONUNBUFFERED makes the test
# output stream rather than arrive in a lump when the container exits.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# NOT setting JAX_PLATFORMS on purpose. The goldens were written with that
# variable unset (`"jax_platforms_env": null` in every provenance sidecar) and
# jax picks the CPU backend on its own in a container with no accelerator, so
# leaving it alone reproduces the reference environment exactly instead of
# merely arriving at the same answer by a different route.

WORKDIR /app

# Two system tools the suite genuinely needs; python:3.11-slim ships neither.
#
#   git   -- simulator/provenance.py:70 stamps every artifact by running
#            `git -C <repo> rev-parse`. Without it the stamp degrades to
#            "unknown" instead of raising, so the image would build, run, pass
#            most things, and quietly emit untraceable artifacts.
#            tests/test_provenance.py is what notices, by asserting a 40-char
#            hash. The .git directory itself is shipped for the same reason;
#            see .dockerignore.
#   patch -- third_party/pylle/verify_vendor.py:89 applies the vendored pyLLE
#            patches with it, checked by tests/test_pylle_vendor.py.
RUN apt-get update \
    && apt-get install --no-install-recommends -y git patch \
    && rm -rf /var/lib/apt/lists/*

# The build backend, pinned, installed BEFORE the locked set. `pip install -e .`
# below then runs with --no-build-isolation: default build isolation would
# silently fetch an unpinned setuptools from the network at build time, which is
# precisely the kind of floating dependency this image exists to eliminate.
RUN pip install --no-cache-dir "setuptools==84.0.0" "wheel==0.48.0"

# Locked dependencies first, in their own layer: they change rarely, the source
# tree changes constantly, and this ordering means editing a solver file does not
# re-resolve jax.
#
# --require-hashes is the point of the exercise. Every distribution in the lock
# carries its sha256, so a compromised or silently re-uploaded wheel fails the
# build instead of quietly changing a benchmark number.
COPY requirements.lock.txt ./
RUN pip install --no-cache-dir --require-hashes -r requirements.lock.txt

COPY . .

# --no-deps because requirements.lock.txt already resolved everything and is the
# authority; letting this step resolve would let an unpinned transitive
# dependency in through the back door.
RUN pip install --no-cache-dir --no-deps --no-build-isolation -e .

# Fail loudly at BUILD time if the image cannot import the solver, rather than
# at run time in someone else's CI.
RUN python -c "import simulator; import simulator.lle_solver as m; print('simulator OK:', m.__file__)"

# The fast suite: the same selection the `fast` CI job runs. `slow` is also
# gated behind --runslow by conftest.py, and `pylle_full` stays skipped because
# it needs a Julia toolchain that deliberately does not live in this image.
CMD ["pytest", "-q", "-m", "not slow and not gpu"]
