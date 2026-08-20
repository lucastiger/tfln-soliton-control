"""Regression tests for config/sin_params.yaml parsing.

PyYAML's default (1.1) resolver only treats an exponential float as a float
when its exponent carries an explicit sign (e.g. ``2.0e+11``). An unsigned
exponent such as ``2.0e11`` is loaded as the *string* ``'2.0e11'`` instead.
These tests lock in the fix that signs every exponent in the config file so
all physical parameters parse natively as numbers.
"""
import math
from pathlib import Path

import pytest
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "sin_params.yaml"

# Documented exceptions to the numeric-leaves rule (the colored-noise /
# TRN-spectral-model keys are GENUINELY non-numeric: a string enum, a file
# path, and nullable geometry). Each entry maps the key to a validator so the
# unsigned-exponent guard stays strong for every other leaf.
_TRN_PSD_MODELS = ("single_pole", "kondratiev_gorodetsky", "csv")
_TRN_CSV_UNITS = ("S_delta_T", "S_delta_omega")
NON_NUMERIC_ALLOWLIST = {
    "trn_psd_model": lambda v: v in _TRN_PSD_MODELS,
    "trn_csv_units": lambda v: v in _TRN_CSV_UNITS,
    "trn_psd_csv_path": lambda v: v is None or isinstance(v, str),
    "trn_R_m": lambda v: v is None or isinstance(v, (int, float)),
    "trn_da_m": lambda v: v is None or isinstance(v, (int, float)),
    "trn_db_m": lambda v: v is None or isinstance(v, (int, float)),
}


@pytest.fixture(scope="module")
def config():
    with open(CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def _str_leaves(node, path=""):
    """Yield (path, value) for every ``str`` leaf scalar in a nested structure."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _str_leaves(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _str_leaves(value, f"{path}[{i}]")
    elif isinstance(node, str):
        yield (path, node)


def test_no_string_leaves_under_physical_parameters(config):
    """No scalar under ``physical_parameters`` should load as a ``str``.

    A string leaf here means an exponent lost its sign and PyYAML failed to
    coerce the value to a float — the exact bug this guards against.
    """
    physical = config["physical_parameters"]
    string_leaves = [
        (path, val)
        for path, val in _str_leaves(physical, "physical_parameters")
        if path.rsplit(".", 1)[-1] not in NON_NUMERIC_ALLOWLIST
    ]
    assert not string_leaves, (
        "expected all physical_parameters to parse as numbers, but these "
        f"leaves are strings: {string_leaves}"
    )


def test_all_physical_parameters_are_numeric(config):
    """Every physical parameter is a native ``int`` or ``float``.

    Keys in ``NON_NUMERIC_ALLOWLIST`` are instead checked against their
    documented type/enum constraint.
    """
    physical = config["physical_parameters"]
    for key, value in physical.items():
        if key in NON_NUMERIC_ALLOWLIST:
            assert NON_NUMERIC_ALLOWLIST[key](value), (
                f"{key}={value!r} violates its allowlisted constraint"
            )
            continue
        assert isinstance(value, (int, float)) and not isinstance(value, bool), (
            f"{key}={value!r} did not parse as a number (type {type(value).__name__})"
        )


@pytest.mark.parametrize(
    "key, expected",
    [
        ("fsr_hz", 2.46e+10),
        ("intrinsic_q", 4.0e+7),
        ("coupling_q", 1.0e+7),
        ("d2_rad_per_s2", 3.76991e+4),
        ("kappa_i_rad_per_s", 3.038e+7),
        ("gamma_LLE_per_J_per_s", 1.029e+18),
    ],
)
def test_previously_unsigned_exponents_parse_to_expected_floats(config, key, expected):
    """The 7 formerly-unsigned-exponent values parse to their intended floats."""
    value = config["physical_parameters"][key]
    assert isinstance(value, float), f"{key} is {type(value).__name__}, not float"
    assert math.isclose(value, expected, rel_tol=1e-12), (key, value, expected)
