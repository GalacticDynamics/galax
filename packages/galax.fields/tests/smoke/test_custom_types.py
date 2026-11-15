"""Smoke tests for `galax.fields.custom_types`."""

import subprocess
import sys

import pytest

import galax.fields.custom_types

NAMES = ("LikeSz0", "RealScalarLike", "Sz0", "SzAny", "SzTime", "Terms")


@pytest.mark.parametrize("name", NAMES)
def test_alias_exists(name):
    assert hasattr(galax.fields.custom_types, name)


def test_imports_no_other_galax_portion():
    out = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            (
                "import sys, galax.fields.custom_types; "
                "print(sorted(m for m in sys.modules if m.startswith("
                "('galax.coordinates', 'galax.potential', 'galax.dynamics'))))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert out.stdout.strip() == "[]"
