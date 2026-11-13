"""Smoke tests for `galax.fields.custom_types`."""

import subprocess
import sys

import galax.fields.custom_types


def test_terms_alias():
    assert hasattr(galax.fields.custom_types, "Terms")


def test_no_dynamics_import():
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            (
                "import sys, galax.fields.custom_types; "
                "assert 'galax.dynamics' not in sys.modules"
            ),
        ],
        check=True,
    )
