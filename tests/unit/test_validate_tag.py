"""A package's CD workflow must refuse a tag belonging to another package."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "validate_tag.py"


def run(tag: str, package: str) -> int:
    return subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), tag, package], capture_output=True, check=False
    ).returncode


@pytest.mark.parametrize(
    ("tag", "package"),
    [
        ("galax-interop-gala-v0.1.0", "galax.interop.gala"),
        ("galax-interop-astropy-v1.2.3", "galax.interop.astropy"),
        ("galax-interop-matplotlib-v0.1.0rc1", "galax.interop.matplotlib"),
    ],
)
def test_accepts_its_own_tag(tag: str, package: str) -> None:
    assert run(tag, package) == 0


@pytest.mark.parametrize(
    ("tag", "package"),
    [
        # Another package's tag -- the failure this exists to prevent.
        ("galax-interop-gala-v0.1.0", "galax.interop.galpy"),
        ("galax-interop-astropy-v0.1.0", "galax.interop.gala"),
        # A prefix that merely starts the same: `gala` vs `galpy` is the real
        # hazard, since `str.startswith` would accept it.
        ("galax-interop-galpy-v0.1.0", "galax.interop.gala"),
        # The coordinator tag is not a package tag.
        ("v0.1.0", "galax.interop.gala"),
        # Malformed.
        ("galax-interop-gala-0.1.0", "galax.interop.gala"),
        ("galax-interop-gala-vabc", "galax.interop.gala"),
        # A valid prefix and version with extra text around it: only anchoring
        # at both ends rejects these.
        ("galax-interop-gala-v0.1.0-extra", "galax.interop.gala"),
        ("prefix-galax-interop-gala-v0.1.0", "galax.interop.gala"),
        # `$` alone matches before a trailing newline; only `fullmatch` rejects.
        ("galax-interop-gala-v0.1.0\n", "galax.interop.gala"),
    ],
)
def test_rejects_anything_else(tag: str, package: str) -> None:
    assert run(tag, package) != 0
