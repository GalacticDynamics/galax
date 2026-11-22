"""`galax.dynamics` must import under jaxtyping's hook with beartype.

Runtime type checking is switched on from outside galax, by wrapping the import
in `jaxtyping.install_import_hook`. A hint beartype cannot handle makes the
import itself fail, so it is checked in a fresh interpreter: the import order is
the thing under test.
"""

import subprocess
import sys

import pytest
from beartype import beartype

from galax.dynamics._src.cluster.api import L1L2LagrangePoints


def test_lagrange_points_is_a_beartype_checkable_return_hint() -> None:
    """Beartype rejects a generic `NamedTuple`, which `L1L2LagrangePoints` was."""

    @beartype
    def points() -> L1L2LagrangePoints:
        return L1L2LagrangePoints(l1=1.0, l2=2.0)

    assert points() == (1.0, 2.0)


@pytest.mark.slow
def test_imports_under_the_runtime_typechecking_hook() -> None:
    """The import hook with beartype instruments `galax.dynamics` without error."""
    code = (
        "from jaxtyping import install_import_hook\n"
        "with install_import_hook('galax.dynamics', 'beartype.beartype'):\n"
        "    import galax.dynamics\n"
    )
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
