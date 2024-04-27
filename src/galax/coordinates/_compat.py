"""Compatibility.

TODO: make all the `gala` compat be in a linked package.

"""

__all__: list[str] = []

try:  # TODO: less hacky way of supporting optional dependencies
    import pytest
except ImportError:
    pass
else:
    _ = pytest.importorskip("gala")

import gala.dynamics as gd
from plum import conversion_method

import galax.coordinates as gcx


@conversion_method(type_from=gd.PhaseSpacePosition, type_to=gcx.PhaseSpacePosition)  # type: ignore[misc]
def gala_psp_to_galax_psp(obj: gd.PhaseSpacePosition, /) -> gcx.PhaseSpacePosition:
    """`gala.dynamics.PhaseSpacePosition` -> `galax.coordinates.PhaseSpacePosition`."""
    return gcx.PhaseSpacePosition(q=obj.pos, p=obj.vel, t=None)
