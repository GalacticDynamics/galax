"""Compatibility.

TODO: make all the `gala` compat be in a linked package.

"""

__all__: list[str] = []

from typing import cast

try:  # TODO: less hacky way of supporting optional dependencies
    import pytest
except ImportError:  # pragma: no cover
    pass
else:
    _ = pytest.importorskip("gala")

import gala.dynamics as gd
from plum import conversion_method

import galax.dynamics as gdx

# =============================================================================
# Orbit


@conversion_method(type_from=gd.Orbit, type_to=gdx.Orbit)  # type: ignore[misc]
def gala_orbit_to_galax_orbit(obj: gd.Orbit, /) -> gdx.Orbit:
    """`gala.dynamics.Orbit` -> `galax.dynamics.Orbit`."""
    return gdx.Orbit(q=obj.pos, p=obj.vel, t=obj.t)


@gdx.Orbit.constructor._f.register  # type: ignore[misc]  # noqa: SLF001
def constructor(_: type[gdx.Orbit], obj: gd.Orbit, /) -> gdx.Orbit:
    """Construct a :mod:`galax` Orbit from a :mod:`gala` one."""
    return cast(gdx.Orbit, gala_orbit_to_galax_orbit(obj))


# =============================================================================
# MockStream


@conversion_method(type_from=gd.MockStream, type_to=gdx.MockStreamArm)  # type: ignore[misc]
def gala_mockstream_to_galax_mockstream(obj: gd.MockStream, /) -> gdx.MockStreamArm:
    """`gala.dynamics.MockStreamArm` -> `galax.dynamics.MockStreamArm`."""
    return gdx.MockStreamArm(q=obj.pos, p=obj.vel, release_time=obj.release_time)


@gdx.MockStreamArm.constructor._f.register  # type: ignore[misc]  # noqa: SLF001
def constructor(_: type[gdx.MockStreamArm], obj: gd.MockStream, /) -> gdx.MockStreamArm:
    """Construct a :mod:`galax` MockStreamArm from a :mod:`gala` one."""
    return cast(gdx.MockStreamArm, gala_mockstream_to_galax_mockstream(obj))
