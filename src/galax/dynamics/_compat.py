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

import galax.dynamics as gdx


@conversion_method(type_from=gd.Orbit, type_to=gdx.Orbit)  # type: ignore[misc]
def gala_orbit_to_galax_orbit(obj: gd.Orbit, /) -> gdx.Orbit:
    """`gala.dynamics.Orbit` -> `galax.dynamics.Orbit`."""
    return gdx.Orbit(q=obj.pos, p=obj.vel, t=obj.t)


@conversion_method(type_from=gd.MockStream, type_to=gdx.MockStream)  # type: ignore[misc]
def gala_mockstream_to_galax_mockstream(obj: gd.MockStream, /) -> gdx.MockStream:
    """`gala.dynamics.MockStream` -> `galax.dynamics.MockStream`."""
    return gdx.MockStream(q=obj.pos, p=obj.vel, release_time=obj.release_time)
