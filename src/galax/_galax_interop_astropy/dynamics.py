"""Compatibility."""

__all__: list[str] = []

from functools import partial
from typing import Literal

import jax
from astropy.units import Quantity as APYQuantity
from plum import dispatch

import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp
import galax.typing as gt

# =============================================================================
# evaluate_orbit


@dispatch  # type: ignore[misc]
@partial(jax.jit, static_argnames=("integrator", "interpolated"))
def evaluate_orbit(
    pot: gp.AbstractPotentialBase,
    w0: gc.PhaseSpacePosition | gt.BatchVec6,
    t: APYQuantity,
    *,
    integrator: gd.Integrator | None = None,
    interpolated: Literal[True, False] = False,
) -> gd.Orbit | gd.InterpolatedOrbit:
    out: gd.Orbit | gd.InterpolatedOrbit
    out = gd.evaluate_orbit(
        pot, w0, t, integrator=integrator, interpolated=interpolated
    )
    return out
