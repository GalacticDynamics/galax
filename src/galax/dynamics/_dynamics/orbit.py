"""galax: Galactic Dynamics in Jax."""

__all__ = ["Orbit", "InterpolatedOrbit"]

from typing import final

import equinox as eqx

from coordinax import Abstract3DVector, Abstract3DVectorDifferential
from unxt import Quantity

import galax.potential as gp
from .base import AbstractOrbit
from galax.coordinates._psp.interp import PhaseSpacePositionInterpolant
from galax.coordinates._psp.utils import _p_converter, _q_converter
from galax.typing import BatchFloatQScalar, QVec1, QVecTime


@final
class Orbit(AbstractOrbit):
    """Represents an orbit.

    An orbit is a set of ositions and velocities (conjugate momenta) as a
    function of time resulting from the integration of the equations of motion
    in a given potential.
    """

    q: Abstract3DVector = eqx.field(converter=_q_converter)
    """Positions (x, y, z)."""

    p: Abstract3DVectorDifferential = eqx.field(converter=_p_converter)
    r"""Conjugate momenta ($v_x$, $v_y$, $v_z$)."""

    # TODO: consider how this should be vectorized
    t: QVecTime | QVec1 = eqx.field(converter=Quantity["time"].constructor)
    """Array of times corresponding to the positions."""

    potential: gp.AbstractPotentialBase
    """Potential in which the orbit was integrated."""


# ==========================================================================


@final
class InterpolatedOrbit(AbstractOrbit):
    """Orbit interpolated by the times."""

    q: Abstract3DVector = eqx.field(converter=_q_converter)
    """Positions (x, y, z)."""

    p: Abstract3DVectorDifferential = eqx.field(converter=_p_converter)
    r"""Conjugate momenta ($v_x$, $v_y$, $v_z$)."""

    # TODO: consider how this should be vectorized
    t: QVecTime | QVec1 = eqx.field(converter=Quantity["time"].constructor)
    """Array of times corresponding to the positions."""

    potential: gp.AbstractPotentialBase
    """Potential in which the orbit was integrated."""

    interpolant: PhaseSpacePositionInterpolant
    """The interpolation function."""

    def __call__(self, t: BatchFloatQScalar) -> Orbit:
        """Call the interpolation."""
        # TODO: more efficilent conversion to Orbit
        qp = self.interpolant(t)
        return Orbit(q=qp.q, p=qp.p, t=qp.t, potential=self.potential)
