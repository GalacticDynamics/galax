"""Shared helpers for the variational-equation tests."""

import jax.numpy as jnp

import unxt as u

import galax.coordinates as gc
import galax.potential as gp
from galax.dynamics._src.orbit.field_base import AbstractOrbitField


def make_w0(xv, t=0.0):
    """Build a phase-space coordinate from a (..., 6) array in kpc, kpc/Myr."""
    return gc.PhaseSpaceCoordinate(
        q=u.Q(xv[..., :3], "kpc"), p=u.Q(xv[..., 3:], "kpc/Myr"), t=u.Q(t, "Myr")
    )


class LinearDragField(AbstractOrbitField):
    """Test-only field with a linear drag, a = -grad Phi - gamma v."""

    potential: gp.AbstractPotential
    gamma: float

    @property
    def units(self):
        return self.potential.units

    def __call__(self, t, qp, args=None, /):
        x, v = qp
        grad = self.potential._gradient(x, t)
        return v, -grad - self.gamma * jnp.asarray(v)
