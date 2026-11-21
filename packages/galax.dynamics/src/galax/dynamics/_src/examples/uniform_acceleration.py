"""Bar-typed potentials."""

__all__ = [
    "UniformAcceleration",
]

import functools as ft
from dataclasses import KW_ONLY

from collections.abc import Callable
from typing import final

import equinox as eqx
import jax

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax.dynamics.custom_types as gt
import galax.potential as gp


@final
class UniformAcceleration(gp.AbstractSinglePotential):
    """Spatially uniform acceleration field.

    This can be useful for working with non-inertial frames.

    Parameters
    ----------
    velocity_fn
        A function that takes the scalar time and returns the Cartesian velocity
        3-vector at that time. The derivative of this function is the
        acceleration of the frame. It must be a pytree that passes through
        `jax.jit`, e.g. an ``interpax`` spline or a plain function wrapped in
        `jax.tree_util.Partial`.
    func_supports_units
        Whether the velocity function supports units.
        It's safer to have the velocity function support units, but it's not
        required. If it does not support units, care must be taken to ensure
        that the units of the velocity function are consistent with the units
        used elsewhere.

    """

    velocity_func: Callable[[gt.Sz0], gt.FloatSz3]
    """The velocity function Callable[[Array[float, ()]], Array[float, (3,)]].

    The velocity function takes the time and returns the Cartesian velocity
    vector at that time.

    """

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    func_supports_units: bool = eqx.field(default=False, static=True)
    """Whether the velocity function supports units.

    It's safer to have the velocity function support units, but it's not
    required. If it does not support units, care must be taken to ensure that
    the units of the velocity function are consistent with the units used
    elsewhere.

    """

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0) -> gt.BBtSz0:
        # This does not have a potential
        raise NotImplementedError

    # `excluded=(0,)` skips `self`, as `vectorize_method` does for the base class.
    @ft.partial(jnp.vectorize, excluded=(0,), signature="(3),()->(3)")
    @ft.partial(jax.jit)
    def _gradient(self, _: gt.FloatQuSz3 | gt.FloatSz3, t: gt.QuSz0, /) -> gt.FloatSz3:
        # The gradient is the jacobian of the velocity function. `jax.jacfwd`
        # can't differentiate through a Quantity, so a unit-aware function is
        # wrapped to take and return plain values in `self.units`.
        t = u.ustrip(AllowValue, self.units["time"], t)
        vel_fn = self.velocity_func
        if self.func_supports_units:
            vel_fn = lambda t: u.ustrip(
                AllowValue,
                self.units["speed"],
                self.velocity_func(u.Q(t, self.units["time"])),
            )
        return jax.jacfwd(vel_fn)(t)  # type: ignore[no-any-return]

    @ft.partial(jnp.vectorize, excluded=(0,), signature="(3),()->(3,3)")
    @ft.partial(jax.jit)
    def _hessian(
        self, xyz: gt.FloatQuSz3 | gt.FloatSz3, t: gt.QuSz0 | gt.Sz0, /
    ) -> gt.Sz33:
        """Compute the hessian through the gradient."""
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.ustrip(AllowValue, self.units["time"], t)
        hess_op = jax.jacfwd(self._gradient, argnums=0)
        return hess_op(xyz, t)  # type: ignore[no-any-return]
