"""Bar-typed potentials."""

__all__ = [
    "UniformAcceleration",
]

from collections.abc import Callable
from functools import partial
from typing import final

import equinox as eqx
import jax

import unxt as u
from unxt.quantity import AllowValue

import galax._custom_types as gt
from galax.potential._src.base_single import AbstractSinglePotential


@final
class UniformAcceleration(AbstractSinglePotential):
    """Spatially uniform acceleration field.

    This can be useful for working with non-inertial frames.

    Parameters
    ----------
    velocity_fn
        A function that takes the scalar time and returns the Cartesian velocity
        3-vector at that time. The derivative of this function is the
        acceleration of the frame.
    supports_units
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

    supports_units: bool = eqx.field(default=False, static=True)
    """Whether the velocity function supports units.

    It's safer to have the velocity function support units, but it's not
    required. If it does not support units, care must be taken to ensure that
    the units of the velocity function are consistent with the units used
    elsewhere.

    """

    @partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0) -> gt.BBtSz0:
        # This does not have a potential
        raise NotImplementedError

    @partial(jax.jit)
    def _gradient(self, _: gt.FloatQuSz3 | gt.FloatSz3, t: gt.QuSz0, /) -> gt.FloatSz3:
        # The gradient is the jacobian of the velocity function
        if not self.supports_units:
            t = u.ustrip(AllowValue, self.units["time"], t)

        grad = jax.jacfwd(self.velocity_func)(t)
        return u.ustrip(AllowValue, self.units["acceleration"], grad)
