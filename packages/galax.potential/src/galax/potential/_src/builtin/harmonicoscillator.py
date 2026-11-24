"""Private module that defines `HarmonicOscillatorPotential` and its functions.

The public API is `galax.potential.harmonicoscillator`.
"""

__all__ = [
    # class
    "HarmonicOscillatorPotential",
    # functions
    "density",
    "gradient",
    "hessian",
    "laplacian",
    "potential",
]

import functools as ft
from dataclasses import KW_ONLY

from typing import final

import equinox as eqx
import jax

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue
from xmmutablemap import ImmutableMap

import galax.potential.custom_types as gt
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.kernels import cartesian_gradient, cartesian_hessian
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.symmetry import Symmetry


@final
class HarmonicOscillatorPotential(AbstractSinglePotential):
    r"""Harmonic oscillator potential.

    The potential of a three-dimensional harmonic oscillator,

    .. math::

        \Phi(\mathbf{q}, t) = \frac{1}{2} \sum_i (\omega_i(t)\, q_i)^2,

    where :math:`q_i` are the Cartesian coordinates and :math:`\omega_i` the
    frequencies. ``omega`` is a scalar, shared by all axes, or a vector with one
    frequency per axis. The potential is isotropic only for a scalar ``omega``,
    so the class asserts only the reflection symmetry in each coordinate plane.

    The density follows from Poisson's equation, :math:`\nabla^2 \Phi = 4 \pi G
    \rho`, and is uniform,

    .. math::

        \rho = \frac{1}{4 \pi G} \sum_i \omega_i^2.

    The density needs the gravitational constant, so it fails with a
    `ValueError` if ``constants`` has no ``"G"``; the Laplacian does not.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.HarmonicOscillatorPotential(omega=u.Q(1, "1 / Myr"),
    ...                                      units="galactic")
    >>> pot
    HarmonicOscillatorPotential(
      units=LTMAUnitSystem( ... ),
      constants=ImmutableMap({'G': ...}),
      omega=ConstantParameter(value=Quantity(weak_i64[], unit='1 / Myr'))
    )

    >>> q = u.Q([1.0, 0, 0], "kpc")
    >>> t = u.Q(0, "Gyr")

    >>> pot.potential(q, t)
    Q(0.5, 'kpc2 / Myr2')

    >>> pot.density(q, t)
    Q(5.30693121e+10, 'solMass / kpc3')

    A vector ``omega`` gives one frequency per axis.

    >>> pot = gp.HarmonicOscillatorPotential(omega=u.Q([1.0, 2.0, 3.0], "1 / Myr"),
    ...                                      units="galactic")
    >>> pot.potential(u.Q([1.0, 1.0, 1.0], "kpc"), t)
    Q(7., 'kpc2 / Myr2')

    """

    omega: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="frequency", doc="The frequency."
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    symmetry = Symmetry.PLANE_REFLECTION
    """`Symmetry.PLANE_REFLECTION`: the potential is even in $x$, $y$ and $z$."""

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)

        # Compute
        params = self.params(t)
        return potential(params, xyz)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtFloatSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)

        # Compute
        params = self.params(t)
        return density(params, xyz)  # type: ignore[no-any-return]


# ====================================================================


@ft.partial(jax.jit)
def potential(p: gt.Params, xyz: gt.BBtSz3, /) -> gt.BBtSz0:
    r"""Potential of the harmonic oscillator.

    .. math::

        \Phi(\mathbf{q}) = \frac{1}{2} \sum_i (\omega_i\, q_i)^2

    where :math:`\omega_i` is ``p["omega"]``, a scalar or a vector of three
    frequencies.
    """
    _result = 0.5 * jnp.sum(jnp.square(jnp.atleast_1d(p["omega"]) * xyz), axis=-1)
    return _result  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def laplacian(p: gt.Params, xyz: gt.BBtSz3, /) -> gt.BBtFloatSz0:
    r"""Laplacian of the harmonic oscillator potential, uniform in space.

    .. math::

        \nabla^2 \Phi = \sum_i \omega_i^2

    A scalar ``omega`` is shared by all three axes, as in `potential`. The
    result does not need the gravitational constant.
    """
    omega = jnp.broadcast_to(p["omega"], xyz.shape)
    return jnp.sum(omega**2, axis=-1)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def density(p: gt.Params, xyz: gt.BBtSz3, /) -> gt.BBtFloatSz0:
    r"""Density of the harmonic oscillator, uniform in space.

    .. math::

        \rho = \frac{1}{4 \pi G} \sum_i \omega_i^2

    The result needs ``p["G"]``, which is missing when the potential was built
    with ``constants={}``; then this raises a `ValueError`.
    """
    if "G" not in p:
        msg = "the density needs the gravitational constant 'G' in `constants`"
        raise ValueError(msg)
    return laplacian(p, xyz) / (4 * jnp.pi * p["G"])  # type: ignore[no-any-return]


# Derived quantities, built from the kernels above.
gradient = cartesian_gradient(potential)
hessian = cartesian_hessian(potential)
