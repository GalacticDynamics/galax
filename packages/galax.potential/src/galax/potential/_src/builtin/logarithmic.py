"""Private module that defines `LogarithmicPotential` and its functions.

The public API is `galax.potential.logarithmic`.
"""

__all__ = [
    # class
    "LogarithmicPotential",
    # functions
    "d2potential_dr2",
    "density",
    "dpotential_dr",
    "gradient",
    "hessian",
    "laplacian",
    "mass_enclosed",
    "potential",
]

import functools as ft
from dataclasses import KW_ONLY

from typing import final

import equinox as eqx
import jax

import quaxed.numpy as jnp
import unxt as u
from xmmutablemap import ImmutableMap

import galax.potential.custom_types as gt
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import (
    AbstractSinglePotential,
    LaplacianFromDensityMixin,
)
from galax.potential._src.kernels import (
    poisson_laplacian,
    spherical_gradient,
    spherical_hessian,
)
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.symmetry import Symmetry
from galax.potential._src.utils import r_spherical


@final
class LogarithmicPotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Logarithmic Potential.

    The spherical (:math:`q=1`) case of the flattened logarithmic potential
    of Binney & Tremaine 2008, *Galactic Dynamics*, 2nd ed., eq. 2.72 (a
    classic flat-rotation-curve, cored-halo model). The potential is

    .. math::

        \Phi(r) = \frac{1}{2} v_c^2 \ln\left(r_s^2 + r^2\right)

    and the corresponding density (their eq. 2.71, with :math:`q=1`) is

    .. math::

        \rho(r) = \frac{v_c^2 (3 r_s^2 + r^2)}{4\pi G (r_s^2+r^2)^2}.

    The mass enclosed within radius :math:`r` is :math:`M(<r) = v_c^2 r^3 /
    [G (r_s^2 + r^2)]`, and the circular speed is :math:`v_c r /
    \sqrt{r_s^2 + r^2}`, which tends to :math:`v_c` far out. The mass grows
    linearly with :math:`r` at large radius, so there is no finite total mass
    and the potential diverges logarithmically.

    """

    v_c: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="speed", doc="Circular velocity."
    )
    r_s: AbstractParameter = ParameterField(dimensions="length", doc="Scale length.")  # type: ignore[assignment]

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    symmetry = Symmetry.SPHERICAL
    """`Symmetry.SPHERICAL`: the density depends only on $r$."""

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])

        # Compute
        params = self.params(t)
        return potential(params, r)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])

        # Compute
        params = self.params(t)
        return density(params, r)  # type: ignore[no-any-return]


# ====================================================================


@ft.partial(jax.jit)
def potential(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Potential for the logarithmic potential.

    .. math::

        \Phi(r) = \frac{1}{2} v_c^2 \ln\left(r_s^2 + r^2\right)

    where :math:`v_c` is the asymptotic circular speed and :math:`r_s` is the
    scale length.
    """
    _result = 0.5 * p["v_c"] ** 2 * jnp.log(p["r_s"] ** 2 + r**2)
    return _result  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def density(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Density for the logarithmic potential.

    .. math::

        \rho(r) = \frac{v_c^2 (3 r_s^2 + r^2)}{4\pi G (r_s^2+r^2)^2}
    """
    numer = p["v_c"] ** 2 * (3 * p["r_s"] ** 2 + r**2)
    denom = 4 * jnp.pi * p["G"] * (p["r_s"] ** 2 + r**2) ** 2
    return numer / denom  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def dpotential_dr(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Radial derivative of the logarithmic potential.

    .. math::

        \Phi'(r) = \frac{v_c^2\, r}{r_s^2 + r^2}
    """
    return p["v_c"] ** 2 * r / (p["r_s"] ** 2 + r**2)


@ft.partial(jax.jit)
def d2potential_dr2(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Second radial derivative of the logarithmic potential.

    .. math::

        \Phi''(r) = \frac{v_c^2 (r_s^2 - r^2)}{(r_s^2 + r^2)^2}
    """
    r2 = r**2
    return p["v_c"] ** 2 * (p["r_s"] ** 2 - r2) / (p["r_s"] ** 2 + r2) ** 2


@ft.partial(jax.jit)
def mass_enclosed(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Enclosed mass for the logarithmic potential.

    .. math::

        M(<r) = \frac{v_c^2\, r^3}{G (r_s^2 + r^2)}
    """
    _result = p["v_c"] ** 2 * r**3 / (p["G"] * (p["r_s"] ** 2 + r**2))
    return _result


# Derived quantities, built from the kernels above.
laplacian = poisson_laplacian(density)
gradient = spherical_gradient(dpotential_dr)
hessian = spherical_hessian(dpotential_dr, d2potential_dr2, laplacian)
