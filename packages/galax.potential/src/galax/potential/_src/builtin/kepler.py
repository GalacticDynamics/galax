"""Private module that defines `KeplerPotential` and its functions.

The public API is `galax.potential.kepler`.
"""

__all__ = [
    # class
    "KeplerPotential",
    # functions
    "d2potential_dr2",
    "density",
    "dpotential_dr",
    "gradient",
    "hessian",
    "laplacian",
    "mass_enclosed",
    "point_mass_potential",
    "potential",
]

import functools as ft
from dataclasses import KW_ONLY

from typing import final

import equinox as eqx
import jax

import quaxed.lax as qlax
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
    mass_from_dpotential,
    poisson_laplacian,
    radial_gradient,
    radial_second_derivative,
    spherical_gradient,
    spherical_hessian,
)
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.symmetry import Symmetry
from galax.potential._src.utils import r_spherical


@final
class KeplerPotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Kepler potential of a point mass.

    The potential is that of all the mass ``m_tot`` concentrated at the origin. It is
    spherical, the enclosed mass is the same at every radius, and there is no length
    scale. Bound orbits in it are Keplerian ellipses [BT08Kepler]_. As for the other
    parameters, ``m_tot`` can depend on time.

    The density is that of a point mass,

    .. math::

        \rho(r) = m_{\mathrm{tot}}\, \delta^{3}(\mathbf{r})

    and the potential is

    .. math::

        \Phi(r) = -\frac{G m_{\mathrm{tot}}}{r}

    where :math:`r` is the distance from the point mass, :math:`m_{\mathrm{tot}}`
    is its mass, :math:`\delta^{3}` is the three-dimensional Dirac delta
    function and :math:`G` is the gravitational constant. The potential
    satisfies Poisson's equation :math:`\nabla^2 \Phi = 4 \pi G \rho`.

    The mass enclosed within any radius :math:`r > 0` is
    :math:`M(<r) = m_{\mathrm{tot}}`. The circular speed is :math:`v_c^2 = G
    m_{\mathrm{tot}} / r` and the escape speed is :math:`v_{\mathrm{esc}}^2 =
    -2 \Phi = 2 G m_{\mathrm{tot}} / r`.

    Notes
    -----
    The potential and the density are singular at the origin. The ``density``
    function returns 0 for :math:`r > 0` and ``inf`` at :math:`r = 0` when
    ``m_tot`` is not zero, which stands in for the delta function. The
    potential falls as :math:`1/r` at all radii, unlike the finite-size models
    in galax such as :class:`~galax.potential.PlummerPotential`, which reduce
    to it far outside their scale radius.

    References
    ----------
    .. [BT08Kepler] Binney, J. & Tremaine, S. (2008). Galactic Dynamics, 2nd ed.
       Princeton University Press.
    """

    m_tot: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass", doc="Total mass of the potential."
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    symmetry = Symmetry.SPHERICAL
    """`Symmetry.SPHERICAL`: the density depends only on $r$."""

    @ft.partial(jax.jit)
    def _potential(  # TODO: inputs w/ units
        self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /
    ) -> gt.BBtSz0:
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])

        params = self.params(t)
        return potential(params, r)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BtFloatSz0:
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])

        params = self.params(t)
        return density(params, r)  # type: ignore[no-any-return]


# ============================================


@ft.partial(jax.jit, inline=True)
def point_mass_potential(G: gt.Sz0, m: gt.Sz0, r: gt.Sz0, /) -> gt.Sz0:
    r"""Potential energy for a point mass.

    .. math::

        \Phi(r) = -\frac{G m}{r}

    where :math:`G` is the gravitational constant and :math:`m` is the mass of
    the point mass.
    """
    return -G * m / r


# TODO: with units
@ft.partial(jax.jit)
def potential(p: gt.Params, r: gt.Sz0, /) -> gt.Sz0:
    r"""Kepler potential specific potential energy.

    .. math::

        \Phi(r) = -\frac{G m_{\mathrm{tot}}}{r}

    where :math:`m_{\mathrm{tot}}` is the total mass of the potential and
    :math:`r` is the distance from the center of the potential.
    """
    return point_mass_potential(p["G"], p["m_tot"], r)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def density(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Density profile for the Kepler potential.

    .. math::

        \rho(r) = m_{\mathrm{tot}}\, \delta^{3}(\mathbf{r})

    where :math:`\delta^{3}` is the three-dimensional Dirac delta function. The
    density is non-zero only at the origin, where the mass is located. The
    function returns ``inf`` at :math:`r = 0` when ``m_tot`` is not zero and 0
    everywhere else.
    """
    pred = jnp.logical_or(  # are we at the origin with non-zero mass?
        jnp.greater(r, jnp.zeros_like(r)),
        jnp.equal(p["m_tot"], jnp.zeros_like(p["m_tot"])),
    )
    _result = qlax.select(pred, jnp.zeros_like(r), jnp.full_like(r, fill_value=jnp.inf))
    return _result  # type: ignore[no-any-return]


# Derived quantities, built from the kernels above.
dpotential_dr = radial_gradient(potential)
d2potential_dr2 = radial_second_derivative(dpotential_dr)
mass_enclosed = mass_from_dpotential(dpotential_dr)
laplacian = poisson_laplacian(density)
gradient = spherical_gradient(dpotential_dr)
hessian = spherical_hessian(dpotential_dr, d2potential_dr2, laplacian)
