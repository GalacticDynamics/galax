"""Private module that defines `PlummerPotential` and its functions.

The public API is `galax.potential.plummer`.
"""

__all__ = [
    # class
    "PlummerPotential",
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
class PlummerPotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Plummer potential.

    The model is spherical, with a constant-density core and a finite total
    mass. The density is flat inside the scale radius ``r_s`` and falls as
    :math:`r^{-5}` far outside it. The parameter ``m_tot`` is the total mass
    and ``r_s`` sets the size of the core. Plummer [Plummer1911]_ introduced
    the density law to fit observations of globular clusters.

    The density is

    .. math::

        \rho(r) = \frac{3 m_{\mathrm{tot}}}{4 \pi r_s^3}
                  \left(1 + \frac{r^2}{r_s^2}\right)^{-5/2}

    and the potential is

    .. math::

        \Phi(r) = -\frac{G m_{\mathrm{tot}}}{\sqrt{r^2 + r_s^2}}

    where :math:`r` is the distance from the centre, :math:`m_{\mathrm{tot}}`
    is the total mass, :math:`r_s` is the scale radius and :math:`G` is the
    gravitational constant. The potential satisfies Poisson's equation
    :math:`\nabla^2 \Phi = 4 \pi G \rho`.

    The mass enclosed within radius :math:`r` is

    .. math::

        M(<r) = \frac{m_{\mathrm{tot}}\, r^3}{(r^2 + r_s^2)^{3/2}}

    which tends to :math:`m_{\mathrm{tot}}` as :math:`r \to \infty`. The
    circular speed is :math:`v_c^2 = G M(<r)/r = G m_{\mathrm{tot}}\, r^2 /
    (r^2 + r_s^2)^{3/2}` and the escape speed is :math:`v_{\mathrm{esc}}^2 =
    -2 \Phi = 2 G m_{\mathrm{tot}} / \sqrt{r^2 + r_s^2}`. The central density
    is :math:`\rho(0) = 3 m_{\mathrm{tot}} / (4 \pi r_s^3)`, the central
    potential is :math:`\Phi(0) = -G m_{\mathrm{tot}} / r_s`, and the
    half-mass radius is :math:`r_s / \sqrt{2^{2/3} - 1} \approx 1.305\, r_s`.

    Notes
    -----
    The potential is smooth at the origin. At large radius it approaches the
    point-mass potential :math:`-G m_{\mathrm{tot}} / r`. The profile has a core,
    unlike :class:`~galax.potential.HernquistPotential`, which has a
    :math:`1/r` cusp, and it falls more steeply (:math:`r^{-5}`) than the
    :math:`r^{-4}` of the Hernquist and isochrone profiles.

    References
    ----------
    .. [Plummer1911] Plummer, H. C. (1911). On the problem of distribution in
       globular star clusters. Monthly Notices of the Royal Astronomical
       Society, 71, 460-470.
       `ADS <https://ui.adsabs.harvard.edu/abs/1911MNRAS..71..460P>`__;
       `doi:10.1093/mnras/71.5.460 <https://doi.org/10.1093/mnras/71.5.460>`__.
    """

    m_tot: AbstractParameter = ParameterField(dimensions="mass", doc="Total mass.")  # type: ignore[assignment]
    r_s: AbstractParameter = ParameterField(dimensions="length", doc="Scale length.")  # type: ignore[assignment]

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    symmetry = Symmetry.SPHERICAL
    """`Symmetry.SPHERICAL`: the density depends only on $r$."""

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        ul = self.units["length"]
        r = r_spherical(xyz, ul)

        params = self.params(t)
        return density(params, r)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        ul = self.units["length"]
        r = r_spherical(xyz, ul)

        params = self.params(t)
        return potential(params, r)  # type: ignore[no-any-return]


# ===================================================================


@ft.partial(jax.jit)
def density(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Density profile for the Plummer potential.

    .. math::

        \rho(r) = \frac{3 m_{\mathrm{tot}}}{4 \pi r_s^3}
                  \left(1 + \frac{r^2}{r_s^2}\right)^{-5/2}

    where :math:`m_{\mathrm{tot}}` is the total mass and :math:`r_s` is the
    scale radius.
    """
    rho0 = 3 * p["m_tot"] / (4 * jnp.pi * p["r_s"] ** 3)
    return rho0 / jnp.power(1 + (r / p["r_s"]) ** 2, 2.5)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def mass_enclosed(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Enclosed mass for the Plummer potential.

    .. math::

        M(<r) = \frac{m_{\mathrm{tot}}\, r^3}{(r^2 + r_s^2)^{3/2}}
    """
    _result = p["m_tot"] * r**3 / jnp.power(r**2 + p["r_s"] ** 2, 1.5)
    return _result  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def potential(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Potential function for the Plummer potential.

    .. math::

        \Phi(r) = -\frac{G m_{\mathrm{tot}}}{\sqrt{r^2 + r_s^2}}

    where :math:`G` is the gravitational constant, :math:`m_{\mathrm{tot}}` is
    the total mass and :math:`r_s` is the scale radius.
    """
    _result = -p["G"] * p["m_tot"] / jnp.sqrt(r**2 + p["r_s"] ** 2)
    return _result  # type: ignore[no-any-return]


# Derived quantities, built from the kernels above.
dpotential_dr = radial_gradient(potential)
d2potential_dr2 = radial_second_derivative(dpotential_dr)
laplacian = poisson_laplacian(density)
gradient = spherical_gradient(dpotential_dr)
hessian = spherical_hessian(dpotential_dr, d2potential_dr2, laplacian)
