"""Private module that defines `HernquistPotential` and its functions.

The public API is `galax.potential.hernquist`.
"""

__all__ = [
    # class
    "HernquistPotential",
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
class HernquistPotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Hernquist potential.

    The model is spherical, with a central density cusp and a finite total mass. The
    density goes as :math:`r^{-1}` well inside the scale radius ``r_s`` and as
    :math:`r^{-4}` well outside it. The parameter ``m_tot`` is the total mass. Hernquist
    [Hernquist1990]_ introduced the model as an analytic description of spherical
    galaxies and bulges whose projected profile is close to the de Vaucouleurs
    :math:`R^{1/4}` law.

    The density is

    .. math::

        \rho(r) = \frac{m_{\mathrm{tot}}\, r_s}{2 \pi\, r\, (r + r_s)^3}

    and the potential is

    .. math::

        \Phi(r) = -\frac{G m_{\mathrm{tot}}}{r + r_s}

    where :math:`r` is the distance from the centre, :math:`m_{\mathrm{tot}}`
    is the total mass, :math:`r_s` is the scale radius and :math:`G` is the
    gravitational constant. The potential satisfies Poisson's equation
    :math:`\nabla^2 \Phi = 4 \pi G \rho`.

    The mass enclosed within radius :math:`r` is

    .. math::

        M(<r) = m_{\mathrm{tot}} \frac{r^2}{(r + r_s)^2}

    which tends to :math:`m_{\mathrm{tot}}` as :math:`r \to \infty`. The
    circular speed is :math:`v_c^2 = G m_{\mathrm{tot}}\, r / (r + r_s)^2` and
    the escape speed is :math:`v_{\mathrm{esc}}^2 = -2 \Phi = 2 G
    m_{\mathrm{tot}} / (r + r_s)`. The central potential is :math:`\Phi(0) = -G
    m_{\mathrm{tot}} / r_s`, which is finite, and the half-mass radius is
    :math:`(1 + \sqrt{2})\, r_s`.

    Notes
    -----
    The density diverges at the origin like :math:`m_{\mathrm{tot}} / (2 \pi
    r_s^2 r)`, while the potential and the gradient stay finite. At large radius
    the potential approaches the point-mass potential :math:`-G
    m_{\mathrm{tot}} / r`. The profile is cuspy, unlike
    :class:`~galax.potential.PlummerPotential` and
    :class:`~galax.potential.IsochronePotential`, which have constant-density
    cores, and it is shallower in the centre than the :math:`r^{-2}` cusp of
    :class:`~galax.potential.JaffePotential`.
    :class:`~galax.potential.TriaxialHernquistPotential` evaluates this
    potential on a triaxial radius.

    References
    ----------
    .. [Hernquist1990] Hernquist, L. (1990). An analytical model for spherical
       galaxies and bulges. The Astrophysical Journal, 356, 359.
       `ADS <https://ui.adsabs.harvard.edu/abs/1990ApJ...356..359H>`__;
       `doi:10.1086/168845 <https://doi.org/10.1086/168845>`__.
    """

    m_tot: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass", doc="Total mass of the potential."
    )

    r_s: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="Scale radius."
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    symmetry = Symmetry.SPHERICAL
    """`Symmetry.SPHERICAL`: the density depends only on $r$."""

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        r = r_spherical(xyz, self.units["length"])

        params = self.params(t)
        return potential(params, r)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BtFloatSz0:
        r = r_spherical(xyz, self.units["length"])

        params = self.params(t)
        return density(params, r)  # type: ignore[no-any-return]


# ============================================


@ft.partial(jax.jit)
def density(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Density profile for the Hernquist potential.

    .. math::

        \rho(r) = \frac{m_{\mathrm{tot}}\, r_s}{2 \pi\, r\, (r + r_s)^3}

    where :math:`m_{\mathrm{tot}}` is the total mass and :math:`r_s` is the
    scale radius.
    """
    s = r / p["r_s"]
    rho0 = p["m_tot"] / (2 * jnp.pi * p["r_s"] ** 3)
    return rho0 / (s * (1 + s) ** 3)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def potential(p: gt.Params, r: gt.Sz0, /) -> gt.Sz0:
    r"""Specific potential energy for the Hernquist potential.

    .. math::

        \Phi(r) = -\frac{G m_{\mathrm{tot}}}{r + r_s}

    where :math:`G` is the gravitational constant, :math:`m_{\mathrm{tot}}` is
    the total mass and :math:`r_s` is the scale radius.
    """
    return -p["G"] * p["m_tot"] / (r + p["r_s"])


# Derived quantities, built from the kernels above.
dpotential_dr = radial_gradient(potential)
d2potential_dr2 = radial_second_derivative(dpotential_dr)
mass_enclosed = mass_from_dpotential(dpotential_dr)
laplacian = poisson_laplacian(density)
gradient = spherical_gradient(dpotential_dr)
hessian = spherical_hessian(dpotential_dr, d2potential_dr2, laplacian)
