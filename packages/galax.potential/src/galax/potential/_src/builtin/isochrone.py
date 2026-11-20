"""galax: Galactic Dynamix in Jax."""

__all__ = [
    # class
    "IsochronePotential",
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
class IsochronePotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Isochrone potential.

    A spherical model with a constant-density core and a finite total mass. The
    potential is harmonic near the centre and approaches the point-mass
    potential far from it, and the scale radius ``r_s`` marks the transition
    between the two. The parameter ``m_tot`` is the total mass. Hénon (1959)
    introduced the model, and it is called isochrone because the radial period
    of an orbit in it depends only on the orbit's energy.

    The potential is

    .. math::

        \Phi(r) = -\frac{G m_{\mathrm{tot}}}{r_s + s},
        \qquad s = \sqrt{r^2 + r_s^2}

    and the density, which follows from Poisson's equation
    :math:`\nabla^2 \Phi = 4 \pi G \rho`, is

    .. math::

        \rho(r) = \frac{m_{\mathrm{tot}}\, r_s
                  \left(3 r_s s + 2 r^2 + 3 r_s^2\right)}
                  {4 \pi\, s^3 (r_s + s)^3}

    where :math:`r` is the distance from the centre, :math:`m_{\mathrm{tot}}`
    is the total mass, :math:`r_s` is the scale radius and :math:`G` is the
    gravitational constant.

    The mass enclosed within radius :math:`r` is

    .. math::

        M(<r) = \frac{m_{\mathrm{tot}}\, r^3}{s\, (r_s + s)^2}

    which tends to :math:`m_{\mathrm{tot}}` as :math:`r \to \infty`. The
    circular speed is :math:`v_c^2 = G M(<r)/r` and the escape speed is
    :math:`v_{\mathrm{esc}}^2 = -2 \Phi = 2 G m_{\mathrm{tot}} / (r_s + s)`.
    The central density is :math:`\rho(0) = 3 m_{\mathrm{tot}} / (16 \pi
    r_s^3)`, the central potential is :math:`\Phi(0) = -G m_{\mathrm{tot}} /
    (2 r_s)`, and the half-mass radius is :math:`\approx 3.06\, r_s`.

    Notes
    -----
    Near the centre :math:`\Phi \approx -G m_{\mathrm{tot}} / (2 r_s) + G
    m_{\mathrm{tot}}\, r^2 / (8 r_s^3)`, which is the potential of a harmonic
    oscillator. At large radius :math:`\Phi \to -G m_{\mathrm{tot}} / r` and
    the density falls as :math:`m_{\mathrm{tot}} r_s / (2 \pi r^4)`, the same
    :math:`r^{-4}` slope as :class:`~galax.potential.HernquistPotential`. Like
    :class:`~galax.potential.PlummerPotential` it has a core, and unlike
    that model its outer density slope is :math:`r^{-4}`.

    References
    ----------
    .. [Henon1959] Hénon, M. (1959). L'amas isochrone: I. Annales
       d'Astrophysique, 22, 126.
       `ADS <https://ui.adsabs.harvard.edu/abs/1959AnAp...22..126H>`_.
    """

    m_tot: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass", doc="Total mass of the potential."
    )

    r_s: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length",
        doc=r"""Scale radius of the potential.

    The value of :math:`r_s` defines the transition between the inner, more
    harmonic oscillator-like behavior of the potential, and the outer, :math:`1
    / r` Keplerian falloff.
    """,
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
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])

        params = self.params(t)
        return density(params, r)  # type: ignore[no-any-return]


# ===================================================================


@ft.partial(jax.jit)
def potential(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Potential function for the isochrone potential.

    .. math::

        \Phi(r) = -\frac{G m_{\mathrm{tot}}}{r_s + \sqrt{r^2 + r_s^2}}

    where :math:`G` is the gravitational constant, :math:`m_{\mathrm{tot}}` is
    the total mass and :math:`r_s` is the scale radius.
    """
    r_s = p["r_s"]
    _result = -p["G"] * p["m_tot"] / (r_s + jnp.sqrt(r**2 + r_s**2))
    return _result  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def density(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Density function for the isochrone potential.

    Derived from the potential via Poisson's equation.

    .. math::

        \rho(r) = \frac{m_{\mathrm{tot}}\, r_s
                  \left(3 r_s s + 2 r^2 + 3 r_s^2\right)}
                  {4 \pi\, s^3 (r_s + s)^3},
        \qquad s = \sqrt{r^2 + r_s^2}

    where :math:`m_{\mathrm{tot}}` is the total mass and :math:`r_s` is the
    scale radius.
    """
    r_s, m_tot = p["r_s"], p["m_tot"]
    s = jnp.sqrt(r**2 + r_s**2)
    numer = m_tot * r_s * (3 * r_s * s + 2 * r**2 + 3 * r_s**2)
    denom = 4 * jnp.pi * s**3 * (r_s + s) ** 3
    return numer / denom  # type: ignore[no-any-return]


# Derived quantities, built from the kernels above.
dpotential_dr = radial_gradient(potential)
d2potential_dr2 = radial_second_derivative(dpotential_dr)
mass_enclosed = mass_from_dpotential(dpotential_dr)
laplacian = poisson_laplacian(density)
gradient = spherical_gradient(dpotential_dr)
hessian = spherical_hessian(dpotential_dr, d2potential_dr2, laplacian)
