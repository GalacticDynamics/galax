"""galax: Galactic Dynamix in Jax."""

__all__ = [
    # class
    "HardCutoffNFWPotential",
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
from .base import (
    density as nfw_density,
    mass_enclosed as nfw_mass_enclosed,
    potential as nfw_potential,
)
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import (
    AbstractSinglePotential,
    LaplacianFromDensityMixin,
)
from galax.potential._src.builtin import kepler
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
class HardCutoffNFWPotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Sharply truncated NFW potential.

    The NFW density profile cut to zero outside the truncation radius ``r_t``.
    Inside ``r_t`` the density is that of the NFW profile, with an
    :math:`r^{-1}` cusp at the centre. Outside it the density is zero, so the
    mass is finite and the potential is that of a point mass. The parameter
    ``m`` is the characteristic mass of the untruncated NFW profile, and the
    total mass of this potential is smaller than ``m`` when
    :math:`r_t < \infty`.

    With :math:`x = r / r_s`, :math:`x_t = r_t / r_s` and
    :math:`A(x) = \ln(1 + x) - x / (1 + x)`, the density is

    .. math::

        \rho(r) = \begin{cases}
            \dfrac{m}{4 \pi r_s^3} \dfrac{1}{x (1 + x)^2}, & r \le r_t, \\
            0, & r > r_t,
        \end{cases}

    the enclosed mass is

    .. math::

        M(<r) = \begin{cases}
            m A(x), & r \le r_t, \\
            m A(x_t), & r > r_t,
        \end{cases}

    so that the total mass is :math:`M_{\rm tot} = m A(x_t)`, and the potential,
    with :math:`\Phi(r) \to 0` as :math:`r \to \infty`, is

    .. math::

        \Phi(r) = \begin{cases}
            -\dfrac{G m}{r_s} \dfrac{\ln(1 + x)}{x} + \dfrac{G m}{r_s + r_t},
                & r \le r_t, \\
            -\dfrac{G M_{\rm tot}}{r}, & r > r_t,
        \end{cases}

    where :math:`G` is the gravitational constant. The central potential is
    :math:`\Phi(0) = -G m / r_s + G m / (r_s + r_t)`, and the circular velocity
    is :math:`v_c^2(r) = G M(<r) / r`.

    Notes
    -----
    Inside :math:`r_t` the potential is the NFW potential of
    :class:`~galax.potential.NFWPotential` plus the constant
    :math:`G m / (r_s + r_t)`, which makes :math:`\Phi` continuous at
    :math:`r_t`. The gradient :math:`G M(<r) / r^2` is also continuous at
    :math:`r_t`, because :math:`M(<r)` is continuous. The density and the second
    derivative of the potential jump at :math:`r_t`.

    The truncation is a modelling choice and not a result of a particular
    paper. The profile inside :math:`r_t` is the NFW profile.

    References
    ----------
    .. [NFW1996HardCutoff] Navarro, J. F., Frenk, C. S., & White, S. D. M. (1996). The
       Structure of Cold Dark Matter Halos. Astrophysical Journal 462, 563.
       `ADS <https://ui.adsabs.harvard.edu/abs/1996ApJ...462..563N>`__;
       `doi:10.1086/177173 <https://doi.org/10.1086/177173>`__;
       `arXiv:astro-ph/9508025 <https://arxiv.org/abs/astro-ph/9508025>`__.
    """

    m: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass",
        doc=r"""Characteristic mass of the untruncated NFW profile. This is NOT the
        total mass, which is smaller.""",
    )
    r_s: AbstractParameter = ParameterField(dimensions="length", doc="Scale radius.")  # type: ignore[assignment]
    r_t: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="Truncation radius."
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    symmetry = Symmetry.SPHERICAL
    """`Symmetry.SPHERICAL`: the density depends only on $r$."""

    # TODO: make public
    @ft.partial(jax.jit)
    def _mass_enclosed(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BtFloatSz0:
        r"""Mass enclosed within a radius.

        .. math::

            M(<r)= \begin{cases}
                M_{NFW}(<r),   & \text{if } r \le r_t, \\
                M_{NFW}(<r_t), & \text{if } r > r_t
            \end{cases}

        where :math:`M_{NFW}(<r) = m [\ln(1 + x) - x / (1 + x)]` with
        :math:`x = r / r_s`, and the other parameters are defined on this class
        as the parameters of this potential.

        Examples
        --------
        >>> import unxt as u
        >>> import galax.potential as gp

        >>> pot = gp.HardCutoffNFWPotential(m=1e11, r_s=15, r_t=20, units="galactic")

        Evaluating at the truncation radius:

        >>> q = u.Q([20, 0, 0], "kpc")
        >>> t = u.Q(0, "Gyr")
        >>> pot._mass_enclosed(q, t)
        Array(2.75869289e+10, dtype=float64)

        Evaluating at a radius larger than the truncation radius:
        >>> q = u.Q([25, 0, 0], "kpc")
        >>> pot._mass_enclosed(q, t)
        Array(2.75869289e+10, dtype=float64)

        Evaluating at a radius smaller than the truncation radius:
        >>> q = u.Q([10, 0, 0], "kpc")
        >>> pot._mass_enclosed(q, t)
        Array(1.10825624e+10, dtype=float64)

        For comparison, here's a standard NFW potential:

        >>> nfw = gp.NFWPotential(m=1e11, r_s=15, units="galactic")
        >>> nfw._mass_enclosed(q, t)
        Array(1.10825624e+10, dtype=float64)

        """
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])

        params = self.params(t)
        return mass_enclosed(params, r)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _potential(  # TODO: inputs w/ units
        self, xyz: gt.Sz3, t: gt.Sz0, /
    ) -> gt.Sz0:
        r = r_spherical(xyz, self.units["length"])

        params = self.params(t)
        return potential(params, r)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BtFloatSz0:
        r"""Density.

        .. math::

            \rho(r) = \begin{cases}
                \rho_{NFW}(r) & r \le r_{t}, \\
                0,            & r > r_{t},
            \end{cases}

        where :math:`\rho_{NFW}(r) = \rho_s / [x (1 + x)^2]` with
        :math:`x = r / r_s` and :math:`\rho_s` is the characteristic density.
        The parameter is not :math:`\rho_s` but a characteristic mass :math:`m`,
        defined as :math:`m = 4\pi\rho_s r_s^3`.

        Examples
        --------
        >>> import unxt as u
        >>> import galax.potential as gp

        >>> pot = gp.HardCutoffNFWPotential(m=1e11, r_s=15, r_t=20, units="galactic")

        Evaluating at the truncation radius:
        >>> q = u.Q([20, 0, 0], "kpc")
        >>> t = u.Q(0, "Gyr")
        >>> pot.density(q, t)
        Q(324806.00630999, 'solMass / kpc3')

        Evaluating at a radius larger than the truncation radius:
        >>> q = u.Q([25, 0, 0], "kpc")
        >>> pot.density(q, t)
        Q(0., 'solMass / kpc3')

        Evaluating at a radius smaller than the truncation radius:
        >>> q = u.Q([10, 0, 0], "kpc")
        >>> pot.density(q, t)
        Q(1273239.54473516, 'solMass / kpc3')

        For comparison, here's a standard NFW potential:

        >>> nfw = gp.NFWPotential(m=1e11, r_s=15, units="galactic")
        >>> nfw.density(q, t)
        Q(1273239.54473516, 'solMass / kpc3')

        """
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])

        params = self.params(t)
        return density(params, r)  # type: ignore[no-any-return]


# ===================================================================


@ft.partial(jax.jit)
def density(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Density profile for the truncated NFW model.

    .. math::

        \rho(r) =
            \begin{cases}
                \rho_{NFW}(r) & r \le r_{t}, \\
                0,            & r > r_{t},
            \end{cases}

    where :math:`\rho_{NFW}(r)` is the NFW density profile and :math:`r_{t}` is
    the truncation radius beyond which the density is zero.

    """
    return jax.lax.select(r <= p["r_t"], nfw_density(p, r), jnp.zeros_like(r))


@ft.partial(jax.jit)
def mass_enclosed(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Mass enclosed within a radius.

    .. math::

        M(<r)= \begin{cases}
            M_{NFW}(<r), & \text{if } r \le r_t, \\
            M_{NFW}(<r_t), & \text{if } r > r_t,
        \end{cases}

    where :math:`M_{NFW}(<r)` is the NFW mass profile and :math:`r_t` is the
    truncation radius beyond which the mass is constant.
    :math:`M_{NFW}(<r)` is given by:

    .. math::

        M_{NFW}(<r) = m \left[\ln(1+x) - \frac{x}{1+x}\right]

    where :math:`x = r / r_s` is the dimensionless radius, :math:`r_s` is the
    scale radius and :math:`m = 4\pi\rho_s r_s^3` is the scale mass of the
    NFW profile.

    For more details see the NFW ``mass_enclosed`` function.

    """
    r_t = p["r_t"]
    _result = nfw_mass_enclosed(p, jnp.where(r <= r_t, r, r_t))
    return _result  # type: ignore[no-any-return]


# -------------------------------------------------------------------


@ft.partial(jax.jit)
def _inner_potential(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""NFW potential within the truncation.

    .. math::

        \Phi(r) = \Phi_{NFW}(r) + \frac{Gm}{r_s + r_t}

    """
    nfw = nfw_potential(p, r)
    constant = p["G"] * p["m"] / (p["r_s"] + p["r_t"])
    return nfw + constant  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def _outer_potential(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Map to Kepler potential.

    .. math::

        \Phi(r) = -\frac{G M_{tot}}{r}

    where :math:`M_{tot}` is the total mass of the potential and :math:`r` is
    the distance from the center of the potential. The total mass is given by
    the NFW mass enclosed within the truncation radius :math:`r_t`.
    """
    m_tot = nfw_mass_enclosed(p, p["r_t"])
    return kepler.point_mass_potential(p["G"], m_tot, r)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def potential(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Specific potential energy.

    .. math::

        \Phi(r) = \begin{cases}
            \Phi_{NFW}(r) + \frac{G m}{r_s + r_t}, & r \le r_t, \\
            -\frac{G M_{NFW}(<r_t)}{r}, & r > r_t,
        \end{cases}

    where

    - :math:`\Phi_{NFW}(r) = -(G m / r_s) \ln(1 + x) / x` is the NFW potential
      for :math:`x = r / r_s`
    - :math:`G, m, r_s` are the parameters of the NFW potential
    - :math:`r_t` is the truncation radius
    - :math:`M_{NFW}(<r_t)` is the enclosed mass of the NFW potential, given by
      :math:`M_{NFW}(<r_t) = m [\ln(1+x_t) - x_t / (1+x_t)]` for
      :math:`x_t = r_t / r_s`

    """
    _result = jnp.where(r <= p["r_t"], _inner_potential(p, r), _outer_potential(p, r))
    return _result  # type: ignore[no-any-return]


# Derived quantities, built from the kernels above.
dpotential_dr = radial_gradient(potential)
d2potential_dr2 = radial_second_derivative(dpotential_dr)
laplacian = poisson_laplacian(density)
gradient = spherical_gradient(dpotential_dr)
hessian = spherical_hessian(dpotential_dr, d2potential_dr2, laplacian)
