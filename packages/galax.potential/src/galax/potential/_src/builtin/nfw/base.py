"""Private module that defines `NFWPotential` and its functions.

The public API is `galax.potential.nfw`.
"""

__all__ = [
    # class
    "NFWPotential",
    # functions
    "d2potential_dr2",
    "density",
    "dpotential_dr",
    "gradient",
    "hessian",
    "laplacian",
    "m_of_rho0",
    "mass_enclosed",
    "potential",
    "rho0_of_m",
]

import functools as ft
from dataclasses import KW_ONLY

from typing import Final, final

import equinox as eqx
import jax
import numpy as np

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
from galax.potential._src.utils import horner, r_spherical, radial_derivatives


@final
class NFWPotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""NFW potential.

    The model is the spherical density profile that Navarro, Frenk & White
    found to fit haloes in cold dark matter simulations [NFW1996NFW]_
    [NFW1997NFW]_. The density falls as :math:`r^{-1}`
    inside the scale radius ``r_s`` and as :math:`r^{-3}` outside it, so the
    enclosed mass grows logarithmically without bound and the total mass is
    infinite. The density diverges at the centre but the potential is finite
    there. The parameter ``m`` is a mass scale, :math:`m = 4 \pi \rho_s r_s^3`
    for the characteristic density :math:`\rho_s`, and it is not the total mass.

    With :math:`x = r / r_s` the density is

    .. math::

        \rho(r) = \frac{\rho_s}{x (1 + x)^2},
        \qquad \rho_s = \frac{m}{4 \pi r_s^3},

    the enclosed mass is

    .. math::

        M(<r) = m \left[ \ln(1 + x) - \frac{x}{1 + x} \right],

    and the potential, with :math:`\Phi(r) \to 0` as :math:`r \to \infty`, is

    .. math::

        \Phi(r) = -\frac{G m}{r_s} \frac{\ln(1 + x)}{x},

    where :math:`G` is the gravitational constant. The potential is finite at
    the centre, :math:`\Phi(0) = -G m / r_s`. The circular velocity is
    :math:`v_c^2(r) = G M(<r) / r`, which peaks at :math:`x \approx 2.163` with
    :math:`v_c^2 \approx 0.2162\, G m / r_s`, and the escape speed is
    :math:`v_{\rm esc}(r) = \sqrt{-2 \Phi(r)}`.

    Notes
    -----
    A halo of mass :math:`M_{200}` inside the radius :math:`r_{200}` that
    encloses 200 times the critical density and has concentration
    :math:`c = r_{200} / r_s` has :math:`r_s = r_{200} / c` and
    :math:`m = M_{200} / [\ln(1 + c) - c / (1 + c)]`. The constructor
    :meth:`~galax.potential.NFWPotential.from_M200_c` uses this relation.

    The truncated variant is
    :class:`~galax.potential.HardCutoffNFWPotential` and the family with a free
    inner slope is :class:`~galax.potential.gNFWPotential`.

    References
    ----------
    .. [NFW1996NFW] Navarro, J. F., Frenk, C. S., & White, S. D. M. (1996). The
       Structure of Cold Dark Matter Halos. Astrophysical Journal 462, 563.
       `ADS <https://ui.adsabs.harvard.edu/abs/1996ApJ...462..563N>`__;
       `doi:10.1086/177173 <https://doi.org/10.1086/177173>`__;
       `arXiv:astro-ph/9508025 <https://arxiv.org/abs/astro-ph/9508025>`__.
    .. [NFW1997NFW] Navarro, J. F., Frenk, C. S., & White, S. D. M. (1997). A
       Universal Density Profile from Hierarchical Clustering. Astrophysical
       Journal 490, 493.
       `ADS <https://ui.adsabs.harvard.edu/abs/1997ApJ...490..493N>`__;
       `doi:10.1086/304888 <https://doi.org/10.1086/304888>`__;
       `arXiv:astro-ph/9611107 <https://arxiv.org/abs/astro-ph/9611107>`__.
    """

    m: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass", doc="Mass parameter. This is NOT the total mass."
    )

    r_s: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="Scale radius of the potential."
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
        r"""Potential energy.

        .. math::

            \Phi(r) = -\frac{G m}{r_s} \frac{\ln(1 + x)}{x},
            \qquad x = r / r_s
        """
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])

        params = self.params(t)
        return potential(params, r)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BtFloatSz0:
        r"""Density.

        .. math::

            \rho(r) = \frac{\rho_s}{x (1 + x)^2},
            \qquad \rho_s = \frac{m}{4 \pi r_s^3},
            \qquad x = r / r_s

        """
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])

        params = self.params(t)
        return density(params, r)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _mass_enclosed(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BtFloatSz0:
        r"""Enclosed mass.

        .. math::

            M(<r) = m \left[ \ln(1 + x) - \frac{x}{1 + x} \right],
            \qquad x = r / r_s

        Examples
        --------
        >>> import unxt as u
        >>> import galax.potential as gp

        >>> nfw = gp.NFWPotential(m=1e11, r_s=15, units="galactic")

        >>> q = u.Q([10, 0, 0], "kpc")
        >>> t = u.Q(0, "Gyr")
        >>> nfw._mass_enclosed(q, t)
        Array(1.10825624e+10, dtype=float64)

        """
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])

        params = self.params(t)
        return mass_enclosed(params, r)  # type: ignore[no-any-return]

    # ===========================================
    # Constructors

    @staticmethod
    def _vc_rs_rref_to_m(
        v_c: u.Quantity["velocity"],
        r_s: u.Quantity["length"],
        r_ref: u.Quantity["length"],
        *,
        G: u.AbstractQuantity,
    ) -> u.Quantity["mass"]:
        uu = r_ref / r_s
        vs2 = v_c**2 / uu / (jnp.log(1 + uu) / uu**2 - 1 / (uu * (1 + uu)))
        return r_s * vs2 / G

    @classmethod
    def from_circular_velocity(
        cls,
        v_c: u.Quantity["velocity"],
        r_s: u.Quantity["length"],
        r_ref: u.Quantity["length"] | None = None,
        *,
        units: u.AbstractUnitSystem | str = "galactic",
        constants: ImmutableMap[str, u.AbstractQuantity] = default_constants,
    ) -> "NFWPotential":
        r"""Create an NFW potential from the circular velocity at a given radius.

        Parameters
        ----------
        v_c
            Circular velocity (at the specified reference radius).
        r_s
            Scale radius.
        r_ref (optional)
            The reference radius for the circular velocity. If None, the scale radius is
            used.

        Returns
        -------
        NFWPotential
            NFW potential instance with the given circular velocity and scale radius.
        """
        r_ref = r_s if r_ref is None else r_ref
        usys = u.unitsystem(units)
        m = cls._vc_rs_rref_to_m(v_c, r_s, r_ref, G=constants["G"])
        return cls(m=m, r_s=r_s, units=usys)

    @classmethod
    def from_M200_c(
        cls,
        M200: u.Quantity["mass"],
        c: u.Quantity["dimensionless"],
        rho_c: u.Quantity["mass density"] | None = None,
        *,
        units: u.AbstractUnitSystem | str,
    ) -> "NFWPotential":
        """Create an NFW potential from a virial mass and concentration.

        Parameters
        ----------
        M200
            Virial mass, or mass at 200 times the critical density, ``rho_c``.
        c
            NFW halo concentration.
        rho_c (optional)
            Critical density at z=0. If not specified, uses the default astropy
            cosmology to obtain this, `~astropy.cosmology.default_cosmology`.
        """
        usys = u.unitsystem(units)
        if rho_c is None:
            # Deferred: `astropy.cosmology` costs ~0.7s to import, and only
            # this one branch needs it.
            from astropy.cosmology import default_cosmology

            cosmo = default_cosmology.get()
            rho_c = (3 * cosmo.H(0.0) ** 2 / (8 * np.pi * default_constants["G"])).to(
                usys["mass density"]
            )
            rho_c = u.Q(rho_c.value, usys["mass density"])

        r_vir = jnp.cbrt(M200 / (200 * rho_c) / (4.0 / 3 * jnp.pi))
        r_s = r_vir / c

        A_NFW = jnp.log(1 + c) - c / (1 + c)
        m = M200 / A_NFW

        return cls(m=m, r_s=r_s, units=usys)


# =============================================================


# -----------------------------------------------


@ft.partial(jax.jit)
def rho0_of_m(p: gt.Params, /) -> gt.Sz0:
    r"""Characteristic density for the NFW model.

    The NFW profile is parametrized by a characteristic mass :math:`m` and a
    scale radius :math:`r_s`. The characteristic density :math:`\rho_s` that
    multiplies :math:`1 / [x (1 + x)^2]` is (the density itself diverges at the
    centre)

    .. math::

        \rho_s = \frac{m}{4 \pi r_s^3}.

    Examples
    --------
    >>> import jax.numpy as jnp

    A quick sanity check:

    >>> rho0_of_m({"m": 1.0, "r_s": 1.0}) - 1 / (4 * jnp.pi)
    Array(0., dtype=float64, weak_type=True)

    """
    return p["m"] / (4 * jnp.pi * p["r_s"] ** 3)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def m_of_rho0(p: gt.Params, /) -> gt.Sz0:
    r"""Characteristic mass for the NFW model.

    The NFW profile is parametrized by a characteristic mass :math:`m` and a
    scale radius :math:`r_s`. The characteristic mass is given by

    .. math::

        m = 4 \pi \rho_s r_s^3,

    where :math:`\rho_s` is the characteristic density (the ``"rho0"`` entry).

    Examples
    --------
    >>> import jax.numpy as jnp

    A quick sanity check:
    >>> m_of_rho0({"rho0": 1.0, "r_s": 1.0}) - 4 * jnp.pi
    Array(0., dtype=float64, weak_type=True)

    """
    return 4 * jnp.pi * p["rho0"] * p["r_s"] ** 3  # type: ignore[no-any-return]


# -----------------------------------------------


@ft.partial(jax.jit)
def density(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""Density profile for the NFW model.

    .. math::

        \rho(r) = \frac{\rho_s}{\frac{r}{r_s} (1 + \frac{r}{r_s})^2}

    where :math:`\rho_s` is the characteristic density and :math:`r_s` is the
    scale radius. The parameter is not :math:`\rho_s` but a characteristic mass
    :math:`m`, defined as

    .. math::

        m = 4 \pi \rho_s r_s^3.

    """
    x = r / p["r_s"]
    rho0: gt.Sz0 = rho0_of_m(p)
    return rho0 / (x * (1 + x) ** 2)


@ft.partial(jax.jit)
def mass_enclosed(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""Enclosed mass for the NFW model.

    .. math::

        M(<r) = m \left[ \ln(1 + x) - \frac{x}{1 + x} \right]

    where :math:`x = r / r_s` is the dimensionless radius and :math:`m` is the
    characteristic mass.

    """
    x = r / p["r_s"]
    return p["m"] * _log1p_minus_x_over_1px(x)


_ATANH_TAIL: Final = tuple(1.0 / (2 * k + 1) for k in range(18, 0, -1))
"""(atanh(w) - w) / w^3 as a polynomial in w^2, highest degree first."""


def _log1p_minus_x_over_1px(x: gt.BBtSz0, /) -> gt.BBtSz0:
    r"""$\ln(1+x) - x/(1+x)$, to a few ulp for every $x \geq 0$.

    The direct difference is $\sim x^2/2$ from two $\sim x$ terms, so it loses
    every digit as $x \to 0$. With $w = x/(2+x)$, $\ln(1+x) = 2\,\text{atanh}\,w$
    and $x/(1+x) = 2w/(1+w)$, so

    .. math::

        \ln(1+x) - \frac{x}{1+x}
        = 2 \sum_{k \geq 1} \frac{w^{2k+1}}{2k+1} + \frac{2w^2}{1+w},

    every term positive. Up to $x = 1$ ($w \leq 1/3$) 18 terms reach
    $9^{-18} \approx 10^{-17}$; beyond, the direct difference cancels by at
    most a factor 3.6.
    """
    w = x / (2 + x)
    w2 = w * w
    series = horner(_ATANH_TAIL, w2)
    small = 2 * w * w2 * series + 2 * w2 / (1 + w)
    return jnp.where(x <= 1, small, jnp.log1p(x) - x / (1 + x))  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def dmass_dr(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""$dM/dr = 4\pi r^2 \rho = m x / (r_s (1+x)^2)$, for `radial_derivatives`."""
    x = r / p["r_s"]
    return p["m"] / p["r_s"] * x / (1 + x) ** 2


@ft.partial(jax.jit)
@radial_derivatives(mass_enclosed, dmass_dr)
def potential(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""Potential for the NFW model.

    .. math::

        \Phi(r) = -\frac{G m}{r_s} \frac{r_s}{r} \ln\left(1 + \frac{r}{r_s}\right)

    where :math:`m` is the characteristic mass and :math:`r_s` is the scale
    radius. It vanishes at infinity and tends to :math:`-G m / r_s` as
    :math:`r \to 0`. The function is evaluated for :math:`r > 0`, and at exactly
    :math:`r = 0` it returns NaN.

    """
    r_s = p["r_s"]
    x = r / r_s
    phi0 = -p["G"] * p["m"] / r_s
    return phi0 * jnp.log1p(x) / x  # type: ignore[no-any-return]


# Derived quantities, built from the kernels above.
dpotential_dr = radial_gradient(potential)
d2potential_dr2 = radial_second_derivative(dpotential_dr)
laplacian = poisson_laplacian(density)
gradient = spherical_gradient(dpotential_dr)
hessian = spherical_hessian(dpotential_dr, d2potential_dr2, laplacian)
