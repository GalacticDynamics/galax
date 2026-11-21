"""galax: Galactic Dynamix in Jax."""

__all__ = [
    # class
    "BurkertPotential",
    # functions
    "d2potential_dr2",
    "density",
    "dpotential_dr",
    "gradient",
    "hessian",
    "laplacian",
    "mass_enclosed",
    "mass_total",
    "potential",
    "rho0",
]

import functools as ft
from dataclasses import KW_ONLY

from typing import Any, Final, final

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
from galax.potential._src.utils import r_spherical, radial_derivatives

CONST_BURKER: Final = 3 * jnp.log(jnp.asarray(2.0)) - jnp.pi / 2


@final
class BurkertPotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Burkert potential.

    A spherical density profile with a flat core, introduced by Burkert
    [Burkert1995Burkert]_ as a phenomenological fit to the dark matter haloes of dwarf
    spiral galaxies. The density is finite and nearly flat inside the core radius and
    falls as :math:`r^{-3}` far out, so the enclosed mass grows logarithmically without
    bound and the total mass is infinite. The parameter ``r_s`` is the core radius and
    ``m`` is the mass inside ``r_s``.

    The density is

    .. math::

        \rho(r) = \frac{m}{\pi C}
            \frac{1}{(r + r_s)(r^2 + r_s^2)},
        \qquad C = 3 \ln 2 - \frac{\pi}{2},

    and with :math:`x = r / r_s` the enclosed mass is

    .. math::

        M(<r) = \frac{m}{C}
            \left[ 2 \ln(1 + x) + \ln(1 + x^2) - 2 \tan^{-1} x \right],

    which gives :math:`M(<r_s) = m`. The potential, with
    :math:`\Phi(r) \to 0` as :math:`r \to \infty`, is

    .. math::

        \Phi(r) = -\frac{G m}{C r_s}
            \left[ \pi - 2 \left(1 + \frac{1}{x}\right) \tan^{-1} x
            + 2 \left(1 + \frac{1}{x}\right) \ln(1 + x)
            - \left(1 - \frac{1}{x}\right) \ln(1 + x^2) \right],

    where :math:`G` is the gravitational constant. The central density is

    .. math::

        \rho_0 = \rho(0) = \frac{m}{\pi C r_s^3},

    the central potential is :math:`\Phi(0) = -\pi G m / (C r_s)`, and the
    circular velocity is :math:`v_c(r) = \sqrt{G M(<r) / r}`.

    Notes
    -----
    Burkert [Burkert1995Burkert]_ (Eq. 2) writes the profile as
    :math:`\rho_0 r_0^3 / [(r + r_0)(r^2 + r_0^2)]` with a central density
    :math:`\rho_0` and a scale radius :math:`r_0`. This class uses
    :math:`r_0 = r_s` and replaces :math:`\rho_0` by the mass
    :math:`m = \pi C \rho_0 r_s^3`, so that :math:`m` is the mass inside
    :math:`r_s`. The constructor
    :meth:`~galax.potential.BurkertPotential.from_central_density` takes
    :math:`\rho_0` directly.

    The central density follows from the density profile near :math:`r = 0`:
    :math:`\rho(r) \approx \rho_0 (1 - r / r_s)`, so the core is flat but the
    first correction is linear in :math:`r`. Because the mass diverges, the
    class has no finite total mass and no half-mass radius.

    References
    ----------
    .. [Burkert1995Burkert] Burkert, A. (1995). The Structure of Dark Matter Halos in
       Dwarf Galaxies. Astrophysical Journal Letters 447, L25.
       `ADS <https://ui.adsabs.harvard.edu/abs/1995ApJ...447L..25B>`__;
       `doi:10.1086/309560 <https://doi.org/10.1086/309560>`__;
       `arXiv:astro-ph/9504041 <https://arxiv.org/abs/astro-ph/9504041>`__.
    """

    m: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass",
        doc="Core mass of the potential (i.e. the mass within r_s).",
    )

    r_s: AbstractParameter = ParameterField(dimensions="length", doc="Scale radius")  # type: ignore[assignment]

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

        # Compute potential
        params = self.params(t)
        return potential(params, r)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BtFloatSz0:
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])

        params = self.params(t)
        return density(params, r)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _mass(self, xyz: gt.BBtQuSz3, /, t: gt.BtQuSz0 | gt.QuSz0) -> gt.BtFloatQuSz0:
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])

        params = self.params(t)
        return mass_enclosed(params, r)

    # -------------------------------------------------------------------

    def rho0(self, t: gt.BtQuSz0 | gt.QuSz0) -> gt.BtFloatQuSz0:
        r"""Central density of the potential.

        .. math::

            \rho_0 = \frac{m}{\pi (3 \ln 2 - \pi / 2) r_s^3}
        """
        return rho0(self.m(t), self.r_s(t))

    # -------------------------------------------------------------------
    # Constructors

    @classmethod
    def from_central_density(
        cls, rho_0: u.Quantity, r_s: u.Quantity, **kwargs: Any
    ) -> "BurkertPotential":
        r"""Create a Burkert potential from the central density.

        Parameters
        ----------
        rho_0 : :class:`~unxt.Quantity`[mass density]
            Central density.
        r_s : :class:`~unxt.Quantity`[length]
            Scale radius.

        Returns
        -------
        :class:`~galax.potential.BurkertPotential`
            Burkert potential.

        Examples
        --------
        >>> import unxt as u
        >>> import galax.potential as gp

        >>> rho_0 = u.Q(1e6, "Msun / kpc3")
        >>> r_s = u.Q(1, "kpc")
        >>> pot = gp.BurkertPotential.from_central_density(rho_0, r_s, units="galactic")
        >>> pot
        BurkertPotential(
            units=LTMAUnitSystem( length=Unit("kpc"), ...),
            constants=ImmutableMap({'G': ...}),
            m=ConstantParameter(...),
            r_s=ConstantParameter(...)
        )

        """
        m = jnp.pi * rho_0 * r_s**3 * CONST_BURKER
        return cls(m=m, r_s=r_s, **kwargs)


# ===================================================================


@ft.partial(jax.jit)
def rho0(m: gt.QuSz0, r_s: gt.QuSz0, /) -> gt.Sz0:
    r"""Central density of the potential.

    .. math::

        \rho_0 = \frac{m}{\pi C r_s^3}, \qquad C = 3 \ln 2 - \frac{\pi}{2}
    """
    return m / (CONST_BURKER * jnp.pi * r_s**3)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def density(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Density profile for the Burkert potential.

    .. math::

        \rho(r) = \frac{m}{\pi (3 \ln 2 - \pi / 2)}
            \frac{1}{(r + r_s)(r^2 + r_s^2)}

    where :math:`m` is the mass inside :math:`r_s` and :math:`r_s` is the core
    radius.
    """
    factor = p["m"] / (jnp.pi * CONST_BURKER)
    _result = factor / ((r + p["r_s"]) * (r**2 + p["r_s"] ** 2))
    return _result  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def mass_enclosed(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Enclosed mass profile for the Burkert potential.

    .. math::

        M(<r) = \frac{m}{3 \ln 2 - \pi / 2}
            \left[ 2 \ln(1 + x) + \ln(1 + x^2) - 2 \tan^{-1} x \right],
        \qquad x = r / r_s,

    where :math:`m = M(<r_s)` and :math:`r_s` is the core radius.
    """
    x = r / p["r_s"]
    return p["m"] / CONST_BURKER * _mass_bracket(x)  # type: ignore[no-any-return]


# Minimax fit, highest degree first, of the bracket below over 4x^3 on
# [0, 0.8]: mpmath.chebyfit at 60 digits, degree 20 (fit error 2e-17).
# `scripts/fit_burkert_mass_bracket.py` regenerates it and checks the bounds.
_BRACKET_FIT: Final = (
    -0.0006694096021792203,
    0.010040494048579117,
    -0.058353840712721566,
    0.1866881514834824,
    -0.37130800035137984,
    0.4746012984355624,
    -0.375570777120145,
    0.15940205688929185,
    -0.0338611455923545,
    0.035339918079129784,
    -0.009666784574964232,
    -0.08128476063349717,
    0.09057637015828068,
    4.068628654442871e-05,
    -3.6484796805761754e-06,
    -0.12499976905261787,
    0.142857133099678,
    2.524511286941352e-10,
    -3.460926234110154e-12,
    -0.24999999999998113,
    0.3333333333333333,
)


def _mass_bracket(x: gt.Sz0, /) -> gt.FloatSz0:
    r"""$2\ln(1+x) + \ln(1+x^2) - 2\tan^{-1}x$, to a few ulp for every $x \geq 0$.

    Directly, three $\sim x$ terms cancel to $\sim 4x^3/3$. It is the integral
    of $4s^2/((1+s)(1+s^2))$, so it is $4x^3 h(x)$ with $h$ analytic and
    $h(0) = 1/3$. Up to $x = 0.8$, $h$ is a degree-20 polynomial fit (2 ulp
    over the range); beyond, the direct sum cancels by less than a factor 10.
    """
    xc = jnp.where(x <= 0.8, x, 0.8)
    c = jnp.asarray(_BRACKET_FIT, dtype=xc.dtype)
    h = jnp.polyval(c, xc, unroll=len(_BRACKET_FIT))
    direct = 2 * jnp.log1p(x) + jnp.log1p(x**2) - 2 * jnp.atan(x)
    return jnp.where(x <= 0.8, 4 * xc**3 * h, direct)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def dmass_dr(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""$dM/dr = 4\pi r^2 \rho$, for `radial_derivatives`."""
    x = r / p["r_s"]
    dbracket = 4 * x**2 / ((1 + x) * (1 + x**2))
    return p["m"] / (CONST_BURKER * p["r_s"]) * dbracket  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def mass_total(p: gt.Params, /) -> gt.FloatSz0:
    r"""Total mass for the Burkert potential.

    .. math::

        M_{\rm tot} = \begin{cases} \infty & m > 0 \\
        0 & m \leq 0 \end{cases}

    The enclosed mass grows as :math:`\ln r` for large :math:`r`.
    """
    return jax.lax.select(p["m"] > 0, jnp.inf, 0.0)


@ft.partial(jax.jit)
@radial_derivatives(mass_enclosed, dmass_dr)
def potential(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Potential for the Burkert potential.

    With :math:`x = r / r_s` and :math:`C = 3 \ln 2 - \pi / 2`,

    .. math::

        \Phi(r) = -\frac{G m}{C r_s}
            \left[ \pi - 2 \left(1 + \frac{1}{x}\right) \tan^{-1} x
            + 2 \left(1 + \frac{1}{x}\right) \ln(1 + x)
            - \left(1 - \frac{1}{x}\right) \ln(1 + x^2) \right]

    where :math:`m = M(<r_s)` and :math:`r_s` is the core radius. It is
    evaluated regrouped as

    .. math::

        2 \tan^{-1}(1/x) + \ln\left(1 + \frac{2x}{1+x^2}\right)
        + \frac{2\ln(1+x) + \ln(1+x^2) - 2\tan^{-1}x}{x},

    every term positive, where the form above cancels both as :math:`x \to 0`
    and as :math:`x \to \infty`.
    """
    r_s = p["r_s"]
    x = r / r_s
    prefactor = p["G"] * p["m"] / (r_s * CONST_BURKER)
    # atan2(1, x) = atan(1/x) for x > 0 without forming 1/x, and the last term
    # -> 0 as x -> 0: x can underflow to 0 at the origin in float32 SI units.
    pos = x > 0
    x_safe = jnp.where(pos, x, 1.0)
    bracket_over_x = jnp.where(pos, _mass_bracket(x_safe) / x_safe, 0.0)
    return -prefactor * (  # type: ignore[no-any-return]
        2 * jnp.atan2(1.0, x) + jnp.log1p(2 * x / (1 + x**2)) + bracket_over_x
    )


# Derived quantities, built from the kernels above.
dpotential_dr = radial_gradient(potential)
d2potential_dr2 = radial_second_derivative(dpotential_dr)
laplacian = poisson_laplacian(density)
gradient = spherical_gradient(dpotential_dr)
hessian = spherical_hessian(dpotential_dr, d2potential_dr2, laplacian)
