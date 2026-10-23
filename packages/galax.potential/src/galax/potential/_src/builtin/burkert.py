"""galax: Galactic Dynamix in Jax."""

__all__ = (
    # class
    "BurkertPotential",
    # functions
    "rho0",
    "density",
    "potential",
    "mass_enclosed",
    "mass_total",
)

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
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.symmetry import Symmetry
from galax.potential._src.utils import r_spherical, radial_derivatives

CONST_BURKER: Final = 3 * jnp.log(jnp.asarray(2.0)) - jnp.pi / 2


@final
class BurkertPotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Burkert Potential.

    https://ui.adsabs.harvard.edu/abs/1995ApJ...447L..25B/abstract,
    https://iopscience.iop.org/article/10.1086/309140/fulltext/50172.text.html.

    The mass parameter sets the core mass:

    .. math::

        M_0 = \\pi \rho_0 r_s^3 (3 \\log(2) - \\pi / 2)

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

        $$ m0 = \pi \rho_0 r_s^3 (3 \log(2) - \pi / 2) $$
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
    r"""Central density of the potential."""
    return m / (CONST_BURKER * jnp.pi * r_s**3)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def density(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Density profile for the Burkert potential.

    $$ \rho(r) = \frac{m}{\pi (3 \log(2) - \pi / 2)}
    \frac{1}{(r + r_s)(r^2 + r_s^2)} $$

    where $m$ is the characteristic mass and $r_s$ is the scale radius.
    """
    factor = p["m"] / (jnp.pi * CONST_BURKER)
    _result = factor / ((r + p["r_s"]) * (r**2 + p["r_s"] ** 2))
    return _result  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def mass_enclosed(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Enclosed mass profile for the Burkert potential.

    $$ M(r) = \frac{m}{3 \log(2) - \pi / 2}
        \left[ 2 \log(1 + r/r_s) + \log(1 + (r/r_s)^2)
        - 2 \tan^{-1}(r/r_s) \right]
    $$

    where $m$ is the characteristic mass and $r_s$ is the scale radius.
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

    $$ M_{\rm tot} = \begin{cases} \infty & m > 0 \\
    0 & m \leq 0 \end{cases} $$

    """
    return jax.lax.select(p["m"] > 0, jnp.inf, 0.0)


@ft.partial(jax.jit)
@radial_derivatives(mass_enclosed, dmass_dr)
def potential(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Potential for the Burkert potential.

    $$
    \Phi(r) = -\frac{G m}{r_s (3 \log(2) - \pi / 2)}
        \left[ \pi - 2 (1 + r/r_s) \tan^{-1}(r/r_s)
        + 2 (1 + r/r_s) \log(1 + r/r_s)
        - (1 - r/r_s) \log(1 + (r/r_s)^2)
        \right]
    $$

    where $m$ is the characteristic mass and $r_s$ is the scale radius.
    Evaluated regrouped as

    $$ 2 \tan^{-1}(1/x) + \ln\left(1 + \frac{2x}{1+x^2}\right)
       + \frac{2\ln(1+x) + \ln(1+x^2) - 2\tan^{-1}x}{x}, $$

    every term positive, where the form above cancels both as $x \to 0$ and
    as $x \to \infty$.
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
