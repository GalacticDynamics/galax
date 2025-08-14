__all__ = ["ZhaoPotential"]

import functools as ft
from dataclasses import KW_ONLY
from typing import final

import equinox as eqx
import jax
import jax.scipy.special as jsp

import quaxed.numpy as jnp
import unxt as u
from xmmutablemap import ImmutableMap

import galax._custom_types as gt
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.utils import r_spherical


@final
class ZhaoPotential(AbstractSinglePotential):
    r"""Zhao (1996) double power-law potential.

    This model represents a double power law in the density, with an inner slope
    :math:`\gamma` and an outer slope :math:`\beta`, but with a third parameter
    :math:`\alpha` that controls the width of the transition region between the two
    power laws.

    This model has a finite total mass for :math:`\beta > 3`. The other power-law
    parameters should satisfy :math:`\alpha > 0` and :math:`0 \leq \gamma < 3`.

    This model also reduces to a number of well-known analytic forms for certain values
    of the parameters (reproduced from Table 1 of Zhao 1996):
    - :math:`(\alpha, \beta, \gamma) = (1, 4, 1)`: Hernquist model
    - :math:`(\alpha, \beta, \gamma) = (1, 4, 2)`: Jaffe model
    - :math:`(\alpha, \beta, \gamma) = (1/2, 5, 0)`: Plummer model
    - :math:`(\alpha, \beta, \gamma) = (1, 3, 1)`: NFW model
    - :math:`(\alpha, \beta, \gamma) = (1, 3, \gamma)`: Generalized NFW model
    """

    m: AbstractParameter = ParameterField(
        dimensions="mass",
        doc=(
            "Scale mass parameter. This is equivalent to the mass enclosed within the "
            "scale radius. When beta > 3, the model has finite mass, but when beta <= 3"
            " the total mass is infinite."
        ),
    )  # type: ignore[assignment]
    r_s: AbstractParameter = ParameterField(dimensions="length", doc="Scale radius.")  # type: ignore[assignment]

    alpha: AbstractParameter = ParameterField(
        dimensions="dimensionless", doc="Transition width (alpha > 0)."
    )  # type: ignore[assignment]
    beta: AbstractParameter = ParameterField(
        dimensions="dimensionless", doc="Outer slope (finite mass when beta > 3)."
    )  # type: ignore[assignment]
    gamma: AbstractParameter = ParameterField(
        dimensions="dimensionless", doc="Inner slope (0 <= gamma < 3)."
    )  # type: ignore[assignment]

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        r = r_spherical(xyz, self.units["length"])
        t = u.Quantity.from_(t, self.units["time"])

        ulen = self.units["length"]
        umass = self.units["mass"]
        udim = self.units["dimensionless"]
        p = {
            "G": self.constants["G"].value,
            "m": self.m(t, ustrip=umass),
            "r_s": self.r_s(t, ustrip=ulen),
            "alpha": self.alpha(t, ustrip=udim),
            "beta": self.beta(t, ustrip=udim),
            "gamma": self.gamma(t, ustrip=udim),
        }
        return potential(p, r)

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BtFloatSz0:
        r = r_spherical(xyz, self.units["length"])
        t = u.Quantity.from_(t, self.units["time"])

        ulen = self.units["length"]
        umass = self.units["mass"]
        udim = self.units["dimensionless"]
        p = {
            "m": self.m(t, ustrip=umass),
            "r_s": self.r_s(t, ustrip=ulen),
            "alpha": self.alpha(t, ustrip=udim),
            "beta": self.beta(t, ustrip=udim),
            "gamma": self.gamma(t, ustrip=udim),
        }
        return density(p, r)



@ft.partial(jax.jit)
def _total_mass_factor(p: gt.Params, r_ref: gt.Sz0) -> gt.FloatSz0:
    """Compute the total mass factor for the Zhao profile."""
    c0, _, q0 = _cpq(p["alpha"], p["beta"], p["gamma"])
    x = r_ref / p["r_s"]
    chi = x ** (1.0 / p["alpha"]) / (1.0 + x ** (1.0 / p["alpha"]))
    # chi = jnp.power(x, 1.0 / p["alpha"]) / (1.0 + jnp.power(x, 1.0 / p["alpha"]))
    return jsp.betainc(c0 - q0, q0, chi)


@ft.partial(jax.jit)
def _rho0(p: gt.Params, r_ref: gt.Sz0 | None = None) -> gt.FloatSz0:
    """Compute the normalization density for the Zhao profile.

    This computes the normalization constant rho_0 (called C in Zhao 1996) for the Zhao
    density profile. The normalization is set that the mass parameter is the mass
    enclosed within ``r_ref``. If no r_ref is specified to this function (as happens in
    the default initializer for the ``ZhaoPotential``), it is set to the scale radius,
    so the mass parameter is interpreted to be the mass enclosed within the scale
    radius.

    This implementation uses the hyp2f1 hypergeometric function instead of the
    incomplete beta function because jax (and scipy) only provide the *regularized*
    version of the incomplete beta function. This means that it blows up when b <= 0 in
    B(a, b, z) because the complete beta function B(a, b) is undefined when b <= 0. The
    hyp2f1 function is defined for all values of a, b, and z, so it can handle the case
    where b <= 0.
    """
    r_ref = r_ref if r_ref is not None else p["r_s"]
    chi_norm = _r_to_u_chi(p, r_ref)[1]
    a = p["alpha"] * (3.0 - p["gamma"])
    b = p["alpha"] * (p["beta"] - 3.0)
    denom = (chi_norm**a / a) * jsp.hyp2f1(a, 1.0 - b, a + 1.0, chi_norm)
    return p["m"] / (4.0 * jnp.pi * p["alpha"] * denom)


@ft.partial(jax.jit)
def _cpq(a: gt.Sz0, b: gt.Sz0, g: gt.Sz0) -> tuple[gt.Sz0, gt.Sz0, gt.Sz0]:
    """Constants defined in appendix of Zhao (1996)."""
    c0 = a * (b - g)
    p0 = a * (2.0 - g)
    q0 = a * (b - 3.0)
    return c0, p0, q0


@ft.partial(jax.jit)
def _r_to_u_chi(p: gt.Params, r: gt.Sz0) -> gt.FloatSz0:
    r"""Convert radius to u and chi variables defined below.

    .. math::

        u = r / r_s
        chi = \frac{u^{1/\alpha}}{1 + u^{1/\alpha}}

    """
    uu = r / p["r_s"]
    return uu, uu ** (1.0 / p["alpha"]) / (1.0 + uu ** (1.0 / p["alpha"]))


@ft.partial(jax.jit)
def density(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    """Spherical density profile for double power-law Zhao model."""
    uu = r / p["r_s"]
    alpha, beta, gamma = p["alpha"], p["beta"], p["gamma"]
    rho0 = _rho0(p)

    b = (beta - gamma) * alpha
    return rho0 / (p["r_s"] ** 3) / uu**gamma / (1.0 + uu ** (1.0 / alpha)) ** b


@ft.partial(jax.jit)
def mass_enclosed(p: gt.Params, r: gt.Sz0) -> gt.Sz0:
    a, b, g = p["alpha"], p["beta"], p["gamma"]
    _, chi = _r_to_u_chi(p, r)
    rho0 = _rho0(p)
    c0, _, q0 = _cpq(a, b, g)
    return (
        4.0
        * jnp.pi
        * rho0
        * a
        * (jsp.beta(c0 - q0, q0) * jsp.betainc(c0 - q0, q0, chi))
    )


@ft.partial(jax.jit)
def potential(p: gt.Params, r: gt.Sz0, /) -> gt.Sz0:
    r"""Spherical potential for double power-law Zhao model.

    See Eq. 6 and 7 in Zhao (1996).

    This function uses the variable z for what Zhao called :math:`\chi`.
    """
    a, b, g = p["alpha"], p["beta"], p["gamma"]

    uu, chi = _r_to_u_chi(p, r)

    # Special case the Jaffe potential, where there is an "inf - inf" below
    is_jaffe = (a == 1.0) & (b == 4.0) & (g == 2.0)

    def Phi_jaffe() -> gt.Sz0:
        # Note: the extra factor of 2 is because m is mass enclosed in r_s, not total
        return -p["G"] * 2 * p["m"] / p["r_s"] * jnp.log1p(1.0 / uu)

    rho0 = _rho0(p)
    c0, p0, _ = _cpq(a, b, g)

    # Left term in Eq. 7
    term_l = mass_enclosed(p, r)

    # Right term in Eq. 7
    eps = jnp.sqrt(jnp.finfo(r.dtype).eps)
    p0_safe = jnp.where(p0 <= 0, eps, p0)
    logB = jsp.betaln(p0_safe, c0 - p0)
    log1mI = jnp.log1p(-jsp.betainc(p0_safe, c0 - p0, chi))
    term_r = 4.0 * jnp.pi * rho0 * a / p["r_s"] * jnp.exp(logB + log1mI)

    def Phi_general() -> gt.Sz0:
        return -p["G"] * (term_l / r + term_r)

    return jax.lax.cond(is_jaffe, Phi_jaffe, Phi_general)
