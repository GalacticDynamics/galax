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
            "Mass parameter. When beta > 3, this is the total mass. When beta <= 3, "
            "this is a scale mass."
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
def _rho0(m: gt.Sz0, alpha: gt.Sz0, beta: gt.Sz0, gamma: gt.Sz0, /) -> gt.Sz0:
    r"""Scale density.

    Define the scale density (called "C" in the Zhao paper) such that the integral:

        \int_0^\infty 4\pi r^2 \rho(r) \, dr = m

    for models with finite mass.

    See Eq. 44 in Zhao (1996).
    """
    denom = alpha * jsp.beta(alpha * (3.0 - gamma), alpha * (beta - 3.0))
    return m / (4.0 * jnp.pi * denom)


@ft.partial(jax.jit)
def density(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    """Spherical density profile for double power-law Zhao model."""
    x = r / p["r_s"]
    alpha, beta, gamma = p["alpha"], p["beta"], p["gamma"]
    rho0 = _rho0(p["m"], alpha, beta, gamma)

    b = (beta - gamma) * alpha
    return (rho0 / (p["r_s"] ** 3)) / x**gamma / (1.0 + x ** (1.0 / alpha)) ** b


@ft.partial(jax.jit)
def potential(p: gt.Params, r: gt.Sz0, /) -> gt.Sz0:
    r"""Spherical potential for double power-law Zhao model.

    See Eq. 6 and 7 in Zhao (1996).

    This function uses the variable z for what Zhao called :math:`\chi`.
    """
    x = r / p["r_s"]
    alpha, beta, gamma = p["alpha"], p["beta"], p["gamma"]

    # What Zhao calls "chi":
    z = x ** (1.0 / alpha) / (1.0 + x ** (1.0 / alpha))

    rho0 = _rho0(p["m"], alpha, beta, gamma)

    # Constants defined in appendix of Zhao (1996)
    p0 = alpha * (2.0 - gamma)
    q0 = alpha * (beta - 3.0)
    c0 = alpha * (beta - gamma)

    # Left term in Eq. 7
    term_l = jsp.beta(c0 - q0, q0) * jsp.betainc(c0 - q0, q0, z)

    # Right term in Eq. 7
    # This uses a trick (from chatgpt) for avoiding the pole as p0 -> 0 (gamma -> 2)
    #   Trick: compute as exp(betaln + log(1 - I_z))
    eps = jnp.sqrt(jnp.finfo(r.dtype).eps)
    p0_safe = jnp.where(p0 <= 0, eps, p0)
    logB = jsp.betaln(p0_safe, c0 - p0)
    log1mI = jnp.log1p(-jsp.betainc(p0_safe, c0 - p0, z))
    term_r = jnp.exp(logB + log1mI)

    return -4.0 * jnp.pi * p["G"] * rho0 * alpha * (term_l / r + term_r / p["r_s"])
