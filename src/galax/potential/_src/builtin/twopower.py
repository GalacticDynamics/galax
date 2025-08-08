__all__ = ["TwoPowerPotential", "potential"]

import functools as ft
from dataclasses import KW_ONLY
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

import unxt as u
from xmmutablemap import ImmutableMap

import galax._custom_types as gt
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.utils import r_spherical


@final
class TwoPowerPotential(AbstractSinglePotential):
    r"""Spherical double power-law potential.

    .. math::

        \rho(r) = \frac{\rho_0}{(r/r_s)^\alpha (1 + r/r_s)^{\beta - \alpha}}
        \rho_0 = \frac{m_s}{4\pi r_s^3}

    This potential is only valid when :math:`\alpha < 2` and :math:`\beta > 3`.

    Parameters
    ----------
    m_tot : :class:`~unxt.Quantity`[mass]
        Total mass.
    r_s : :class:`~unxt.Quantity`[length]
        Scale radius.
    alpha : :class:`~unxt.Quantity`[dimensionless]
        Inner slope. Must satisfy ``alpha < 2``.
    beta : :class:`~unxt.Quantity`[dimensionless]
        Outer slope. Must satisfy ``beta > 3`` for finite mass.
    """

    m_tot: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass", doc="Total mass."
    )
    r_s: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="Scale radius."
    )
    alpha: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless", doc="Inner slope."
    )
    beta: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless", doc="Outer slope."
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQuSz3, t: gt.BBtQuSz0, /) -> gt.BtSz0:
        r = r_spherical(xyz, self.units["length"])
        t = u.Quantity.from_(t, self.units["time"])

        params = {
            "G": self.constants["G"].value,
            "m_tot": self.m_tot(t, ustrip=self.units["mass"]),
            "r_s": self.r_s(t, ustrip=self.units["length"]),
            "alpha": self.alpha(t, ustrip=self.units["dimensionless"]),
            "beta": self.beta(t, ustrip=self.units["dimensionless"]),
        }
        return potential(params, r)

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQuSz3, t: gt.BBtQuSz0, /) -> gt.BtFloatSz0:
        r = r_spherical(xyz, self.units["length"])
        t = u.Quantity.from_(t, self.units["time"])

        params = {
            "G": self.constants["G"].value,
            "m_tot": self.m_tot(t, ustrip=self.units["mass"]),
            "r_s": self.r_s(t, ustrip=self.units["length"]),
            "alpha": self.alpha(t, ustrip=self.units["dimensionless"]),
            "beta": self.beta(t, ustrip=self.units["dimensionless"]),
        }
        return density(params, r)


@ft.partial(jax.jit)
def density(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""Density for the two-power density profile.

    .. math::

        \rho(r) = \frac{\rho_0}{(r/r_s)^\alpha (1 + r/r_s)^{\beta - \alpha}}
        \rho_0 = \frac{m}{4\pi r_s^3}
            \frac{\Gamma(\beta - \alpha)}{\Gamma(3 - \alpha) \Gamma(\beta - 3)}
    """
    x = r / p["r_s"]
    a = p["alpha"]
    b = p["beta"]

    rho0 = (
        p["m_tot"]
        / (4 * jnp.pi * p["r_s"] ** 3)
        * jsp.gamma(b - a)
        / (jsp.gamma(3 - a) * jsp.gamma(b - 3))
    )
    return rho0 / (x**a * (1 + x) ** (b - a))


@ft.partial(jax.jit)
def potential(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Gravitational potential for the two-power density profile.

    This is valid for beta > 3.
    """
    x = r / p["r_s"]
    a = p["alpha"]
    b = p["beta"]

    rho0 = (
        p["m_tot"]
        / (4 * jnp.pi * p["r_s"] ** 3)
        * jsp.gamma(b - a)
        / (jsp.gamma(3 - a) * jsp.gamma(b - 3))
    )

    # inner integral
    u = x / (1 + x)
    a1, b1 = 3 - a, b - 3
    I_in = p["r_s"] ** 3 / r * jsp.betainc(a1, b1, u) * jsp.beta(a1, b1)

    # outer integral
    t = 1 / (1 + x)
    a2, b2 = b - 2, 2 - a
    I_out = p["r_s"] ** 2 * jsp.betainc(a2, b2, t) * jsp.beta(a2, b2)

    return -4 * jnp.pi * p["G"] * rho0 * (I_in + I_out)
