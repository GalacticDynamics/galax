"""galax: Galactic Dynamix in Jax."""

__all__ = [
    # class
    "PowerLawCutoffPotential",
    # functions
    "potential",
]

import functools as ft
from dataclasses import KW_ONLY
from typing import final

import equinox as eqx
import jax

import quaxed.scipy.special as jsp
import unxt as u
from xmmutablemap import ImmutableMap

import galax._custom_types as gt
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.utils import r_spherical


@ft.partial(jax.jit)
def _safe_gamma_inc(a: gt.SzN, x: gt.SzN) -> gt.SzN:
    return jsp.gammainc(a, x) * jsp.gamma(a)


@final
class PowerLawCutoffPotential(AbstractSinglePotential):
    r"""A spherical power-law density profile with an exponential cutoff.

    .. math::

        \rho(r) = \frac{G M}{2\pi \Gamma((3-\alpha)/2) r_c^3} \left(\frac{r_c}{r}\right)^\alpha \exp{-(r / r_c)^2}

    Parameters
    ----------
    m_tot : :class:`~unxt.Quantity`[mass]
        Total mass.
    alpha : :class:`~unxt.Quantity`[dimensionless]
        Power law index. Must satisfy: ``0 <= alpha < 3``.
    r_c : :class:`~unxt.Quantity`[length]
        Cutoff radius.
    """  # noqa: E501

    m_tot: AbstractParameter = ParameterField(dimensions="mass", doc="Total mass.")  # type: ignore[assignment]
    """Total mass of the potential."""

    alpha: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        doc="Power law index. Must satisfy: ``0 <= alpha < 3``",
    )

    r_c: AbstractParameter = ParameterField(dimensions="length", doc="Cutoff radius.")  # type: ignore[assignment]

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQuSz3, t: gt.BBtQuSz0, /) -> gt.BtSz0:
        # Parse inputs
        ul = self.units["length"]
        r = r_spherical(xyz, ul)
        t = u.Quantity.from_(t, self.units["time"])

        params = {
            "G": self.constants["G"].value,
            "m_tot": self.m_tot(t, ustrip=self.units["mass"]),
            "alpha": self.alpha(t, ustrip=self.units["dimensionless"]),
            "r_c": self.r_c(t, ustrip=ul),
        }
        return potential(params, r)


# ===================================================================


@ft.partial(jax.jit)
def potential(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Potential for the power-law cutoff density profile.

    $$ \Phi(r) = -\frac{G M}{2\pi \Gamma\left(\frac{3 - \alpha}{2}\right) r_c^3}
    \left(\frac{r_c}{r}\right)^\alpha
    \exp\left[-\left(\frac{r}{r_c}\right)^2\right] $$
    """
    a = p["alpha"] / 2
    s2 = (r / p["r_c"]) ** 2
    GM = p["G"] * p["m_tot"]

    return GM * (
        (a - 1.5) * _safe_gamma_inc(1.5 - a, s2) / (r * jsp.gamma(2.5 - a))
        + _safe_gamma_inc(1 - a, s2) / (p["r_c"] * jsp.gamma(1.5 - a))
    )
