"""Bar-typed potentials."""

__all__ = [
    # class
    "MonariEtAl2016BarPotential",
    # function
    "potential",
]

import functools as ft
from dataclasses import KW_ONLY
from typing import final

import jax

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax._custom_types as gt
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.utils._jax import vectorize_method


@final
class MonariEtAl2016BarPotential(AbstractSinglePotential):
    """Monari et al. (2016) Bar Potential.

    This is a generalization to 3D of the Dehnen 2000 bar potential.
    We take the defaults from Monari et al. (2016) paper.

    https://ui.adsabs.harvard.edu/abs/2016MNRAS.461.3835M/abstract

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.MonariEtAl2016BarPotential(
    ...     alpha=0.01,
    ...     R0=u.Quantity(8.0, "kpc"),
    ...     v0=u.Quantity(220.0, "km/s"),
    ...     Rb=u.Quantity(3.5, "kpc"),
    ...     phi_b=u.Quantity(25, "deg"),
    ...     Omega=u.Quantity(52.2, "km/(s kpc)"),
    ...     units="galactic",
    ... )
    >>> pot(u.Quantity([8.0, 0.0, 0.0], "kpc"), u.Quantity(0.0, "Gyr"))
    Quantity(Array(-0.00010847, dtype=float64), unit='kpc2 / Myr2')

    """

    _: KW_ONLY

    alpha: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.Quantity(0.01, ""),
        dimensions="dimensionless",
        doc="""The amplitude.

    the ratio between the bar's and axisymmetric contribution to the radial
    force, along the bar's long axis at (R,z) = (R0,0).
    """,
    )

    R0: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="The Galactocentric radius of the Sun."
    )

    v0: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="speed", doc="The circular velocity at R0."
    )

    Rb: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.Quantity(3.5, "kpc"),
        dimensions="length",
        doc="The length of the bar.",
    )

    phi_b: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.Quantity(25, "deg"), dimensions="angle", doc="Bar angle."
    )

    Omega: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.Quantity(52.2, "km/(s kpc)"),
        dimensions="frequency",
        doc="Bar pattern speed.",
    )

    @ft.partial(jax.jit)
    @vectorize_method(signature="(3),()->()")
    def _potential(self, xyz: gt.QuSz3 | gt.Sz3, t: gt.QuSz0 | gt.Sz0) -> gt.Sz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.ustrip(AllowValue, self.units["time"], t)
        # Compute parameters
        params = {
            "alpha": self.alpha(t, ustrip=self.units["dimensionless"]),
            "v0": self.v0(t, ustrip=self.units["speed"]),
            "R0": self.R0(t, ustrip=self.units["length"]),
            "Rb": self.Rb(t, ustrip=self.units["length"]),
            "phi_b": self.phi_b(t, ustrip=self.units["angle"]),
            "Omega": self.Omega(t, ustrip=self.units["frequency"]),
        }
        return potential(params, xyz, t)


@ft.partial(jax.jit)
def U_of_r(s: gt.Sz0, /) -> gt.Sz0:
    # M+2016 eq.3, modified to work on s=r/Rb
    def gtr_func(s: gt.Sz0) -> gt.Sz0:
        return -(s**-3)

    def less_func(s: gt.Sz0) -> gt.Sz0:
        return s**3 - 2.0

    pred = s >= 1
    return jax.lax.cond(pred, gtr_func, less_func, s)


# ===================================================================


@ft.partial(jax.jit)
def potential(p: gt.Params, xyz: gt.Sz3, t: gt.Sz0, /) -> gt.Sz0:
    r"""Specific potential energy."""
    R2 = xyz[0] ** 2 + xyz[1] ** 2
    r2 = R2 + xyz[2] ** 2

    prefactor = p["alpha"] * (p["v0"] ** 2 / 3) * (p["R0"] / p["Rb"]) ** 3
    u_of_r = U_of_r(jnp.sqrt(r2) / p["Rb"])
    phi = jnp.arctan2(xyz[1], xyz[0])
    gamma_b = 2 * (phi - p["phi_b"] - p["Omega"] * t)  # M+2016 eq.2

    return prefactor * u_of_r * (R2 / r2) * jnp.cos(gamma_b)  # M+2016 eq.1
