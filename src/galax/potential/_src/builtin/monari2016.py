"""Bar-typed potentials."""

__all__ = [
    # class
    "MonariEtAl2016BarPotential",
    # function
    "potential",
    "density",
]

import functools as ft
from dataclasses import KW_ONLY

from typing import final

import jax

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax.potential.custom_types as gt
from galax.potential._src.base_single import (
    AbstractSinglePotential,
    LaplacianFromDensityMixin,
)
from galax.potential._src.jax import vectorize_method
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField


@final
class MonariEtAl2016BarPotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Monari et al. (2016) Bar Potential.

    This is a generalization to 3D of the Dehnen (2000) bar potential
    (Dehnen, W. 2000, AJ, 119, 800,
    https://ui.adsabs.harvard.edu/abs/2000AJ....119..800D), replacing
    Dehnen's cylindrical radius with the spherical radius. We take the
    defaults from Monari et al. (2016).

    https://ui.adsabs.harvard.edu/abs/2016MNRAS.461.3835M/abstract

    The density, derived from the potential below via Poisson's equation
    (matching Dehnen's construction: a genuine mass quadrupole confined to
    :math:`r < R_b`, and exactly zero outside):

    $$
    \rho(r,\theta,\phi) = \frac{\text{prefactor}}{4\pi G} \,
        L(r) \, \sin^2\theta \, \cos(2(\phi-\phi_b-\Omega t)),
        \qquad
        L(r) = \begin{cases}
            \dfrac{12}{r^2} + \dfrac{6r}{R_b^3}, & r < R_b \\[4pt]
            0, & r \geq R_b
        \end{cases}
    $$

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.MonariEtAl2016BarPotential(
    ...     alpha=0.01,
    ...     R0=u.Q(8.0, "kpc"),
    ...     v0=u.Q(220.0, "km/s"),
    ...     Rb=u.Q(3.5, "kpc"),
    ...     phi_b=u.Q(25, "deg"),
    ...     Omega=u.Q(52.2, "km/(s kpc)"),
    ...     units="galactic",
    ... )
    >>> pot(u.Q([8.0, 0.0, 0.0], "kpc"), u.Q(0.0, "Gyr"))
    Q(-0.00010847, 'kpc2 / Myr2')

    """

    _: KW_ONLY

    alpha: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.Q(0.01, ""),
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
        default=u.Q(3.5, "kpc"),
        dimensions="length",
        doc="The length of the bar.",
    )

    phi_b: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.Q(25, "deg"), dimensions="angle", doc="Bar angle."
    )

    Omega: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.Q(52.2, "km/(s kpc)"),
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
        return potential(params, xyz, t)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    @vectorize_method(signature="(3),()->()")
    def _density(self, xyz: gt.QuSz3 | gt.Sz3, t: gt.QuSz0 | gt.Sz0) -> gt.Sz0:
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
            "G": self.constants["G"].value,
        }
        return density(params, xyz, t)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def U_of_r(s: gt.Sz0, /) -> gt.Sz0:
    # M+2016 eq.3, modified to work on s=r/Rb
    def gtr_func(s: gt.Sz0) -> gt.Sz0:
        return -(s**-3)

    def less_func(s: gt.Sz0) -> gt.Sz0:
        return s**3 - 2.0

    pred = s >= 1
    return jax.lax.cond(pred, gtr_func, less_func, s)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def L_of_r(r: gt.Sz0, r_b: gt.Sz0, /) -> gt.Sz0:
    r"""Radial part of the density, :math:`\nabla^2_r[U(r/R_b)]` for l=2.

    Exactly zero outside the bar (:math:`r \geq R_b`), since ``U`` there is
    :math:`-(r/R_b)^{-3}`, a source-free (harmonic) l=2 radial solution.
    """
    return jnp.where(r < r_b, 12 / r**2 + 6 * r / r_b**3, 0.0)  # type: ignore[no-any-return]


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

    energy = prefactor * u_of_r * (R2 / r2) * jnp.cos(gamma_b)  # M+2016 eq.1
    return energy  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def density(p: gt.Params, xyz: gt.Sz3, t: gt.Sz0, /) -> gt.Sz0:
    r"""Density derived from the potential above via Poisson's equation."""
    R2 = xyz[0] ** 2 + xyz[1] ** 2
    r2 = R2 + xyz[2] ** 2
    r = jnp.sqrt(r2)

    prefactor = p["alpha"] * (p["v0"] ** 2 / 3) * (p["R0"] / p["Rb"]) ** 3
    phi = jnp.arctan2(xyz[1], xyz[0])
    gamma_b = 2 * (phi - p["phi_b"] - p["Omega"] * t)  # M+2016 eq.2

    lap_ang = prefactor * L_of_r(r, p["Rb"]) * (R2 / r2) * jnp.cos(gamma_b)
    return lap_ang / (4 * jnp.pi * p["G"])  # type: ignore[no-any-return]
