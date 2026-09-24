"""Bar-typed potentials."""

__all__ = [
    # class
    "LongMuraliBarPotential",
    # function
    "potential",
    "density",
]

import functools as ft

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
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField


@final
class LongMuraliBarPotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Long & Murali Bar Potential.

    Long, K., & Murali, C. 1992, ApJ, 397, 44.
    https://ui.adsabs.harvard.edu/abs/1992ApJ...397...44L

    A simple, triaxial model for a galaxy bar. This is a softened “needle”
    density distribution with an analytic potential form. See Long & Murali
    (1992) for details.

    The corresponding density (:func:`density`) is obtained by applying
    Poisson's equation directly to the potential formula below; because of
    the finite bar length, this expression is algebraically messy (it was
    derived with a computer algebra system rather than transcribed from a
    textbook), but it is exact and verified against the finite-difference /
    autodiff Hessian of the potential.

    """

    m_tot: AbstractParameter = ParameterField(dimensions="mass", doc="The total mass.")  # type: ignore[assignment]

    a: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="Half-length defining the semi-major axis"
    )
    b: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="Thickness softening length"
    )
    c: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="Vertical softening length"
    )

    alpha: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="angle", doc="Position angle of the bar in the x-y plane."
    )

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Q.from_(t, self.units["time"])

        ul = self.units["length"]
        params = {
            "G": self.constants["G"].value,
            "m_tot": self.m_tot(t, ustrip=self.units["mass"]),
            "a": self.a(t, ustrip=ul),
            "b": self.b(t, ustrip=ul),
            "c": self.c(t, ustrip=ul),
            "alpha": self.alpha(t, ustrip=self.units["angle"]),
        }
        return potential(params, xyz)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Q.from_(t, self.units["time"])

        ul = self.units["length"]
        params = {
            "G": self.constants["G"].value,
            "m_tot": self.m_tot(t, ustrip=self.units["mass"]),
            "a": self.a(t, ustrip=ul),
            "b": self.b(t, ustrip=ul),
            "c": self.c(t, ustrip=ul),
            "alpha": self.alpha(t, ustrip=self.units["angle"]),
        }
        return density(params, xyz)  # type: ignore[no-any-return]


# ===================================================================


@ft.partial(jax.jit)
def potential(p: gt.Params, xyz: gt.Sz3, /) -> gt.Sz0:
    r"""Specific potential energy."""
    alpha = p["alpha"]
    a, b, c = p["a"], p["b"], p["c"]

    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    xp = x * jnp.cos(alpha) + y * jnp.sin(alpha)
    yp = -x * jnp.sin(alpha) + y * jnp.cos(alpha)

    yz2 = yp**2 + (b + jnp.sqrt(c**2 + z**2)) ** 2
    T_plus = jnp.sqrt((a + xp) ** 2 + yz2)
    T_minus = jnp.sqrt((a - xp) ** 2 + yz2)

    GM_R = p["G"] * p["m_tot"] / (2.0 * a)
    _result = GM_R * jnp.log((xp - a + T_minus) / (xp + a + T_plus))
    return _result  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def density(p: gt.Params, xyz: gt.Sz3, /) -> gt.Sz0:
    r"""Density for the Long & Murali (1992) bar potential.

    Obtained by applying Poisson's equation, :math:`\nabla^2\Phi = 4\pi G
    \rho`, to :func:`potential` with a computer algebra system (the
    intermediate quantities below are its common sub-expressions, not
    individually meaningful); verified numerically against the
    finite-difference Hessian of :func:`potential`.
    """
    alpha = p["alpha"]
    a, b, c, G, m_tot = p["a"], p["b"], p["c"], p["G"], p["m_tot"]

    x0, y0, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    x = x0 * jnp.cos(alpha) + y0 * jnp.sin(alpha)
    y = -x0 * jnp.sin(alpha) + y0 * jnp.cos(alpha)

    e0 = a - x
    e1 = e0**2
    e2 = y**2
    e3 = z**2
    e4 = c**2 + e3
    e5 = jnp.sqrt(e4)
    e6 = b + e5
    e7 = e6**2
    e8 = e2 + e7
    e9 = e1 + e8
    e10 = jnp.sqrt(e9)
    e11 = -a + x + e10
    e12 = 1 / e11
    e13 = e9 ** (-1.5)
    e14 = 1 / e10
    e15 = a + x
    e16 = e15**2
    e17 = e16 + e8
    e18 = jnp.sqrt(e17)
    e19 = 1 / e18
    e20 = e15 * e19 + 1
    e21 = e15 + e18
    e22 = 1 / e21
    e23 = e11 * e22
    e24 = e0 * e14 - 1
    e25 = e20 * e23 + e24
    e26 = 1 / e4
    e27 = 1 / e5
    e28 = e21**-2.0
    e29 = e19 * e23
    e30 = -e14 + e29
    e31 = e19 * e22
    e32 = e26 * e3
    e33 = e32 * e7
    e34 = e4 ** (-1.5)
    e35 = e17 ** (-1.5)
    e36 = 1 / e17
    e37 = 2 * e31

    lap = (
        0.5
        * G
        * m_tot
        * e12
        * (
            e11 * e19 * e22 * e3 * e34 * e6
            + e11 * e19 * e22 * (e16 * e36 - 1)
            + e11 * e2 * e22 * e35
            + 2 * e11 * e2 * e28 * e36
            + 2 * e11 * e20**2 * e28
            + e11 * e22 * e26 * e3 * e35 * e7
            + 2 * e11 * e26 * e28 * e3 * e36 * e7
            + e12 * e14 * e2 * e30
            + e12 * e14 * e26 * e3 * e30 * e7
            - e12 * e24 * e25
            - e13 * e2
            - e13 * e33
            - e14 * e2 * e37
            + e14 * e26 * e3
            + e14 * e27 * e6
            - e14 * e3 * e34 * e6
            - e14 * e32 * e37 * e7
            - e14 * (e1 / e9 - 1)
            - e2 * e30 * e31
            + 2 * e20 * e22 * e24
            - e20 * e22 * e25
            - e27 * e29 * e6
            - e29 * e32
            - e30 * e31 * e33
            - e30
        )
        / a
    )
    return lap / (4 * jnp.pi * G)  # type: ignore[no-any-return]
