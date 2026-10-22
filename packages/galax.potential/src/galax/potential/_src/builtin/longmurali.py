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
from galax.potential._src.utils import with_r_derivative


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

        params = self.params(t)
        return potential(params, xyz)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)

        params = self.params(t)
        return density(params, xyz)  # type: ignore[no-any-return]


# ===================================================================


def _bar_terms(p: gt.Params, xyz: gt.Sz3, /) -> tuple[gt.Sz0, ...]:
    r"""Bar-frame $x'$, $y'$, $\zeta = \sqrt{c^2+z^2}$, $\eta$ and $T_\pm$.

    $\eta^2 = y'^2 + (b + \zeta)^2$ and $T_\pm = \sqrt{(a \pm |x'|)^2 + \eta^2}$;
    everything below is even in $x'$, so it is written for $|x'|$.
    """
    alpha, a, b, c = p["alpha"], p["a"], p["b"], p["c"]
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    xp = x * jnp.cos(alpha) + y * jnp.sin(alpha)
    yp = -x * jnp.sin(alpha) + y * jnp.cos(alpha)
    zeta = jnp.sqrt(c**2 + z**2)
    eta2 = yp**2 + (b + zeta) ** 2
    ax = jnp.abs(xp)
    T_plus = jnp.sqrt((a + ax) ** 2 + eta2)
    T_minus = jnp.sqrt((a - ax) ** 2 + eta2)
    return xp, yp, zeta, eta2, ax, T_plus, T_minus


def _gradient_impl(p: gt.Params, xyz: gt.Sz3, /) -> gt.Sz3:
    r"""Gradient, in forms that do not cancel.

    With $\Phi = -\frac{GM}{2a}\left[\sinh^{-1}\frac{a+x'}{\eta}
    + \sinh^{-1}\frac{a-x'}{\eta}\right]$ (the log form, rationalized),

    .. math::

        \frac{\partial\Phi}{\partial x'}
            = \frac{2 G M x'}{(T_- + T_+) T_+ T_-}, \qquad
        \frac{1}{\eta}\frac{\partial\Phi}{\partial\eta}
            = \frac{G M}{2\eta^2}
              \frac{(T_- + T_+)^2 - 4x'^2}{(T_- + T_+) T_+ T_-},

    where autodiff of the log takes differences of $\sim 1/r$ terms that
    cancel to $\sim a/r^2$ far out, and to $\sim |x|/a^2$ at the centre.
    $(T_-+T_+)^2 - 4x'^2 = (T_-+T_+-2|x'|)(T_-+T_++2|x'|)$, with
    $T_- + T_+ - 2|x'| = \eta^2/(T_+ + |x'| + a) + (T_- - |x'| + a)$ and the
    last term rationalized like the potential's numerator.
    """
    alpha, a, b = p["alpha"], p["a"], p["b"]
    GM = p["G"] * p["m_tot"]
    xp, yp, zeta, eta2, ax, T_plus, T_minus = _bar_terms(p, xyz)
    T_sum, T_prod = T_minus + T_plus, T_plus * T_minus

    dphi_dxp = 2 * GM * xp / (T_sum * T_prod)
    # T_- - |x'| + a; T_- > |a - |x'|| since eta >= b + c > 0, so no 0/0.
    minus_part = jnp.where(ax >= a, eta2 / (T_minus + ax - a), T_minus + a - ax)
    T_sum_m2x = eta2 / (T_plus + ax + a) + minus_part
    # (1/eta) dPhi/deta
    k = GM / (2 * eta2) * T_sum_m2x * (T_sum + 2 * ax) / (T_sum * T_prod)

    ca, sa = jnp.cos(alpha), jnp.sin(alpha)
    z = xyz[..., 2]
    # z / zeta, 0 on the mid-plane of a razor-thin (c = 0) bar, where it is 0/0
    on_plane = zeta == 0
    z_zeta = jnp.where(on_plane, 0.0, z / jnp.where(on_plane, 1.0, zeta))
    return jnp.stack(  # type: ignore[no-any-return]
        [
            dphi_dxp * ca - k * yp * sa,
            dphi_dxp * sa + k * yp * ca,
            k * (b + zeta) * z_zeta,
        ],
        axis=-1,
    )


def _potential_impl(p: gt.Params, xyz: gt.Sz3, /) -> gt.Sz0:
    a = p["a"]
    _, _, _, eta2, ax, T_plus, T_minus = _bar_terms(p, xyz)
    # |x'| - a + T_-, rationalized where its terms cancel (|x'| < a)
    numer = jnp.where(ax >= a, ax - a + T_minus, eta2 / (T_minus + a - ax))
    denom = ax + a + T_plus
    # Far out numer / denom -> 1, so take log1p of their difference, which is
    # -2a (T_- + T_+ + 2|x'|) / (T_- + T_+): no cancellation.
    T_sum = T_minus + T_plus
    diff = -2 * a * (T_sum + 2 * ax) / T_sum
    ratio = numer / denom
    log_ratio = jnp.where(ratio > 0.5, jnp.log1p(diff / denom), jnp.log(ratio))
    return p["G"] * p["m_tot"] / (2.0 * a) * log_ratio  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def potential(p: gt.Params, xyz: gt.Sz3, /) -> gt.Sz0:
    r"""Specific potential energy.

    .. math::

        \Phi = \frac{G M}{2a}
            \ln\frac{x' - a + T_-}{x' + a + T_+}

    Evaluated in forms that do not cancel (`_potential_impl`), with a
    `jax.custom_jvp` whose position derivative is the analytic gradient
    (`_gradient_impl`), so every autodiff derivative is accurate too.
    """
    return _potential_jvp(p, xyz)


_potential_jvp = with_r_derivative(
    _potential_impl, _gradient_impl, lambda g, v: jnp.sum(g * v, axis=-1)
)


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
