"""Private module that defines `LongMuraliBarPotential` and its functions.

The public API is `galax.potential.longmuralibar`.
"""

__all__ = [
    # class
    "LongMuraliBarPotential",
    # functions
    "density",
    "gradient",
    "hessian",
    "laplacian",
    "potential",
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
from galax.potential._src.kernels import (
    cartesian_gradient,
    cartesian_hessian,
    poisson_laplacian,
)
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.utils import with_r_derivative


@final
class LongMuraliBarPotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Long & Murali bar potential.

    The Long & Murali model describes a triaxial bar of finite total mass. It is
    built by smoothing a thin uniform "needle" of length :math:`2a` along the
    bar axis with the Miyamoto-Nagai density, so that the potential is an
    elementary function and the density is positive everywhere
    [LongMurali1992]_. The bar lies in the :math:`x`-:math:`y` plane with its long
    axis at the position angle ``alpha`` from the :math:`x` axis, and is
    symmetric about the planes :math:`x' = 0`, :math:`y' = 0` and :math:`z = 0`,
    where :math:`x'` and :math:`y'` are the coordinates in the frame of the bar.
    The parameter ``m_tot`` is the total mass, ``a`` the half-length of the
    needle, ``b`` the softening length in the plane perpendicular to the bar axis
    and ``c`` the vertical softening length. The ratio ``c / b`` sets the
    thickness of the bar. The model does not rotate: the bar angle is fixed in
    time.

    The potential is

    .. math::

        \Phi = \frac{G m_{\mathrm{tot}}}{2a}
            \ln\frac{x' - a + T_-}{x' + a + T_+},

    where

    .. math::

        x' &= x \cos\alpha + y \sin\alpha, \qquad
        y' = -x \sin\alpha + y \cos\alpha, \\
        \zeta &= \sqrt{c^2 + z^2}, \qquad
        \eta^2 = y'^2 + (b + \zeta)^2, \qquad
        T_\pm = \sqrt{(a \pm x')^2 + \eta^2},

    and :math:`G` is the gravitational constant. This is the convolution of the
    needle with the Miyamoto-Nagai potential of scales :math:`(b, c)`,

    .. math::

        \Phi = -\frac{G m_{\mathrm{tot}}}{2a} \int_{-a}^{a}
            \frac{\mathrm{d}s}{\sqrt{(x' - s)^2 + \eta^2}}.

    The density is the same convolution of the Miyamoto-Nagai density,

    .. math::

        \rho = \frac{m_{\mathrm{tot}}}{2a} \int_{-a}^{a}
            \frac{c^2}{4\pi}
            \frac{b \left[(x' - s)^2 + y'^2\right] + (b + 3\zeta)(b + \zeta)^2}
                 {\left[(x' - s)^2 + \eta^2\right]^{5/2} \zeta^3} \, \mathrm{d}s,

    and it integrates to the total mass :math:`m_{\mathrm{tot}}`. The gradient
    of the potential in the frame of the bar is

    .. math::

        \frac{\partial\Phi}{\partial x'} &=
            \frac{2 G m_{\mathrm{tot}} x'}{T_+ T_- (T_+ + T_-)}, \\
        \frac{\partial\Phi}{\partial y'} &=
            \frac{G m_{\mathrm{tot}} y'}{2 \eta^2 T_+ T_-}
            \left(T_+ + T_- - \frac{4 x'^2}{T_+ + T_-}\right), \\
        \frac{\partial\Phi}{\partial z} &=
            \frac{G m_{\mathrm{tot}} (b + \zeta) z}{2 \eta^2 \zeta T_+ T_-}
            \left(T_+ + T_- - \frac{4 x'^2}{T_+ + T_-}\right).

    Notes
    -----
    The potential is even in :math:`x'`. The code uses :math:`|x'|`, and it
    evaluates the logarithm and the gradient in rearranged forms that avoid
    cancellation near the centre and far from the bar. Far from the bar the
    potential tends to :math:`-G m_{\mathrm{tot}} / r`. For
    :math:`a \to 0` the needle shrinks to a point and the model becomes the
    Miyamoto-Nagai potential with scale length ``b`` and scale height ``c``
    (see :class:`~galax.potential.MiyamotoNagaiPotential`).

    The density function has no short closed form. It is Poisson's equation,
    :math:`\rho = \nabla^2 \Phi / (4 \pi G)`, applied to the potential with a
    computer algebra system, and the module function evaluates the resulting long
    expression in terms of common sub-expressions. It agrees with the
    convolution integral above. The enclosed mass, the circular velocity and the
    escape velocity have no simple closed form for this triaxial model.

    References
    ----------
    .. [LongMurali1992] Long, K., & Murali, C. (1992). Analytical Potentials for
       Barred Galaxies. The Astrophysical Journal, 397, 44-48.
       `ADS <https://ui.adsabs.harvard.edu/abs/1992ApJ...397...44L>`__;
       `doi:10.1086/171764 <https://doi.org/10.1086/171764>`__.
       The potential is their eq. 8b, the needle integral their eq. 8a and the
       forces their eqs. 9a-9c.

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
    r"""Bar-frame coordinates and the auxiliary quantities of the potential.

    The bar-frame coordinates are :math:`x'` and :math:`y'`, and the auxiliary
    quantities are :math:`\zeta = \sqrt{c^2+z^2}`,
    :math:`\eta^2 = y'^2 + (b + \zeta)^2` and
    :math:`T_\pm = \sqrt{(a \pm |x'|)^2 + \eta^2}`. Everything below is even in
    :math:`x'`, so it is written for :math:`|x'|`.
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

    With :math:`\Phi = -\frac{GM}{2a}\left[\sinh^{-1}\frac{a+x'}{\eta}
    + \sinh^{-1}\frac{a-x'}{\eta}\right]` (the log form, rationalized),

    .. math::

        \frac{\partial\Phi}{\partial x'}
            = \frac{2 G M x'}{(T_- + T_+) T_+ T_-}, \qquad
        \frac{1}{\eta}\frac{\partial\Phi}{\partial\eta}
            = \frac{G M}{2\eta^2}
              \frac{(T_- + T_+)^2 - 4x'^2}{(T_- + T_+) T_+ T_-},

    where autodiff of the log takes differences of :math:`\sim 1/r` terms that
    cancel to :math:`\sim a/r^2` far out, and to :math:`\sim |x|/a^2` at the centre.
    :math:`(T_-+T_+)^2 - 4x'^2 = (T_-+T_+-2|x'|)(T_-+T_++2|x'|)`, with
    :math:`T_- + T_+ - 2|x'| = \eta^2/(T_+ + |x'| + a) + (T_- - |x'| + a)` and the
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
    r"""Specific potential energy of the Long & Murali bar (eq. 8b).

    .. math::

        \Phi = \frac{G m_{\mathrm{tot}}}{2a}
            \ln\frac{x' - a + T_-}{x' + a + T_+},

    with :math:`x' = x \cos\alpha + y \sin\alpha`,
    :math:`y' = -x \sin\alpha + y \cos\alpha`,
    :math:`\zeta = \sqrt{c^2 + z^2}`,
    :math:`\eta^2 = y'^2 + (b + \zeta)^2` and
    :math:`T_\pm = \sqrt{(a \pm x')^2 + \eta^2}`.

    Evaluated in forms that do not cancel (``_potential_impl``), with a
    ``jax.custom_jvp`` whose position derivative is the analytic gradient
    (``_gradient_impl``), so every autodiff derivative is accurate too.
    """
    return _potential_jvp(p, xyz)


_potential_jvp = with_r_derivative(
    _potential_impl, _gradient_impl, lambda g, v: jnp.sum(g * v, axis=-1)
)


@ft.partial(jax.jit)
def density(p: gt.Params, xyz: gt.Sz3, /) -> gt.Sz0:
    r"""Density for the Long & Murali (1992) bar potential.

    The convolution of the needle with the Miyamoto-Nagai density of scales
    :math:`(b, c)`,

    .. math::

        \rho = \frac{m_{\mathrm{tot}}}{2a} \int_{-a}^{a}
            \frac{c^2}{4\pi}
            \frac{b \left[(x' - s)^2 + y'^2\right] + (b + 3\zeta)(b + \zeta)^2}
                 {\left[(x' - s)^2 + \eta^2\right]^{5/2} \zeta^3} \, \mathrm{d}s,

    with :math:`x'`, :math:`y'`, :math:`\zeta` and :math:`\eta` as in
    :func:`potential`. Equivalently, it is Poisson's equation,
    :math:`\nabla^2\Phi = 4\pi G \rho`, applied to :func:`potential` with a
    computer algebra system (the intermediate quantities below are its common
    sub-expressions, not individually meaningful). It has been checked against
    the convolution integral and against the finite-difference Hessian of
    :func:`potential`.
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


# Derived quantities, built from the kernels above.
laplacian = poisson_laplacian(density)
gradient = cartesian_gradient(potential)
hessian = cartesian_hessian(potential)
