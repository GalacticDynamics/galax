"""Private module that defines `LeeSutoTriaxialNFWPotential` and its functions.

The public API is `galax.potential.leesutotriaxialnfw`.
"""

__all__ = [
    # class
    "LeeSutoTriaxialNFWPotential",
    # functions
    "density",
    "gradient",
    "hessian",
    "laplacian",
    "potential",
]

import functools as ft
import math
from dataclasses import KW_ONLY

from collections.abc import Callable
from typing import Any, Final, final

import equinox as eqx
import jax

import quaxed.lax as qlax
import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue
from xmmutablemap import ImmutableMap

import galax.potential.custom_types as gt
from .base import potential as nfw_potential
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.jax import vectorize_method
from galax.potential._src.kernels import (
    cartesian_gradient,
    cartesian_hessian,
    cartesian_laplacian,
    poisson_density,
)
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.symmetry import Symmetry
from galax.potential._src.utils import safe_vector_norm, with_r_derivative

LOG2: Final = math.log(2.0)


@final
class LeeSutoTriaxialNFWPotential(AbstractSinglePotential):
    r"""Lee-Suto triaxial NFW potential.

    An analytic approximation to the potential of an NFW halo whose iso-density
    surfaces are concentric, coaxial ellipsoids [LeeSuto2003]_. The equipotentials
    are flattened too, but they are rounder than the iso-density ellipsoids. The
    potential is the first-order expansion in the squared eccentricities of the
    exact potential of the triaxial density,
    so it is accurate for moderate flattening and is built from the same
    elementary functions as the NFW potential. The parameter ``m`` sets the
    overall mass scale, ``r_s`` is the scale radius, and ``a1``, ``a2`` and
    ``a3`` are the lengths of the major, intermediate and minor axes of the
    ellipsoids, of which only the ratios matter. The inner density slope is
    :math:`-1` and the outer slope is :math:`-3`, as for the NFW profile, and the
    total mass is infinite.

    .. warning::

        This potential is only physical for :math:`a_1 \ge a_2 \ge a_3`.

    The density that the potential approximates is the triaxial NFW profile of
    [LeeSuto2003]_ and [JingSuto2002LS]_, with the NFW profile of
    [NFW1997LeeSuto]_ evaluated at the ellipsoidal radius :math:`R`,

    .. math::

        \rho_{\mathrm{tri}}(\mathbf{x}) = \frac{\rho_s}{(R / r_s)(1 + R / r_s)^2},
        \qquad
        R^2 = a_1^2 \left(\frac{x^2}{a_1^2} + \frac{y^2}{a_2^2}
            + \frac{z^2}{a_3^2}\right).

    With the squared eccentricities :math:`e_b^2 = 1 - (a_2 / a_1)^2` and
    :math:`e_c^2 = 1 - (a_3 / a_1)^2` and the scaled radius :math:`s = r / r_s`
    with :math:`r^2 = x^2 + y^2 + z^2`, the potential is

    .. math::

        \Phi(\mathbf{x}) = \Phi_0 \left[ F_1(s)
            + \frac{e_b^2 + e_c^2}{2} F_2(s)
            + \frac{e_b^2 y^2 + e_c^2 z^2}{2 r^2} F_3(s) \right],

    where

    .. math::

        F_1(s) &= -\frac{\ln(1 + s)}{s}, \\
        F_2(s) &= -\frac{1}{3} + \frac{2 s^2 - 3 s + 6}{6 s^2}
            + \left(\frac{1}{s} - \frac{1}{s^3}\right) \ln(1 + s), \\
        F_3(s) &= \frac{s^2 - 3 s - 6}{2 s^2 (1 + s)} + \frac{3 \ln(1 + s)}{s^3}.

    The functions :math:`F_1`, :math:`F_2` and :math:`F_3` are those of
    [LeeSuto2003]_ (their eqs. 20-22 with inner slope :math:`\alpha = 1`,
    written out in their eqs. A1-A3), and :math:`F_1` is the NFW potential in
    units of :math:`4 \pi G \rho_s r_s^2`. In polar angles the angular factor is
    :math:`(e_b^2 \sin^2\theta \sin^2\phi + e_c^2 \cos^2\theta) / 2`. The
    amplitude is

    .. math::

        \Phi_0 = 4 \pi G \rho_s r_s^2 = \frac{G m / r_s}
            {\ln 2 - \tfrac{1}{2} + (\ln 2 - \tfrac{3}{4})(e_b^2 + e_c^2)},

    where :math:`G` is the gravitational constant. This fixes ``m`` through the
    circular velocity on the major axis at the scale radius,
    :math:`v_c^2(r_s) = x \, \partial_x \Phi |_{x = r_s} = G m / r_s`, and gives
    the scale density :math:`\rho_s = \Phi_0 / (4 \pi G r_s^2)`. At the centre
    :math:`F_1(0) = -1`, :math:`F_2(0) = 2/3` and :math:`F_3(0) = 0`, so
    :math:`\Phi(0) = -\Phi_0 \left[1 - (e_b^2 + e_c^2) / 3\right]`.

    The density of this class is :math:`\rho = \nabla^2 \Phi / (4 \pi G)`. As the
    potential is linear in :math:`e_b^2` and :math:`e_c^2`, it is the first-order
    expansion of :math:`\rho_{\mathrm{tri}}` in the squared eccentricities,

    .. math::

        \rho(\mathbf{x}) = \frac{\rho_s}{s (1 + s)^2}
            \left[ 1 - \frac{e_b^2 y^2 + e_c^2 z^2}{2 r^2}
                \frac{1 + 3 s}{1 + s} \right].

    Notes
    -----
    For :math:`a_1 = a_2 = a_3` the model is the spherical NFW halo with
    characteristic mass :math:`m / (\ln 2 - 1/2)`, whose enclosed mass is
    :math:`M(<r) = m \left[\ln(1 + s) - s / (1 + s)\right] / (\ln 2 - 1/2)`. In
    that case ``m`` is the mass inside :math:`r_s`, which differs from the ``m`` of
    :class:`~galax.potential.NFWPotential` (the characteristic mass
    :math:`4 \pi \rho_s r_s^3`).

    The potential is the first-order expansion of [LeeSuto2003]_, so it departs
    from the potential obtained by integrating the density
    :math:`\rho_{\mathrm{tri}}` numerically, with the same :math:`\rho_s`, at
    second order in the squared eccentricities. At :math:`r = 5` kpc, for
    :math:`r_s = 15` kpc, a numerical comparison gives differences of 0.4, 2 and
    10 per cent for axis ratios :math:`(a_1, a_2, a_3) = (1, 0.95, 0.9)`,
    :math:`(1, 0.9, 0.8)` and :math:`(1, 0.8, 0.6)`. The density of this class is
    not the triaxial NFW density, and it becomes negative far out along the minor
    axis when :math:`e_c^2 > 2/3` (and along the intermediate axis when
    :math:`e_b^2 > 2/3`). The mass inside an ellipsoid has no simple closed form,
    and neither do the circular velocity away from the major axis and the escape
    velocity. The module function ``density`` evaluates the Laplacian of the
    potential by automatic differentiation and agrees with the expression above
    to rounding error. The functions :math:`F_2` and :math:`F_3` are evaluated in
    forms that avoid cancellation at small and large :math:`s` (see the private
    helpers of the module). The implementation follows the one in ``gala``.

    References
    ----------
    .. [LeeSuto2003] Lee, J., & Suto, Y. (2003). Modeling Intracluster Gas in
       Triaxial Dark Halos: An Analytic Approach. The Astrophysical Journal,
       585, 151-160.
       `ADS <https://ui.adsabs.harvard.edu/abs/2003ApJ...585..151L>`__;
       `doi:10.1086/345931 <https://doi.org/10.1086/345931>`__;
       `arXiv:astro-ph/0211007 <https://arxiv.org/abs/astro-ph/0211007>`__.
       The potential is their eq. 19 with the functions of their appendix A.

    .. [JingSuto2002LS] Jing, Y. P., & Suto, Y. (2002). Triaxial Modeling of Halo
       Density Profiles with High-Resolution N-Body Simulations. The
       Astrophysical Journal, 574, 538-553.
       `doi:10.1086/341065 <https://doi.org/10.1086/341065>`__;
       `arXiv:astro-ph/0202064 <https://arxiv.org/abs/astro-ph/0202064>`__.

    .. [NFW1997LeeSuto] Navarro, J. F., Frenk, C. S., & White, S. D. M. (1997). A
       Universal Density Profile from Hierarchical Clustering. The
       Astrophysical Journal, 490, 493-508.
       `doi:10.1086/304888 <https://doi.org/10.1086/304888>`__;
       `arXiv:astro-ph/9611107 <https://arxiv.org/abs/astro-ph/9611107>`__.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.LeeSutoTriaxialNFWPotential(
    ...     m=1e11, r_s=15, a1=1, a2=0.9, a3=0.8, units="galactic")

    >>> q = u.Q([1, 0, 0], "kpc")
    >>> t = u.Q(0, "Gyr")
    >>> pot.potential(q, t).decompose(pot.units)
    Q(-0.14620419, 'kpc2 / Myr2')

    >>> q = u.Q([0, 1, 0], "kpc")
    >>> pot.potential(q, t).decompose(pot.units)
    Q(-0.14593972, 'kpc2 / Myr2')

    >>> q = u.Q([0, 0, 1], "kpc")
    >>> pot.potential(q, t).decompose(pot.units)
    Q(-0.14570309, 'kpc2 / Myr2')
    """

    m: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass",
        doc=r"""Scale mass.

    This is the mass corresponding to the circular velocity at the scale radius.
    :math:`v_c = \sqrt{G M / r_s}`
    """,
    )

    r_s: AbstractParameter = ParameterField(dimensions="length", doc="Scale radius.")  # type: ignore[assignment]

    a1: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless", default=u.StaticQuantity(1.0, ""), doc="Major axis."
    )

    a2: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        default=u.StaticQuantity(1.0, ""),
        doc="Intermediate axis.",
    )

    a3: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless", default=u.StaticQuantity(1.0, ""), doc="Minor axis."
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    symmetry = Symmetry.PLANE_REFLECTION
    """`Symmetry.PLANE_REFLECTION`: the potential is even in $x$, $y$ and $z$."""

    def __check_init__(self) -> None:
        t = u.Q(0.0, "Myr")
        # TODO: simplify this check
        p = self.params(t)
        _ = eqx.error_if(
            t,
            jnp.logical_or(p["a1"] < p["a2"], p["a2"] < p["a3"]),
            f"a1 {p['a1']} >= a2 {p['a2']} >= a3 {p['a3']} is required",
        )

    @ft.partial(jax.jit)
    def _potential(  # TODO: inputs w/ units
        self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /
    ) -> gt.BBtSz0:
        params = self.params(t)
        return potential(params, xyz)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    @vectorize_method(signature="(3),()->()")
    def _density(self, xyz: gt.QuSz3 | gt.Sz3, t: gt.QuSz0 | gt.Sz0) -> gt.Sz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.ustrip(AllowValue, self.units["time"], t)

        params = self.params(t)
        return density(params, xyz)  # type: ignore[no-any-return]


# =============================================================


@ft.partial(jax.jit)
def potential(p: gt.Params, xyz: gt.BBtSz3, /) -> gt.BtFloatSz0:
    r"""Specific potential energy of the Lee & Suto triaxial NFW halo.

    .. math::

        \Phi = \Phi_0 \left[ F_1(s)
            + \frac{e_b^2 + e_c^2}{2} F_2(s)
            + \frac{e_b^2 y^2 + e_c^2 z^2}{2 r^2} F_3(s) \right],
        \qquad
        \Phi_0 = \frac{G m / r_s}
            {\ln 2 - \tfrac{1}{2} + (\ln 2 - \tfrac{3}{4})(e_b^2 + e_c^2)},

    with :math:`s = r / r_s`, :math:`e_b^2 = 1 - (a_2 / a_1)^2`,
    :math:`e_c^2 = 1 - (a_3 / a_1)^2` and the functions :math:`F_1`, :math:`F_2`
    and :math:`F_3` of Lee & Suto (2003), eq. 19 and appendix A. See
    :class:`LeeSutoTriaxialNFWPotential` for the definitions.

    """
    # Compute parameters
    # https://github.com/adrn/gala/blob/2067009de41518a71c674d0252bc74a7b2d78a36/gala/potential/potential/builtin/builtin_potentials.c#L1472

    # Parse inputs
    r = safe_vector_norm(xyz)

    v_c2 = p["G"] * p["m"] / p["r_s"]

    # 1- eccentricities
    e_b2 = 1 - jnp.square(p["a2"] / p["a1"])
    e_c2 = 1 - jnp.square(p["a3"] / p["a1"])

    # The potential at the origin
    phi0 = v_c2 / (LOG2 - 0.5 + (LOG2 - 0.75) * (e_b2 + e_c2))

    # The potential at the given position
    s = r / p["r_s"]

    # The angular factors
    costh2 = xyz[..., 2] ** 2 / r**2  # z^2 / r^2
    sinthsinphi2 = xyz[..., 1] ** 2 / r**2  # (sin(theta) * sin(phi))^2
    # Note that ꜛ is safer than computing the separate pieces, as it avoids
    # x=y=0, z!=0, which would result in a NaN.

    # F1 = -ln(1+s)/s is NFW's radial function; reuse it for its derivatives.
    F1 = nfw_potential({"G": 1.0, "m": 1.0, "r_s": 1.0}, s)
    F2, F3 = _F2(s), _F3(s)

    # Select the output, r=0 is a special case.
    out: gt.BtFloatSz0 = phi0 * qlax.select(
        s == 0,
        jnp.ones_like(s),
        (F1 + (e_b2 + e_c2) / 2 * F2 + (e_b2 * sinthsinphi2 + e_c2 * costh2) / 2 * F3),
    )
    return out


def _horner2(
    coeffs: tuple[float, ...], w: gt.BtFloatSz0, /
) -> tuple[gt.BtFloatSz0, ...]:
    """``(P, P', P'')`` of ``sum(c_k w^k)``, in one Horner pass."""
    p, d1, d2 = jnp.full_like(w, coeffs[-1]), jnp.zeros_like(w), jnp.zeros_like(w)
    for c in reversed(coeffs[:-1]):
        d2 = d2 * w + d1
        d1 = d1 * w + p
        p = p * w + c
    return p, d1, 2 * d2


def _series_or_far(
    series: Callable[[gt.BtFloatSz0], tuple[gt.BtFloatSz0, ...]],
    far: Callable[[gt.BtFloatSz0], gt.BtFloatSz0],
) -> Callable[[gt.BtFloatSz0], gt.BtFloatSz0]:
    r"""``F(s)``: a series in $w = s/(2+s)$ for $s \leq 2$, ``far(s)`` beyond.

    ``series(w)`` gives $(F, dF/dw, d^2F/dw^2)$. The $s$-derivatives are
    supplied through `with_r_derivative` rather than left to autodiff of a
    50-term series, which would make the Hessian ~9x slower: with
    $w_s = (1-w)^2/2$ and $w_{ss} = -(1-w)^3/2$,
    $F_s = F_w w_s$ and $F_{ss} = F_{ww} w_s^2 + F_w w_{ss}$.
    """

    def near(s: gt.BtFloatSz0) -> tuple[gt.BtFloatSz0, ...]:
        sc = jnp.where(s <= 2, s, 2.0)
        w = sc / (2 + sc)
        F, F_w, F_ww = series(w)
        w_s, w_ss = (1 - w) ** 2 / 2, -((1 - w) ** 3) / 2
        return F, F_w * w_s, F_ww * w_s**2 + F_w * w_ss

    def far1(s: gt.BtFloatSz0) -> gt.BtFloatSz0:
        return jax.jvp(far, (s,), (jnp.ones_like(s),))[1]  # type: ignore[no-any-return]

    def far2(s: gt.BtFloatSz0) -> gt.BtFloatSz0:
        return jax.jvp(far1, (s,), (jnp.ones_like(s),))[1]  # type: ignore[no-any-return]

    def pick(k: int, far_k: Callable[[gt.BtFloatSz0], gt.BtFloatSz0]) -> Any:
        # far_k gets s > 2 only, so the unused branch (~1/s^5 at small s)
        # cannot put a NaN into a reverse-mode derivative.
        return lambda _, s: jnp.where(
            s <= 2, near(s)[k], far_k(jnp.where(s <= 2, 3.0, s))
        )

    F = with_r_derivative(pick(0, far), with_r_derivative(pick(1, far1), pick(2, far2)))
    return lambda s: F({}, s)


# F2 = (1 - w) P(w): P's even coefficients (j+2)/((2j+1)(2j+3)) and odd ones
# 1/(2(2j+3)), all positive; 54 terms reach 2^-54 ~ 5e-17 at w = 1/2.
_F2_P: Final = tuple(
    (k // 2 + 2) / ((k + 1) * (k + 3)) if k % 2 == 0 else 1 / (2 * (k + 2))
    for k in range(54)
)
# G = (1 + s) F3 = w/2 - (3/2)(1 - w) w^2 sum_i w^(2i) / (4(i+2)^2 - 1).
_F3_G: Final = tuple(
    0.5 if k == 1
    else -1.5 / (4 * (k // 2 + 1) ** 2 - 1) if k >= 2 and k % 2 == 0
    else 1.5 / (4 * ((k - 1) // 2 + 1) ** 2 - 1) if k >= 3
    else 0.0
    for k in range(54)
)  # fmt: skip


def _F2_series(w: gt.BtFloatSz0, /) -> tuple[gt.BtFloatSz0, ...]:
    P, P1, P2 = _horner2(_F2_P, w)
    return (1 - w) * P, (1 - w) * P1 - P, (1 - w) * P2 - 2 * P1


def _F3_series(w: gt.BtFloatSz0, /) -> tuple[gt.BtFloatSz0, ...]:
    G, G1, G2 = _horner2(_F3_G, w)
    h, h1, h2 = (1 - w) / (1 + w), -2 / (1 + w) ** 2, 4 / (1 + w) ** 3
    return G * h, G1 * h + G * h1, G2 * h + 2 * G1 * h1 + G * h2


def _F2_far(s: gt.BtFloatSz0, /) -> gt.BtFloatSz0:
    L = jnp.log1p(s)
    return (L - 0.5) / s + (1 - L / s) / s**2  # type: ignore[no-any-return]


def _F3_far(s: gt.BtFloatSz0, /) -> gt.BtFloatSz0:
    return (s**2 - 3 * s - 6) / (2 * s**2 * (1 + s)) + 3 * jnp.log1p(s) / s**3  # type: ignore[no-any-return]


_F2: Final = _series_or_far(_F2_series, _F2_far)
r"""Lee & Suto's $F_2(s) = -\frac13 + \frac{2s^2 - 3s + 6}{6s^2}
+ \left(\frac1s - \frac1{s^3}\right)\ln(1+s)$, without cancellation.

As written, terms $\sim 1/s^2$ cancel to $F_2(0) = 2/3$, and far out the two
$1/3$ cancel. For $s \leq 2$, with $w = s/(2+s)$ (so
$\ln(1+s) = 2\,\text{atanh}\,w$) and $X = w^2$,

$$ F_2 = (1-w) \sum_{j \geq 0} \left[\frac{(j+2) X^j}{(2j+1)(2j+3)}
   + \frac{w X^j}{2(2j+3)}\right], $$

every term positive. Beyond, $F_2 = (\ln(1+s) - \tfrac12)/s
+ (1 - \ln(1+s)/s)/s^2$, also positive.
"""

_F3: Final = _series_or_far(_F3_series, _F3_far)
r"""Lee & Suto's $F_3(s) = \frac{s^2 - 3s - 6}{2s^2(1+s)}
+ \frac{3\ln(1+s)}{s^3}$, without cancellation.

As written, terms $\sim 1/s^2$ cancel to $F_3(0) = 0$, and up to $s \sim 2$
by up to a factor 160. For $s \leq 2$, with $w = s/(2+s)$ and $X = w^2$,
$F_3 = G (1-w)/(1+w)$ for $G = (1+s) F_3$,

$$ G = \frac{w}{2} - \frac{3}{2}(1-w) X \sum_{j \geq 0}
   \frac{X^j}{4(j+2)^2 - 1}, $$

which cancels by at most a factor 1.1 (at $s = 2$). Beyond, the direct form
cancels by less than a factor 5.
"""


# Derived quantities. There is no closed-form density: it is Poisson's equation
# applied to the autodiff Laplacian of `potential`.
gradient = cartesian_gradient(potential)
hessian = cartesian_hessian(potential)
laplacian = cartesian_laplacian(potential)
density = poisson_density(laplacian)
