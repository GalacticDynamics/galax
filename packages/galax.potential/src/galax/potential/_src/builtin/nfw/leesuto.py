"""galax: Galactic Dynamix in Jax."""

__all__ = ["LeeSutoTriaxialNFWPotential"]

import functools as ft
from dataclasses import KW_ONLY

from collections.abc import Callable
from typing import Any, Final, final

import equinox as eqx
import jax

import quaxed.lax as qlax
import quaxed.numpy as jnp
import unxt as u
from xmmutablemap import ImmutableMap

import galax.potential.custom_types as gt
from .base import potential as nfw_potential
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.symmetry import Symmetry
from galax.potential._src.utils import safe_vector_norm, with_r_derivative

LOG2: Final = jnp.log(jnp.asarray(2.0))


@final
class LeeSutoTriaxialNFWPotential(AbstractSinglePotential):
    """Approximate triaxial (in the density) NFW potential.

    Approximation of a Triaxial NFW Potential with the flattening in the
    density, not the potential. See Lee & Suto (2003) for details.

    .. warning::

        This potential is only physical for `a1 >= a2 >= a3`.

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
        dimensions="dimensionless", default=u.Q(1.0, ""), doc="Major axis."
    )

    a2: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        default=u.Q(1.0, ""),
        doc="Intermediate axis.",
    )

    a3: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless", default=u.Q(1.0, ""), doc="Minor axis."
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
        a1 = self.a1(t, ustrip=self.units["dimensionless"])
        a2 = self.a2(t, ustrip=self.units["dimensionless"])
        a3 = self.a3(t, ustrip=self.units["dimensionless"])
        _ = eqx.error_if(
            t,
            jnp.logical_or(a1 < a2, a2 < a3),
            f"a1 {a1} >= a2 {a2} >= a3 {a3} is required",
        )

    @ft.partial(jax.jit)
    def _potential(  # TODO: inputs w/ units
        self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /
    ) -> gt.BBtSz0:
        t = u.Q.from_(t, self.units["time"])
        ul = self.units["dimensionless"]
        params = {
            "G": self.constants["G"].value,
            "m": self.m(t, ustrip=self.units["mass"]),
            "r_s": self.r_s(t, ustrip=self.units["length"]),
            "a1": self.a1(t, ustrip=ul),
            "a2": self.a2(t, ustrip=ul),
            "a3": self.a3(t, ustrip=ul),
        }
        return potential(params, xyz)  # type: ignore[no-any-return]


# =============================================================


@ft.partial(jax.jit)
def potential(p: gt.Params, xyz: gt.BBtSz3, /) -> gt.BtFloatSz0:
    r"""Specific potential energy."""
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


_F2 = _series_or_far(_F2_series, _F2_far)
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

_F3 = _series_or_far(_F3_series, _F3_far)
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
