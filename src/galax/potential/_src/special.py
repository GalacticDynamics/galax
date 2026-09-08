r"""Special functions used by the potential models.

This module is private API. See the public API in `galax.potential`.

The incomplete beta function

.. math::

    B(a, b, z) = \int_0^z t^{a-1} (1-t)^{b-1} dt

is the *unregularized* one (Zhao 1996, Eq. 43; DLMF 8.17.1); potentials with
double power-law densities are written in terms of it. `jax.scipy.special`
provides only the regularized `betainc`, and reconstructing the above as
``beta(a, b) * betainc(a, b, z)`` is nan whenever ``b <= 0``, because the
complete beta function diverges there even though the product does not.

`jax.scipy.special.hyp2f1` can express it for any ``b`` (DLMF 8.17.7), but it
is a `lax.while_loop` whose trip count is data-dependent: under `vmap` every
lane pays the worst lane's iteration count, and its derivative runs a second
such loop. `incomplete_beta` below is instead two fixed-length, geometrically
convergent series with an exact O(1) derivative rule.

See also `galax.potential._src.builtin.nfw.hyp2f1`, which solves the same
problem for `gNFWPotential`, but only for the two parameter patterns that
model needs (``a == 1``, or ``b == 0``), for which it has closed forms.

When ``a`` and ``b`` are fixed for the lifetime of a potential,
`ChebyshevIncompleteBeta` replaces those series with a short polynomial fitted
once at construction.
"""

__all__: tuple[str, ...] = ()

import functools as ft

from collections.abc import Callable
from jaxtyping import Float

import equinox as eqx
import jax
import numpy as np
from jax.custom_derivatives import SymbolicZero

import quaxed.numpy as jnp

import galax.potential.custom_types as gt

_NTERMS = 64
"""Series length. Both series below converge like 2^-k, so this is ~1e-19."""

_POLE_BAND = 1e-3
"""Half-width of the band around `b + m == 0` where `_large_z` expands.

Below this the direct form cancels; above it the expansion's O(s^4 L^5)
truncation shows, for `L = log(1-z)`.
"""


def _small_z(a: gt.Sz0, b: gt.Sz0, z: gt.BBtSz0) -> gt.BBtFloatSz0:
    r"""$B(a, b, z)$ for $z \leq 1/2$, by the defining Taylor series.

    Expanding $(1-t)^{b-1}$ binomially and integrating term by term,

    .. math::

        B(a, b, z) = z^a \sum_{k=0}^\infty \frac{(1-b)_k}{k!\,(a+k)} z^k

    The terms fall off like $z^k \leq 2^{-k}$, hence the fixed term count.

    Summed into a `jax.lax.scan` carry: the terms are batched over `z`, so
    materializing them all at once would cost a ``(*batch, 64)`` temporary and
    make this memory- rather than flop-bound.
    """

    def step(
        carry: tuple[gt.BBtFloatSz0, gt.BBtFloatSz0, gt.Sz0], k: gt.Sz0
    ) -> tuple[tuple[gt.BBtFloatSz0, gt.BBtFloatSz0, gt.Sz0], None]:
        total, z_pow, coeff = carry  # coeff = (1-b)_k / k!
        total = total + coeff / (a + k) * z_pow
        return (total, z_pow * z, coeff * (k + 1.0 - b) / (k + 1.0)), None

    init = (jnp.zeros_like(z), jnp.ones_like(z), jnp.ones_like(a))
    (total, _, _), _ = jax.lax.scan(step, init, jnp.arange(_NTERMS))
    return z**a * total  # type: ignore[no-any-return]


def _large_z(a: gt.Sz0, b: gt.Sz0, z: gt.BBtSz0) -> gt.BBtFloatSz0:
    r"""$B(a, b, z)$ for $z > 1/2$, by reflecting about $t = 1/2$.

    Substituting $t = 1-u$ and splitting the range at $u = 1/2$,

    .. math::

        B(a, b, z) = \int_{w}^{1} u^{b-1}(1-u)^{a-1} du
                   = B(a, b, 1/2) + \int_w^{1/2} u^{b-1}(1-u)^{a-1} du

    with $w = 1 - z$. Expanding $(1-u)^{a-1}$ binomially (legitimate since
    $u \leq 1/2$ on the remaining range) and integrating term by term,

    .. math::

        B(a, b, z) = B(a, b, 1/2)
            + \sum_{m=0}^\infty \frac{(1-a)_m}{m!}
              \frac{(1/2)^{b+m} - w^{b+m}}{b+m}

    where the $b + m \to 0$ term is $\ln(1/(2w))$. Both pieces converge like
    $2^{-m}$.

    This branch is what makes $b \leq 0$ work: it never forms the complete beta
    function $B(a, b)$, which is what diverges there, while keeping the genuine
    $z \to 1$ divergence of $B(a, b, z)$ itself exact (as $w^b$, or $-\ln w$
    when $b = 0$).

    Summed into a `jax.lax.scan` carry, for the same reason as `_small_z`.

    Each term is $(e^{sL_1} - e^{sL_2})/s$ with $s = b + m$, $L_1 = \ln(1/2)$
    and $L_2 = \ln w$, which the running powers below evaluate directly. That
    cancels catastrophically as $s \to 0$, so within `_POLE_BAND` of zero it
    switches to the expansion of the same expression,

    .. math::

        \frac{e^{sL_1} - e^{sL_2}}{s}
            = \sum_{k \geq 1} \frac{L_1^k - L_2^k}{k!} s^{k-1}

    whose $L$-powers are loop-invariant, so this costs a few extra FMAs rather
    than transcendentals. Carrying the $O(s)$ term also makes the
    $b$-derivative right *at* $s = 0$, which integer slopes do hit (e.g.
    $\gamma = 2$ in Zhao's Eq. 7).
    """
    w = 1.0 - z
    log_half, log_w = jnp.log(0.5), jnp.log(w)
    # L1^k - L2^k over k!, for the near-pole expansion.
    d1 = log_half - log_w
    d2 = (log_half**2 - log_w**2) / 2
    d3 = (log_half**3 - log_w**3) / 6
    d4 = (log_half**4 - log_w**4) / 24

    def step(
        carry: tuple[gt.BBtFloatSz0, gt.BBtFloatSz0, gt.Sz0, gt.Sz0], m: gt.Sz0
    ) -> tuple[tuple[gt.BBtFloatSz0, gt.BBtFloatSz0, gt.Sz0, gt.Sz0], None]:
        total, w_pow, half_pow, coeff = carry  # coeff = (1-a)_m / m!
        s = b + m
        near_pole = jnp.abs(s) < _POLE_BAND
        s_safe = jnp.where(near_pole, 1.0, s)
        term = jnp.where(
            near_pole,
            d1 + s * (d2 + s * (d3 + s * d4)),
            (half_pow - w_pow) / s_safe,
        )
        total = total + coeff * term
        carry = (total, w_pow * w, half_pow * 0.5, coeff * (m + 1.0 - a) / (m + 1.0))
        return carry, None

    init = (jnp.zeros_like(w), w**b, 0.5**b, jnp.ones_like(a))
    (total, _, _, _), _ = jax.lax.scan(step, init, jnp.arange(_NTERMS))
    return _small_z(a, b, jnp.full_like(w, 0.5)) + total  # type: ignore[no-any-return]


def _incomplete_beta_impl(a: gt.Sz0, b: gt.Sz0, z: gt.BBtSz0) -> gt.BBtFloatSz0:
    r"""$B(a, b, z)$ for $a > 0$, any real $b$, and $z \in [0, 1]$.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax.scipy.special as jsp
    >>> from galax.potential._src.special import incomplete_beta

    It agrees with the regularized incomplete beta function wherever that is
    defined, including close to the $z \to 1$ endpoint:

    >>> a, b = 2.0, 1.5
    >>> z = jnp.asarray([0.3, 0.999])
    >>> bool(jnp.allclose(incomplete_beta(a, b, z),
    ...                   jsp.beta(a, b) * jsp.betainc(a, b, z)))
    True

    But unlike that product it stays finite for ``b <= 0``, where the complete
    beta function diverges:

    >>> round(float(incomplete_beta(2.0, 0.0, jnp.asarray(0.5))), 8)
    0.19314718

    >>> bool(jnp.isnan(jsp.beta(2.0, 0.0) * jsp.betainc(2.0, 0.0, 0.5)))
    True

    """
    z = jnp.asarray(z)
    # Both branches are evaluated, so clamp each one's input to the range where
    # it is well behaved; `where` then discards the unused value.
    return jnp.where(  # type: ignore[no-any-return]
        z <= 0.5,
        _small_z(a, b, jnp.minimum(z, 0.5)),
        _large_z(a, b, jnp.maximum(z, 0.5)),
    )


@jax.custom_jvp
def incomplete_beta(a: gt.Sz0, b: gt.Sz0, z: gt.BBtSz0) -> gt.BBtFloatSz0:
    """See `_incomplete_beta_impl` for the definition and examples.

    The `jax.custom_jvp` gives the `z`-derivative in O(1) -- it is just the
    integrand evaluated at the endpoint (Leibniz) -- instead of differentiating
    through a 64-term series. It is a `custom_jvp`, not a `custom_vjp`, so that
    `jax.hessian`'s ``jacfwd(jacrev(...))`` still composes for anything
    downstream that has not been given an analytic form.
    """
    return _incomplete_beta_impl(a, b, z)


@ft.partial(incomplete_beta.defjvp, symbolic_zeros=True)
def _incomplete_beta_jvp(
    primals: tuple[gt.Sz0, gt.Sz0, gt.BBtSz0],
    tangents: tuple[
        gt.Sz0 | SymbolicZero, gt.Sz0 | SymbolicZero, gt.BBtSz0 | SymbolicZero
    ],
) -> tuple[gt.BBtFloatSz0, gt.BBtFloatSz0]:
    a, b, z = primals
    a_dot, b_dot, z_dot = tangents

    primal_out = _incomplete_beta_impl(a, b, z)
    tangent_out = jnp.zeros_like(primal_out)

    # d/dz B(a, b, z) = z^(a-1) (1-z)^(b-1): the integrand at the endpoint.
    if not isinstance(z_dot, SymbolicZero):
        tangent_out = tangent_out + z ** (a - 1.0) * (1.0 - z) ** (b - 1.0) * z_dot

    # The a/b tangents are only needed when the parameters are themselves
    # differentiated; there is no cheap closed form, so fall back to autodiff of
    # the series. Skipped entirely in the common case, where the differentiation
    # is with respect to position at fixed a and b.
    a_zero, b_zero = isinstance(a_dot, SymbolicZero), isinstance(b_dot, SymbolicZero)
    if not (a_zero and b_zero):
        _, ab_tangent = jax.jvp(
            lambda aa, bb: _incomplete_beta_impl(aa, bb, z),
            (a, b),
            (
                jnp.zeros_like(a) if a_zero else a_dot,
                jnp.zeros_like(b) if b_zero else b_dot,
            ),
        )
        tangent_out = tangent_out + ab_tangent

    return primal_out, tangent_out


incomplete_beta = jax.jit(incomplete_beta)  # type: ignore[assignment]


# ===================================================================
# Fitted form, for a potential whose (a, b) never change


_FIT_SERIES_TERMS = 400
"""Series terms used to build the fit targets. Host-side and once, so cheap."""


def _poch_over_fact(x: float, n: int) -> np.ndarray:
    """(x)_k / k! for k = 0 .. n-1."""
    k = np.arange(1, n)
    return np.cumprod(np.concatenate([[1.0], (x + k - 1) / k]))


def _lo_target(a: float, b: float, z: np.ndarray) -> np.ndarray:
    r"""$_2F_1(a, 1-b; a+1; z) = a B(a,b,z) / z^a$, the analytic part of `_small_z`."""
    k = np.arange(_FIT_SERIES_TERMS)
    coeff = _poch_over_fact(1.0 - b, _FIT_SERIES_TERMS) * a / (a + k)
    return (coeff * z[..., None] ** k).sum(-1)


def _hi_target(a: float, b: float, w: np.ndarray) -> np.ndarray:
    r"""$V(w) = \sum_m \frac{(1-a)_m}{m!} \frac{w^m}{b+m}$, or its `b == 0` form.

    This is the analytic factor in `B(a, b, 1-w) = const - w^b V(w)` -- see
    `ChebyshevIncompleteBeta`. Summing the series rather than differencing
    values of `B` avoids the cancellation that makes the latter useless as
    `w -> 0`, which is exactly where the fit needs its nodes.
    """
    m = np.arange(_FIT_SERIES_TERMS)
    c = _poch_over_fact(1.0 - a, _FIT_SERIES_TERMS)
    w = w[..., None]
    if b == 0:  # the m = 0 term is the -log(w) split off in __call__
        return (c[1:] * w ** m[1:] / m[1:]).sum(-1)
    return (c * w**m / (b + m)).sum(-1)


def _chebfit(f: Callable[[np.ndarray], np.ndarray], n: int) -> np.ndarray:
    """Chebyshev coefficients of `f` on [0, 1/2], from `n` Gauss-Chebyshev nodes."""
    x = np.cos(np.pi * (np.arange(n) + 0.5) / n)
    return np.polynomial.chebyshev.chebfit(x, f(0.25 * (x + 1)), n - 1)


def _clenshaw(coef: Float[np.ndarray, " n"], x: gt.BBtSz0) -> gt.BBtFloatSz0:
    """Evaluate a Chebyshev series at `x` in [-1, 1] by Clenshaw recurrence."""
    b1 = jnp.zeros_like(x)
    b2 = jnp.zeros_like(x)
    for c in coef[:0:-1]:  # static length, so this unrolls at trace time
        b1, b2 = 2.0 * x * b1 - b2 + c, b1
    return x * b1 - b2 + coef[0]  # type: ignore[no-any-return]


class ChebyshevIncompleteBeta(eqx.Module):
    r"""$B(a, b, \cdot)$ for fixed $a, b$, as two Chebyshev panels.

    `incomplete_beta` sums 64 series terms on every call because it must work
    for any $(a, b)$. A potential with fixed power-law indices needs only one
    function of one variable, which a short polynomial captures instead.

    The function is a power law at both ends -- $z^a$ as $z \to 0$, and $w^b$
    (or $\ln w$, or nothing) as $w = 1 - z \to 0$ -- so neither half is
    analytic across the whole interval. Splitting at $z = 1/2$ and dividing out
    the end behaviour leaves two analytic functions, which Chebyshev series
    converge on geometrically:

    .. math::

        B(a, b, z) &= \frac{z^a}{a} \, {}_2F_1(a, 1-b; a+1; z), & z \leq 1/2 \\
        B(a, b, z) &= \mathrm{const} - w^b V(w),                & w \leq 1/2 \\
        B(a, b, z) &= \mathrm{const} + \ln w - V_0(w),          & w \leq 1/2,\, b = 0

    In practice ~18 coefficients per panel reach 1e-13 over the whole
    $(\alpha, \beta, \gamma)$ range of `ZhaoPotential`.

    ``const`` is fixed by matching the two panels at $z = 1/2$, so the fit
    needs no reference implementation beyond its own series.
    """

    a: float = eqx.field(static=True)
    b: float = eqx.field(static=True)
    const: float = eqx.field(static=True)
    at_half: float = eqx.field(static=True)
    """``B(a, b, 1/2)``, which the panel matching computes anyway."""
    lo_coef: Float[np.ndarray, " n"]
    hi_coef: Float[np.ndarray, " n"]

    def __init__(self, a: float, b: float, n: int = 24) -> None:
        a, b = float(a), float(b)
        if a <= 0:
            msg = f"`a` must be positive, got {a}."
            raise ValueError(msg)
        # b <= 0 is fine, and b == 0 has its own (log) branch, but the other
        # non-positive integers need a log term this does not carry.
        if b < 0 and b == int(b):
            msg = (
                f"b = {b} is a negative integer, where B(a, b, .) picks up a "
                "logarithmic term this fit does not represent. Use "
                "`ZhaoPotential` (the series form) for these indices."
            )
            raise ValueError(msg)

        self.a, self.b = a, b
        # numpy, not jax: the fit may run while a jit trace is active (the
        # first call through a jitted method), and device arrays made there
        # would be tracers. As numpy they are plain constants, which is also
        # what we want them folded into the jaxpr as.
        self.lo_coef = _chebfit(lambda z: _lo_target(a, b, z), n)
        self.hi_coef = _chebfit(lambda w: _hi_target(a, b, w), n)

        # Match the panels at z = w = 1/2, which fixes `const`.
        half = 0.5**a * float(_lo_target(a, b, np.array(0.5))) / a
        v_half = float(_hi_target(a, b, np.array(0.5)))
        self.const = float(
            half - np.log(2.0) + v_half if b == 0 else half + 0.5**b * v_half
        )
        self.at_half = float(half)

    def __call__(self, z: gt.BBtSz0) -> gt.BBtFloatSz0:
        """Evaluate B(a, b, z), for z in [0, 1].

        Not jitted: every caller already is, and the coefficients are
        constants that should fold into the caller's jaxpr.
        """
        z = jnp.asarray(z)
        w = 1.0 - z
        # Both panels are evaluated, so clamp each to its own domain; `where`
        # then discards the extrapolated one.
        z_lo = jnp.minimum(z, 0.5)
        w_hi = jnp.minimum(w, 0.5)

        lo = z_lo**self.a * _clenshaw(self.lo_coef, 4.0 * z_lo - 1.0) / self.a

        v = _clenshaw(self.hi_coef, 4.0 * w_hi - 1.0)
        hi = (
            self.const - jnp.log(w_hi) - v
            if self.b == 0
            else self.const - w_hi**self.b * v
        )
        return jnp.where(z <= 0.5, lo, hi)  # type: ignore[no-any-return]
