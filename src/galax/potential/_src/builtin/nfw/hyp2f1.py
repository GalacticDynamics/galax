r"""Native evaluation of the incomplete-beta-like function `gNFWPotential` needs.

`gNFWPotential`'s enclosed-mass and potential formulas both reduce to
evaluating

$$ B_z(a, b) = \int_0^z t^{a-1}(1-t)^{b-1}\,dt $$

(NIST DLMF 8.17.7), for exactly two parameter patterns: `a == 1` (any `b`),
or `b == 0` (any `a`). This used to be computed via
`tensorflow_probability.math.hypergeometric.hyp2f1_small_argument`, which
turned out to have three problems for this exact use:

- it returns `NaN` for the `b == 0` pattern once `z >= 0.9`
  (GalacticDynamics/galax#817);
- its custom gradient supports *only* `z`, silently returning no gradient
  for `a`/`b`/`c` (GalacticDynamics/galax#819);
- that custom gradient is implemented via `jax.custom_vjp`, which can never
  be forward-differentiated, so merely calling it anywhere in the
  computation graph made `gNFWPotential.hessian()`/`.tidal_tensor()` raise
  unconditionally (GalacticDynamics/galax#820).

`Bz_from_hyp2f1` below replaces it with a native, closed-form / rapidly
convergent implementation for exactly those two patterns (see `_Bz_a_eq_1`,
`_Bz_b_eq_0`), plus a `jax.custom_jvp` giving an exact O(1) `z`-derivative.
It does not implement the general `(a, b)` case; see `_Bz_from_hyp2f1_impl`'s
docstring.
"""

__all__ = ["Bz_from_hyp2f1"]

import functools as ft

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
from jax.custom_derivatives import SymbolicZero

import galax.potential.custom_types as gt

# ===================================================================
# a == 1: exact elementary closed form


def _Bz_a_eq_1(b: gt.FloatSz0, z: gt.BBtFloatSz0) -> gt.BBtFloatSz0:
    r"""$B_z(1, b) = \int_0^z (1-t)^{b-1}\,dt$, valid for any $b$ (elementary).

    $$ B_z(1, b) = \frac{1 - (1-z)^b}{b} \xrightarrow{b\to0} -\ln(1-z) $$
    """
    log1mz = jnp.log1p(-z)
    safe_b = jnp.where(b == 0, jnp.ones_like(b), b)  # avoid 0/0 below
    generic = -jnp.expm1(safe_b * log1mz) / safe_b
    return jnp.where(b == 0, -log1mz, generic)


# ===================================================================
# b == 0: derived from the defining integral int_0^z t^(a-1)/(1-t) dt.
# Two series, switched on z for fast & accurate convergence in both regimes.

_EULER_GAMMA = 0.5772156649015328606


def _Bz0_taylor_series(
    a: gt.FloatSz0, z: gt.BBtFloatSz0, n: int = 60
) -> gt.BBtFloatSz0:
    r"""$B_z(a, 0)$ via its defining series, accurate for small $z$.

    $$ B_z(a, 0) = \int_0^z \frac{t^{a-1}}{1-t}\,dt
    = \sum_{k=0}^\infty \frac{z^{a+k}}{a+k} $$
    """
    k = jnp.arange(n)
    return jnp.sum(z ** (a + k) / (a + k))


def _Bz0_log_series(a: gt.FloatSz0, z: gt.BBtFloatSz0, n: int = 60) -> gt.BBtFloatSz0:
    r"""$B_z(a, 0)$ via a series in $(1-z)$, accurate for $z$ near 1.

    $B_z(a,0)$ diverges logarithmically as $z \to 1$ (see
    `_Bz0_taylor_series`'s docstring), so this splits off that divergence
    and expands the (smooth) remainder in $w = 1-z$:

    $$ B_z(a, 0) = -\gamma_E - \psi(a) - \ln(w)
    - \sum_{m=0}^\infty \frac{(1-a)_{m+1}}{(m+1)^2\,m!}\,w^{m+1} $$

    Derived by substituting $t=1-u$, splitting off the $t\to1$ singularity as
    $\int_0^z\frac{1}{1-t}dt=-\ln(w)$, expanding the (now finite at $u=0$)
    remainder $(1-u)^{a-1}-1$ as a binomial series, and integrating
    term-by-term; the resulting $w\to1$ ($z\to0$) limit is Gauss's digamma
    integral $\psi(a) = -\gamma_E + \int_0^1\frac{1-t^{a-1}}{1-t}dt$. Verified
    numerically against direct quadrature to ~1e-11 for $a \in (0, 4]$,
    $z \in [0, 1)$ up to $z = 1 - 10^{-7}$.
    """
    w = 1 - z

    def accumulate_term(
        carry: tuple[gt.BBtFloatSz0, gt.BBtFloatSz0], m: gt.Sz0
    ) -> tuple[tuple[gt.BBtFloatSz0, gt.BBtFloatSz0], gt.BBtFloatSz0]:
        coeff, w_power = carry
        coeff = coeff * (m + 2 - a) * (m + 1) / (m + 2) ** 2
        w_power = w_power * w
        return (coeff, w_power), coeff * w_power

    (_, _), terms = jax.lax.scan(accumulate_term, (1 - a, w), jnp.arange(n - 1.0))
    series_sum = (1 - a) * w + jnp.sum(terms)
    return -_EULER_GAMMA - jsp.digamma(a) - series_sum - jnp.log(w)


def _Bz_b_eq_0(a: gt.FloatSz0, z: gt.BBtFloatSz0) -> gt.BBtFloatSz0:
    """$B_z(a, 0)$, switching series based on $z$ for fast, accurate convergence."""
    return jnp.where(z <= 0.5, _Bz0_taylor_series(a, z), _Bz0_log_series(a, z))


# ===================================================================
# Combined implementation + a custom derivative rule for speed


def _Bz_from_hyp2f1_impl(
    a: gt.FloatSz0, b: gt.FloatSz0, z: gt.BBtFloatSz0
) -> gt.BBtFloatSz0:
    r"""Incomplete beta function from hypergeometric function.

    $$ B_z(a, 0) = \frac{z^a}{a} \cdot {}_2F_1(a, 1 - b; a + 1; z) $$

    See NIST DLMF 8.17.7 @ https://dlmf.nist.gov/8.17

    Parameters
    ----------
    a, b
        The parameters of the incomplete beta function.
    z
        The value at which to evaluate the incomplete beta function.
        Must be in the range [0, 1].

    Notes
    -----
    `gNFWPotential` only ever calls this with `a == 1` (any `b`) or `b == 0`
    (any `a`), both of which have closed-form / rapidly-convergent native
    implementations above (`_Bz_a_eq_1`, `_Bz_b_eq_0`). This module does not
    implement the general `(a, b)` case; see this module's docstring for why
    (`tensorflow_probability`'s general implementation has three bugs for
    this use case).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax.scipy.special as jsp

    >>> a, b = 1.0, 2.0
    >>> z = jnp.array(0.5)

    >>> Bz_from_hyp2f1(a, b, z)
    Array(0.375, dtype=float64)

    >>> jsp.beta(a,b) * jsp.betainc(a, b, z)
    Array(0.375, dtype=float64)

    `Bz_from_hyp2f1` works for b = 0:

    >>> b = 0.0
    >>> Bz_from_hyp2f1(a, b, z)
    Array(0.69314718, dtype=float64)

    But `jsp.beta` does not work for b = 0:

    >>> jsp.beta(a,b) * jsp.betainc(a, b, z)
    Array(nan, dtype=float64)

    We can confirm that `Bz_from_hyp2f1` is correct by comparison when $b \sim 0$:

    >>> b = 1e-4
    >>> jsp.beta(a,b) * jsp.betainc(a, b, z)
    Array(0.69312316, dtype=float64)

    """
    is_a1 = a == 1.0
    return jnp.where(is_a1, _Bz_a_eq_1(b, z), _Bz_b_eq_0(a, z))


def _dBz_dz(a: gt.FloatSz0, b: gt.FloatSz0, z: gt.BBtFloatSz0) -> gt.BBtFloatSz0:
    r"""Exact $\partial B_z(a,b)/\partial z$, by the Leibniz integral rule.

    $B_z(a,b) = \int_0^z t^{a-1}(1-t)^{b-1}\,dt$, so its $z$-derivative is
    just the integrand evaluated at $z$ -- true regardless of which branch
    of `_Bz_from_hyp2f1_impl` computed the primal.
    """
    return z ** (a - 1) * (1 - z) ** (b - 1)


def _ab_tangent(
    a: gt.FloatSz0,
    b: gt.FloatSz0,
    z: gt.BBtFloatSz0,
    a_dot: gt.FloatSz0 | SymbolicZero,
    b_dot: gt.FloatSz0 | SymbolicZero,
    primal_out: gt.BBtFloatSz0,
) -> gt.BBtFloatSz0:
    """Compute the `a`/`b` contribution to `Bz_from_hyp2f1`'s tangent.

    Needed only if `gamma` itself is differentiated (`a`/`b` both depend on
    it); falls back to plain autodiff of `_Bz_from_hyp2f1_impl` since there's
    no cheap closed form for these, unlike `_dBz_dz`. Skips that autodiff
    call entirely when neither `a` nor `b` is being differentiated (the
    common case: `gradient`/`hessian`/`tidal_tensor` always differentiate
    with respect to position at fixed `gamma`).
    """
    a_is_zero = isinstance(a_dot, SymbolicZero)
    b_is_zero = isinstance(b_dot, SymbolicZero)
    if a_is_zero and b_is_zero:
        return jnp.zeros_like(primal_out)

    a_dot_v = jnp.zeros_like(a) if a_is_zero else a_dot
    b_dot_v = jnp.zeros_like(b) if b_is_zero else b_dot
    _, tangent = jax.jvp(
        lambda aa, bb: _Bz_from_hyp2f1_impl(aa, bb, z), (a, b), (a_dot_v, b_dot_v)
    )
    return tangent  # type: ignore[no-any-return]


@jax.custom_jvp
def Bz_from_hyp2f1(a: gt.FloatSz0, b: gt.FloatSz0, z: gt.BBtFloatSz0) -> gt.BBtFloatSz0:
    """See `_Bz_from_hyp2f1_impl` for the definition, examples and caveats.

    This wraps it with a `jax.custom_jvp` (which JAX can also transpose for
    reverse-mode -- a plain `jax.custom_vjp` cannot be forward-differentiated
    at all, which `jax.hessian`'s `jacfwd(jacrev(...))` needs): differentiating
    with respect to `z` costs O(1) (`_dBz_dz`) instead of autodiff-ing through
    `_Bz0_log_series`'s `jax.lax.scan`; see `_ab_tangent` for `a`/`b`.
    """
    return _Bz_from_hyp2f1_impl(a, b, z)


@ft.partial(Bz_from_hyp2f1.defjvp, symbolic_zeros=True)
def _Bz_from_hyp2f1_jvp(
    primals: tuple[gt.FloatSz0, gt.FloatSz0, gt.BBtFloatSz0],
    tangents: tuple[
        gt.FloatSz0 | SymbolicZero,
        gt.FloatSz0 | SymbolicZero,
        gt.BBtFloatSz0 | SymbolicZero,
    ],
) -> tuple[gt.BBtFloatSz0, gt.BBtFloatSz0]:
    a, b, z = primals
    a_dot, b_dot, z_dot = tangents

    primal_out = _Bz_from_hyp2f1_impl(a, b, z)
    tangent_out = _ab_tangent(a, b, z, a_dot, b_dot, primal_out)
    if not isinstance(z_dot, SymbolicZero):
        tangent_out = tangent_out + _dBz_dz(a, b, z) * z_dot
    return primal_out, tangent_out


Bz_from_hyp2f1 = jax.jit(Bz_from_hyp2f1)  # type: ignore[assignment]
