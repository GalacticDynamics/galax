r"""Angular projection of a density onto real spherical harmonics.

The harmonics come from `iter_Ylm`, which evaluates them from a Cartesian
unit direction so they stay differentiable on the z-axis, rather than from a
second Legendre recurrence in :math:`(\theta, \phi)`.
"""

__all__: tuple[str, ...] = ()

import functools as ft
import math

from collections.abc import Callable
from jaxtyping import Array, Float
from typing import Any, assert_never, cast

import jax
from jax.extend.core import (  # type: ignore[attr-defined,unused-ignore]
    ClosedJaxpr,
    jaxpr_as_fun,
)
from jax.flatten_util import ravel_pytree

import quaxed.numpy as jnp

import galax.potential.custom_types as gt
from .ylm import iter_Ylm
from galax.potential._src.symmetry import Symmetry
from galax.potential._src.utils import gauss_legendre_nodes

SQRT2: float = math.sqrt(2.0)
"""Real-harmonic normalization for ``m != 0``."""


# TODO: revisit as a `plum` dispatched function on `Symmetry` once
# https://github.com/beartype/plum/issues/290 lands -- that would replace the
# `match` below with one implementation per member.
def lm_keys(
    l_max: int, symmetry: Symmetry | str | None = None, /
) -> tuple[tuple[int, int], ...]:
    r"""Return the ``(l, m)`` modes retained for a given symmetry.

    Modes that vanish identically under the stated symmetry are skipped,
    which reduces build cost and memory without affecting accuracy. See
    `Symmetry` for the set each member retains and why. ``symmetry`` may be a
    `Symmetry`, its plain string value, or `None` (an alias for
    `Symmetry.NONE`).

    Symmetry is never inferred from parameter values: that inference is only
    sound for concrete Python scalars and would silently degrade to
    `Symmetry.NONE` under `jax.jit`, making the build depend on whether the
    caller traced.
    """
    # `_missing_` maps `None` to `NONE`; `EnumMeta.__call__` is typed `str`.
    match Symmetry(symmetry):  # type: ignore[arg-type]
        case Symmetry.NONE:
            return tuple((l, m) for l in range(l_max + 1) for m in range(-l, l + 1))
        case Symmetry.SPHERICAL:
            return ((0, 0),)
        case Symmetry.ZROTATION:
            return tuple((l, 0) for l in range(l_max + 1))
        case Symmetry.ZROTATION_ZREFLECTION:
            return tuple((l, 0) for l in range(0, l_max + 1, 2))
        case Symmetry.PLANE_REFLECTION:
            return tuple(
                (l, m) for l in range(0, l_max + 1, 2) for m in range(0, l + 1, 2)
            )
        case _ as never:  # pragma: no cover
            assert_never(never)


def real_ylm(
    l_max: int, keys: tuple[tuple[int, int], ...], uvec: gt.BtSz3, /
) -> Float[Array, "n_modes *batch"]:
    r"""Stack real orthonormal :math:`Y_{lm}` for ``keys``.

    `iter_Ylm` gives :math:`(\Re Y_l^m, \Im Y_l^m)` for :math:`m \ge 0`. The
    real orthonormal basis follows as

    .. math::

        Y_{l0} = \Re Y_l^0, \quad
        Y_{lm} = \sqrt{2}\,(-1)^m \Re Y_l^m, \quad
        Y_{l,-m} = \sqrt{2}\,(-1)^m \Im Y_l^m \quad (m > 0)

    which is the standard real basis, as `scipy` and ``agama`` define it.
    Both factors matter and are separately pinned. The :math:`\sqrt{2}` is
    fixed by `test_real_ylm_is_orthonormal`: get it wrong and the Gram
    diagonal comes out at 2 or 1/2 rather than 1. The :math:`(-1)^m` is
    invisible to :math:`\Phi` and :math:`\rho` -- it enters the projection
    and the reconstruction alike and squares away -- so orthonormality
    cannot see it either, and it is pinned instead by
    `test_real_ylm_matches_the_standard_convention`, which compares against
    `scipy.special.sph_harm_y` term by term. Without that the sign could
    drift unnoticed and only surface as odd-:math:`m` disagreement against
    another code's coefficients.

    The complex :math:`Y_l^m` from `iter_Ylm` already carries the
    Condon-Shortley phase; the :math:`(-1)^m` here is the *additional* factor
    the real basis is conventionally defined with, not a second copy of it.

    ``uvec`` is a *Cartesian* unit direction, not :math:`(\theta, \phi)`.
    That is what keeps the result differentiable on the z-axis: a
    :math:`(\theta, \phi)` form has an ``atan2`` gradient of :math:`0/0`
    there for every :math:`m \ge 1`. See `iter_Ylm`, which evaluates the
    harmonic from the Cartesian direction for exactly this reason.
    """
    table = {(l, m): (cY, sY) for l, m, cY, sY in iter_Ylm(l_max, uvec)}
    out = []
    for l, m in keys:
        cY, sY = table[(l, abs(m))]
        if m == 0:
            out.append(cY)
        else:
            cs = SQRT2 if abs(m) % 2 == 0 else -SQRT2
            out.append(cs * (cY if m > 0 else sY))
    return cast('Float[Array, "n_modes *batch"]', jnp.stack(out))


def default_angular_resolution(l_max: int, /) -> tuple[int, int]:
    r"""Return the default angular quadrature resolution for ``l_max``.

    ``n_theta = 2 l_max + 2`` Gauss-Legendre nodes in :math:`\cos\theta` and
    ``n_phi = 4 l_max + 2`` uniform points in :math:`\phi`: roughly 2x the
    Nyquist minimum in each angle.

    The minimum exact rule for a *band-limited* integrand -- one with no
    power above :math:`l_\max` -- is (``l_max + 2``, ``2 l_max + 1``), and
    that is what a naive choice lands on. Real densities are not
    band-limited: a flattened halo has
    power at every :math:`l`, and under a minimal rule that power aliases into
    the retained modes rather than being discarded.

    Measured on a :math:`q = 0.4` flattened NFW with ``n_r = 256``, against a
    converged ``(120, 121)`` rule at the *same* ``l_max`` -- so this is
    aliasing alone, with truncation held fixed -- the maximum relative error
    in the potential is

    ======= =============== ==============
    l_max   minimal rule    this default
    ======= =============== ==============
    2       9.8e-2          2.0e-2
    4       2.1e-2          8.2e-4
    8       9.5e-4          1.3e-6
    ======= =============== ==============

    Acceleration tracks it within a factor of ~2. The cost is ~4x in the
    one-off projection at build time and nothing at evaluation, since the
    retained mode count is unchanged.

    `harmonic_coeffs` takes ``n_theta`` / ``n_phi`` directly, so a caller can
    trade build cost against residual aliasing deliberately rather than
    taking this default.
    """
    return 2 * l_max + 2, 4 * l_max + 2


def angular_grid(
    n_theta: int, n_phi: int, /
) -> tuple[Float[Array, "n_theta n_phi 3"], Float[Array, "n_theta n_phi"]]:
    r"""Gauss-Legendre :math:`\times` uniform-:math:`\phi` grid on the sphere.

    Returns Cartesian unit directions and quadrature weights such that
    :math:`\int f \, d\Omega \approx \sum_{ij} w_{ij} f(\hat{u}_{ij})`.

    The GL rule is applied in :math:`\cos\theta` over :math:`[-1, 1]`, so its
    weights already carry the :math:`d(\cos\theta)` measure and no
    :math:`\sin\theta` Jacobian is needed. Nodes come from
    `gauss_legendre_nodes`, shared with the Gaussian and triaxial-NFW
    potentials, and are static data computed at trace time from a static
    Python `int` -- there is nothing to differentiate through, so `numpy`'s
    ``leggauss`` not being JAX-differentiable is not a correctness problem.
    """
    cos_t, w = gauss_legendre_nodes(n_theta, (-1.0, 1.0))
    sin_t = jnp.sqrt(jnp.clip(1.0 - cos_t**2, 0.0))
    phi = jnp.arange(n_phi, dtype=float) * (2.0 * jnp.pi / n_phi)

    uvec = jnp.stack(
        [
            sin_t[:, None] * jnp.cos(phi)[None, :],
            sin_t[:, None] * jnp.sin(phi)[None, :],
            jnp.broadcast_to(cos_t[:, None], (n_theta, n_phi)),
        ],
        axis=-1,
    )
    w_scaled = w[:, None] * (2.0 * jnp.pi / n_phi)
    weights = jnp.broadcast_to(w_scaled, (n_theta, n_phi))
    return uvec, weights


def _hoist_consts(
    fun: Callable[..., Any], *args: Any
) -> tuple[Callable[..., Any], list[Any]]:
    """Like `jax.closure_convert`, but hoists every closed-over value.

    `jax.closure_convert` hoists only the inexact values being differentiated.
    A traced integer or bool stays closed over, which a `jax.custom_jvp` rule
    cannot capture.
    """
    closed = jax.make_jaxpr(fun)(*args)
    jaxpr = closed.jaxpr
    flat = jaxpr.replace(  # type: ignore[no-untyped-call]
        constvars=[], invars=[*jaxpr.constvars, *jaxpr.invars]
    )
    call = jaxpr_as_fun(ClosedJaxpr(flat, ()))

    def converted(xyz: Any, t: Any, *consts: Any) -> Any:
        (out,) = call(*consts, xyz, t)
        return out

    return converted, list(closed.consts)


_MAX_PARAMS = 128
"""Most differentiable scalars `harmonic_coeffs` forms derivative columns for."""

_FD_CHUNK = 32
"""Rows the central-difference fallback evaluates at a time."""


def _leaf_columns(g: Callable[[list[Array]], Any], leaves: list[Array], i: int) -> Any:
    """Return ``g``'s derivative in each element of ``leaves[i]``, on a new axis."""
    leaf = leaves[i]

    def gi(x: Array) -> Any:
        return g([*leaves[:i], x, *leaves[i + 1 :]])

    def tangent(basis: Array) -> Any:
        return jax.jvp(gi, (leaf,), (basis,))[1]

    if leaf.size == 1:
        return tangent(jnp.ones_like(leaf))[..., None]
    zero = jnp.zeros_like(leaf).reshape(-1)
    basis = jax.vmap(lambda e: zero.at[e].set(1).reshape(leaf.shape))(
        jnp.arange(leaf.size)
    )
    return jnp.moveaxis(jax.vmap(tangent)(basis), 0, -1)


def _columns(g: Callable[[list[Array]], Any], /) -> Callable[[list[Array]], Any]:
    """Append one axis to ``g``'s output: its derivative in each scalar input.

    Each array gets its own `jax.jvp`, batched over its elements, with every
    other input held constant, so their tangents are symbolic zeros.
    `jax.jacfwd` materialises them as zero vectors instead, and ``0 * inf``
    then spoils a column that does not depend on the overflowing input at all.
    """

    def dg(leaves: list[Array]) -> Any:
        cols = [_leaf_columns(g, leaves, i) for i in range(len(leaves))]
        return jnp.concat(cols, axis=-1)

    return dg


class _DerivativeChain:
    """The ``k``-th derivative of ``f_leaves`` for every row, for any ``k``.

    ``Z`` has one row per radius and one column per differentiable scalar, the
    radius last; ``split`` turns a row into the list of arrays that
    ``f_leaves(leaves, aux)`` takes.

    ``self(0, Z, aux)`` is the value. ``self(k, Z, aux)`` for ``k >= 1`` is the
    derivative tensor with ``k`` trailing axes of length ``Z.shape[1]``, and
    each order is a `jax.custom_jvp` whose rule is the next order contracted
    with the input tangent. Differentiating one therefore never differentiates
    ``f_leaves`` itself, so a derivative that is not finite is repaired here,
    once, and a zero cotangent never meets it (``0 * inf``). The orders are
    built lazily, one per differentiation.

    An entry that is not finite is repaired as follows:

    * the coefficient is not finite: the whole row is zero;
    * the coefficient is zero: the entry is zero (an underflowed product);
    * otherwise the entry is a central difference of the ``(k - 1)``-th
      derivative, or zero if that is not finite either.
    """

    def __init__(
        self, f_leaves: Callable[..., Any], split: Callable[[Array], list[Array]]
    ) -> None:
        self.f_leaves = f_leaves
        self.split = split
        self.cache: dict[int, Callable[..., Any]] = {}

    def __call__(self, k: int, Z: Array, aux: Any) -> Any:
        if k not in self.cache:
            self.cache[k] = self._order(k)
        return self.cache[k](Z, aux)

    def value(self, Z: Array, aux: Any) -> Any:
        return jax.vmap(lambda z: self.f_leaves(self.split(z), aux))(Z)

    def raw(self, k: int, Z: Array, aux: Any) -> Any:
        """Return the derivative as JAX forms it, `nan` and all."""
        g: Callable[..., Any] = ft.partial(self.f_leaves, aux=aux)
        for _ in range(k):
            g = _columns(g)
        return jax.vmap(lambda z: g(self.split(z)))(Z)

    def central(self, k: int, Z: Array, aux: Any) -> Any:
        """Central differences of the ``(k - 1)``-th derivative, per column.

        All ``2 * n_columns`` perturbations of every row go through one call, so
        the lower order is traced once, not once per column.
        """
        n_rows, n_cols = Z.shape
        h = float(jnp.finfo(Z.dtype).eps) ** (1 / 3)
        step = h * jnp.where(Z != 0, jnp.abs(Z), 1.0)  # (n_rows, n_cols)
        shift = jnp.eye(n_cols, dtype=Z.dtype)[:, None, :] * step.T[:, :, None]
        both = jnp.concat([Z + shift, Z - shift])  # (2 n_cols, n_rows, n_cols)
        out = self.impl(k - 1, both.reshape(-1, n_cols), aux)
        hi, lo = out.reshape(2, n_cols, n_rows, *out.shape[1:])
        step = step.T.reshape((n_cols, n_rows) + (1,) * (out.ndim - 1))
        return jnp.moveaxis((hi - lo) / (2 * step), 0, -1)

    def repair(self, k: int, Z: Array, aux: Any, need: Array, dtype: Any) -> Any:
        """``central`` on the rows where ``need`` is set, a chunk at a time.

        Few rows need it, so evaluating them all (or running both branches of a
        `lax.cond`, which is what `vmap` turns it into) would cost more than
        the derivative it repairs.
        """
        n_rows = Z.shape[0]
        chunk = min(n_rows, _FD_CHUNK)

        def body(state: tuple[Array, Array]) -> tuple[Array, Array]:
            todo, fd = state
            (idx,) = jnp.nonzero(todo, size=chunk, fill_value=n_rows)
            rows = self.central(k, Z[jnp.minimum(idx, n_rows - 1)], aux).astype(dtype)
            return todo.at[idx].set(False, mode="drop"), fd.at[idx].set(
                rows, mode="drop"
            )

        todo = jnp.any(need, axis=tuple(range(1, need.ndim)))
        fd0 = jnp.zeros(need.shape, dtype=dtype)
        _, fd = jax.lax.while_loop(lambda s: jnp.any(s[0]), body, (todo, fd0))
        return fd

    def impl(self, k: int, Z: Array, aux: Any) -> Any:
        """Return the order-``k`` tensor, repaired."""
        if k == 0:
            return self.value(Z, aux)
        D = self.raw(k, Z, aux)
        v = self.value(Z, aux)[(...,) + (None,) * k]
        bad = ~jnp.isfinite(D)
        need = bad & jnp.isfinite(v) & (v != 0)
        fd = self.repair(k, Z, aux, need, D.dtype)
        D = jnp.where(bad, jnp.where(need & jnp.isfinite(fd), fd, 0.0), D)
        return jnp.where(jnp.isfinite(v), D, 0.0)

    def _order(self, k: int) -> Callable[..., Any]:
        @jax.custom_jvp
        def G(Z: Array, aux: Any) -> Any:
            return self.impl(k, Z, aux)

        @G.defjvp
        def G_jvp(primals: tuple[Any, ...], tangents: tuple[Any, ...]) -> Any:
            Z, aux = primals
            D = self(k + 1, Z, aux)
            dZ = tangents[0].reshape((Z.shape[0], 1) + (1,) * k + (Z.shape[1],))
            # A row with no non-zero derivative is dead: the caller discarded
            # it, and its cotangent may be anything (`nan`, `inf`). Select it
            # away on a primal-only mask; `nan * 0` would not.
            dead = ~jnp.any(D != 0, axis=-1)
            tangent = jnp.where(dead, 0.0, jnp.sum(D * dZ, axis=-1))
            return self.impl(k, Z, aux), tangent

        return G


@ft.partial(jax.jit, static_argnums=(0, 2, 3, 4, 5))
def harmonic_coeffs(
    rho_fn: Callable[[gt.BtSz3, gt.BBtSz0], Float[Array, "..."]],
    r_knots: Float[Array, "n_r"],
    l_max: int,
    keys: tuple[tuple[int, int], ...],
    n_theta: int,
    n_phi: int,
    t: gt.BBtSz0,
    /,
) -> Float[Array, "n_r n_modes"]:
    r"""Return the real-spherical-harmonic coefficients of a density.

    .. math::

        \rho_{lm}(r) = \int \rho(r, \theta, \phi) Y_{lm}(\theta, \phi)
                       \, d\Omega

    ``rho_fn`` takes ``(xyz, t)`` with ``xyz`` of shape ``(..., 3)`` and must
    broadcast over the leading axes -- the `galax` ``_density`` signature. It
    is called once per radius on the full angular grid.

    Differentiating through it is safe where ``rho_fn`` over- or underflows.
    The derivatives are formed explicitly, one column per differentiable
    scalar (and one for the radius), and an entry that is not finite is
    repaired rather than propagated:

    * where the coefficient is not finite, the whole row is zero;
    * where the coefficient is zero, an entry that is not finite is zero (an
      underflowed ``0 * inf``), but a finite one is kept, so a zero-amplitude
      component has its gradient;
    * where the coefficient is finite and non-zero, an entry that is not
      finite is a central difference (``A / x**g`` differentiates through
      ``1 / x**(2 g)``, which overflows long before the density does), and zero
      if that is not finite either: a derivative the dtype cannot represent,
      at a sample the radial build discards.

    The same holds for every higher derivative. A column is differentiated
    with the others held constant, so an overflow in one does not reach the
    rest. The values are unchanged. With more than ``_MAX_PARAMS``
    differentiable scalars the columns are not formed and ``rho_fn`` is
    differentiated as written.
    """
    uvec, weights = angular_grid(n_theta, n_phi)
    Y = real_ylm(l_max, keys, uvec)  # (n_modes, n_theta, n_phi)

    # `rho_fn` may close over the values being differentiated (an amplitude, a
    # scale radius). Hoist them out of the closure so they are explicit inputs
    # to the rule, and split off the ones with no tangent.
    rho_cc, consts = _hoist_consts(rho_fn, r_knots[0] * uvec, t)
    args = [*consts, t]
    diff = [jnp.issubdtype(a.dtype, jnp.inexact) for a in args]
    flat, unravel = ravel_pytree([a for a, d in zip(args, diff, strict=True) if d])
    n_r, n_p = r_knots.shape[0], flat.shape[0]
    # Rows are independent: row `i` is a function of the parameters and of
    # `r_knots[i]` alone, so its derivative has `n_p + 1` columns, not `n_r`.
    Z = jnp.concat([jnp.broadcast_to(flat, (n_r, n_p)), r_knots[:, None]], axis=1)
    aux = ((uvec, Y, weights), [a for a, d in zip(args, diff, strict=True) if not d])

    def split(z: Array) -> list[Array]:
        return [*unravel(z[:-1]), z[-1]]

    def f_leaves(leaves: list[Array], aux: Any) -> Any:
        (uvec, Y, weights), fixed = aux
        *diff_leaves, r = leaves
        it_diff, it_fixed = iter(diff_leaves), iter(fixed)
        *cs, t_ = (next(it_diff) if d else next(it_fixed) for d in diff)
        rho = rho_cc(r * uvec, t_, *cs)  # (n_theta, n_phi)
        return jnp.einsum("kij,ij,ij->k", Y, rho, weights)

    if n_p > _MAX_PARAMS:
        plain: Array = jax.vmap(lambda z: f_leaves(split(z), aux))(Z)
        return plain
    out: Array = _DerivativeChain(f_leaves, split)(0, Z, aux)
    return out
