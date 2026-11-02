r"""Angular projection of a density onto real spherical harmonics.

The harmonics come from `iter_Ylm`, which evaluates them from a Cartesian
unit direction so they stay differentiable on the z-axis, rather than from a
second Legendre recurrence in :math:`(\theta, \phi)`.
"""

__all__: tuple[str, ...] = ()

import dataclasses
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
    if leaf.size == 0:
        out = jax.eval_shape(g, leaves)
        return jnp.zeros((*out.shape, 0), dtype=out.dtype)

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
    other input held constant (symbolic-zero tangents).
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

    ``self(0, Z, aux)`` is the value; ``self(k, Z, aux)`` is the derivative
    tensor with ``k`` trailing axes of length ``Z.shape[1]``. Each order is a
    `jax.custom_jvp` whose rule is the next order contracted with the input
    tangent, so ``f_leaves`` itself is never differentiated. Non-finite entries
    are repaired as `harmonic_coeffs` describes.
    """

    def __init__(
        self,
        f_leaves: Callable[..., Any],
        split: Callable[[Array], list[Array]],
        eps: float,
    ) -> None:
        self.f_leaves = f_leaves
        self.split = split
        self.eps = eps  # the coarsest precision among the parameters
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
        h = self.eps ** (1 / 3)
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
        # A zero coefficient has nothing to repair from unless a parameter it is
        # differentiated in is itself zero: the derivative at A = 0 is not in the
        # value. Any of the k derivative axes can be that parameter.
        at_zero = jnp.zeros((), dtype=bool)
        for a in range(k):
            shape = [1] * D.ndim
            shape[0], shape[D.ndim - k + a] = Z.shape
            at_zero = at_zero | (Z == 0).reshape(shape)
        need = bad & jnp.isfinite(v) & ((v != 0) | at_zero)
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
            # The caller discards rows with no non-zero derivative; select them
            # away on a primal mask, since their cotangent may be nan or inf.
            dead = ~jnp.any(D != 0, axis=-1)
            tangent = jnp.where(dead, 0.0, jnp.sum(D * dZ, axis=-1))
            # The primal goes through `G` so an outer reverse pass differentiates
            # it by this rule, not through `impl`'s raw (nan-bearing) derivatives.
            return self(k, Z, aux), tangent

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

    Derivatives are finite where ``rho_fn`` over- or underflows. They are formed
    explicitly, one column per traced float (and one for the radius), at every
    order, and a non-finite entry is repaired:

    * coefficient not finite: the whole row is zero;
    * coefficient zero: non-finite entries are zero, finite ones kept, except
      in the column of a parameter that is itself zero, which is repaired as below;
    * otherwise: a central difference of the next-lower derivative, or zero
      where that is not finite either (the radial build discards such samples).

    A column is differentiated with the others held constant, except between the
    elements of one array. Integers and concrete arrays get no column (a table
    passed as a `jax.jit` argument is traced, and gets one). Past
    ``_MAX_PARAMS`` columns ``rho_fn`` is differentiated as written.

    The central difference step is ``eps**(1/3)`` of the value, ``eps`` from the
    coarsest parameter dtype: accurate to ~1e-4 for an amplitude or a scale
    radius, ~2% for an exponent over tens of decades in radius, and ~1e-2 if
    ``rho_fn`` computes in lower precision than its parameters.
    """
    uvec, weights = angular_grid(n_theta, n_phi)
    Y = real_ylm(l_max, keys, uvec)  # (n_modes, n_theta, n_phi)

    # Traced floats (an amplitude, a scale radius) get derivative columns;
    # integers and concrete arrays (a lookup table) are held fixed.
    rho_cc, consts = _hoist_consts(rho_fn, r_knots[0] * uvec, t)
    args = [*consts, t]
    diff = [
        jnp.issubdtype(a.dtype, jnp.inexact) and isinstance(a, jax.core.Tracer)
        for a in consts
    ] + [jnp.issubdtype(t.dtype, jnp.inexact)]
    flat, unravel = ravel_pytree([a for a, d in zip(args, diff, strict=True) if d])
    n_r, n_p = r_knots.shape[0], flat.shape[0]
    # Rows are independent: a row's derivative has `n_p + 1` columns.
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
    traced = [r_knots, *(a for a, d in zip(args, diff, strict=True) if d)]
    eps = max(
        (
            float(jnp.finfo(a.dtype).eps)
            for a in traced
            if jnp.issubdtype(a.dtype, jnp.inexact)
        ),
        default=float(jnp.finfo(Z.dtype).eps),
    )
    out: Array = _DerivativeChain(f_leaves, split, eps)(0, Z, aux)
    return out


@dataclasses.dataclass(frozen=True)
class LogDensity:
    """A density given as ``log(rho)``, to be projected by `log_harmonic_coeffs`.

    `build_expansion` takes the density as a static argument and projects it
    with `log_harmonic_coeffs` where this is passed, `harmonic_coeffs`
    otherwise. Hashable by ``fn``, so it does not force a retrace.
    """

    fn: Callable[[gt.BtSz3, gt.BBtSz0], Float[Array, "..."]]
    """``(xyz, t) -> log(rho)``, broadcasting over the leading axes of ``xyz``."""


_LOG_MARGIN = 12.0
"""Natural-log head-room below the dtype's range that a row must leave.

It keeps ``exp(m) * (quadrature sum)`` finite, and it keeps a forward-mode
tangent, ``rho * d log(rho)``, finite for ``|d log(rho)|`` up to ``~e**12 / 3.5``
(``1 / A`` for ``A`` down to 1e-5, or a ``log(r / s)`` of 40 times an exponent):
rows closer to the maximum than that are dropped like an overflow, and the
build anchors above them. At 4 a forward-mode derivative in an amplitude of
1e-3 was ``inf``; at 18 the value error doubled for no gain.
"""


@ft.partial(jax.jit, static_argnums=(0, 2, 3, 4, 5))
def log_harmonic_coeffs(
    log_rho_fn: Callable[[gt.BtSz3, gt.BBtSz0], Float[Array, "..."]],
    r_knots: Float[Array, "n_r"],
    l_max: int,
    keys: tuple[tuple[int, int], ...],
    n_theta: int,
    n_phi: int,
    t: gt.BBtSz0,
    /,
) -> Float[Array, "n_r n_modes"]:
    r"""Return the harmonic coefficients of a density given as its logarithm.

    The same projection as `harmonic_coeffs`, formed as

    .. math::

        \rho_{lm}(r) = e^{m} \sum_{ij} w_{ij} Y_{lm}(\hat u_{ij})
                       e^{\ln\rho(r \hat u_{ij}) - m}

    with :math:`m` the largest :math:`\ln\rho` on the angular grid at that
    radius, held constant under differentiation. Every term of the sum is at
    most :math:`|w Y|`, so no derivative of it overflows, whatever the
    density: a derivative is the softmax-weighted sum of the derivatives of
    :math:`\ln\rho`, which the caller writes as a sum of logs. There is nothing
    to repair, at any order and in forward or reverse mode, and the cost is
    that of the plain forward pass.

    The density must be non-negative; ``-inf`` is a zero, and has to come from
    a ``where`` that selects it *around* the expression (``log(A) + 2 *
    log(clip(1 - u, 0))`` has a ``nan`` derivative where it is partly zero).
    A ``nan`` anywhere on a row makes the row ``nan``. A radius where
    :math:`\rho` over- or underflows has its coefficients set to ``inf`` or
    ``0`` (`build_expansion` discards the former), and gets no derivative.
    ``log_rho_fn`` is differentiated as written, where it is representable, so
    its derivative must be finite there: ``log(x) - log(s)`` rather than
    ``log(x / s)``, whose reverse pass multiplies :math:`1/x` and
    :math:`x/s^2`, each of which can overflow where their product does not.
    A row it cannot represent is evaluated at the middle radius instead, so
    its derivative is never formed.
    """
    uvec, weights = angular_grid(n_theta, n_phi)
    Yw = real_ylm(l_max, keys, uvec) * weights  # (n_modes, n_theta, n_phi)

    def on_grid(radii: Array) -> Array:
        return jax.vmap(lambda r: log_rho_fn(r * uvec, t))(radii)  # (n_r, nt, np)

    # Find the radii the density cannot represent from a pass that is not
    # differentiated, then evaluate the one that is somewhere benign instead.
    # The cotangent of such a row is zero, but `log_rho_fn`'s derivative at
    # 1e-19 (a `sqrt` of a square that underflows, say) can be infinite, and
    # `0 * inf` is `nan`.
    top = jnp.max(jax.lax.stop_gradient(on_grid(r_knots)), axis=(1, 2))
    info = jnp.finfo(top.dtype)
    over = top >= math.log(float(info.max)) - _LOG_MARGIN
    # `not >` so a -inf row, an empty one, is dead too.
    under = ~(top > math.log(float(info.tiny)) + _LOG_MARGIN)
    # `max` propagates a nan, but a nan row is neither over nor under: say so.
    isnan = jnp.isnan(top)
    over, under = over & ~isnan, under & ~isnan
    dead = over | under | isnan
    r_mid = jax.lax.stop_gradient(jnp.exp(jnp.mean(jnp.log(r_knots))))
    a = on_grid(jnp.where(dead, r_mid, r_knots))
    m = jnp.where(dead, 0.0, top)
    # Select the dead rows out of the sum: their cotangent may be nan or inf.
    a_live = jnp.where(dead[:, None, None], -jnp.inf, a)
    S = jnp.einsum("kij,rij->rk", Yw, jnp.exp(a_live - m[:, None, None]))
    out: Array = jnp.where(
        dead[:, None],
        jnp.where(isnan, jnp.nan, jnp.where(over, jnp.inf, 0.0))[:, None],
        jnp.exp(m)[:, None] * S,
    )
    return out
