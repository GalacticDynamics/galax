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

    Differentiating through it is safe where ``rho_fn`` overflows. A radius at
    which the density is not finite, or is zero everywhere while it is not at
    other radii, is one the caller of this function discards, so the
    coefficients there are treated as
    constants and ``rho_fn`` is differentiated at a representable radius
    instead. Otherwise the zero cotangent would meet an infinite derivative
    inside ``rho_fn`` (``0 * inf``) and every float32 gradient would be `nan`.
    The values are unchanged. A radius whose value is finite and non-zero is
    differentiated as usual, so a density whose own derivative overflows there
    still gives a non-finite gradient; see `MultipoleProfilePotential.from_density`.
    A density that is zero at *every* radius is differentiated as usual, so the
    gradient with respect to an amplitude of exactly zero is kept. A radius
    where it is zero while it is not elsewhere is masked even if a parameter
    would make it non-zero (a zero-amplitude component whose support does not
    overlap the rest has a zero gradient there).
    """
    uvec, weights = angular_grid(n_theta, n_phi)
    Y = real_ylm(l_max, keys, uvec)  # (n_modes, n_theta, n_phi)

    # `rho_fn` may close over the values being differentiated (an amplitude, a
    # scale radius). `custom_jvp` needs them as explicit arguments, so hoist
    # them out of the closure. The angular grid is passed explicitly too: under
    # `jit` it is itself a tracer, which a custom rule must not close over.
    rho_cc, consts = _hoist_consts(rho_fn, r_knots[0] * uvec, t)

    def evaluate(
        consts: tuple[Array, ...],
        t: gt.BBtSz0,
        r: Float[Array, "n_r"],
        grid: tuple[Array, Array, Array],
    ) -> tuple[Float[Array, "n_r n_modes"], Array]:
        uvec, Y, weights = grid

        def at_radius(
            r: Float[Array, ""],
        ) -> tuple[Float[Array, "n_modes"], Array, Array]:
            rho = rho_cc(r * uvec, t, *consts)  # (n_theta, n_phi)
            coeffs = jnp.einsum("kij,ij,ij->k", Y, rho, weights)
            return coeffs, jnp.all(jnp.isfinite(rho)), jnp.any(rho != 0)

        coeffs, finite, nonzero = jax.vmap(at_radius)(r)
        # A radius whose density is not finite, or is zero while others are
        # not, is one the caller's parameters cannot usefully change. If the
        # density is zero at every radius (an amplitude of exactly 0) nothing
        # was discarded, and the tangent is kept.
        return coeffs, finite & (nonzero | ~jnp.any(nonzero))

    @jax.custom_jvp
    def project(
        consts: tuple[Array, ...],
        t: gt.BBtSz0,
        r: Float[Array, "n_r"],
        grid: tuple[Array, Array, Array],
    ) -> Float[Array, "n_r n_modes"]:
        return evaluate(consts, t, r, grid)[0]

    @project.defjvp
    def project_jvp(primals: tuple[Any, ...], tangents: tuple[Any, ...]) -> Any:
        # The derivative of `rho_fn` at an overflowed sample is `inf` or `nan`,
        # and zeroing the result afterwards does not help: the zero cotangent
        # still meets that derivative (`0 * inf`) inside `rho_fn`. A `where`
        # on the output of `rho_fn` is therefore too late, and one inside it is
        # not ours to write. So the tangent is taken at a radius where the
        # density is representable, and masked off for the radii that were not.
        # That includes a density that underflowed to exactly zero while an
        # intermediate (`x**gamma`) overflowed: its value is a finite 0 but its
        # derivative is `0 * inf`. The value is unchanged either way.
        consts, t, r, grid = primals
        d_consts, d_t, d_r, _ = tangents
        out, ok = evaluate(consts, t, r, grid)

        def evaluate_rows(c: Any, t_: Any, r_: Any) -> Any:
            return evaluate(c, t_, r_, grid)[0]

        r_eval = jnp.where(ok, r, r[jnp.argmax(ok)])
        _, d_out = jax.jvp(
            evaluate_rows,
            (consts, t, r_eval),
            (
                d_consts,
                d_t,
                d_r if d_r.dtype == jax.dtypes.float0 else jnp.where(ok, d_r, 0.0),
            ),
        )
        return out, jnp.where(ok[:, None], d_out, 0.0)

    return project(tuple(consts), t, r_knots, (uvec, Y, weights))
