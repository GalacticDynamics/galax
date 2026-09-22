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
from typing import cast

import jax

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
    raise AssertionError  # pragma: no cover  # exhaustive over `Symmetry`


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
    return cast(Float[Array, "n_modes *batch"], jnp.stack(out))


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
    """
    uvec, weights = angular_grid(n_theta, n_phi)
    Y = real_ylm(l_max, keys, uvec)  # (n_modes, n_theta, n_phi)

    def at_radius(r: Float[Array, ""]) -> Float[Array, "n_modes"]:
        rho = rho_fn(r * uvec, t)  # (n_theta, n_phi)
        return cast(
            Float[Array, "n_modes"], jnp.einsum("kij,ij,ij->k", Y, rho, weights)
        )

    return jax.vmap(at_radius)(r_knots)
