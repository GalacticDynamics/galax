"""Angular projection of a density onto real spherical harmonics.

Ported from ``bfeax.density_coeffs`` and ``bfeax.sph_harm``, with the
harmonics themselves supplied by `galax`'s own `iter_Ylm` rather than a second
Legendre recurrence.
"""

__all__: tuple[str, ...] = ()

import functools as ft
import math

from collections.abc import Callable
from jaxtyping import Array, Float
from typing import cast

import jax
import numpy as np

import quaxed.numpy as jnp

import galax.potential.custom_types as gt
from galax.potential._src.builtin.multipole import iter_Ylm

SQRT2: float = math.sqrt(2.0)
"""Real-harmonic normalization for ``m != 0``."""

SYMMETRIES: tuple[str, ...] = ("spherical", "axisymmetric", "triaxial")


def lm_keys(l_max: int, symmetry: str | None = None, /) -> tuple[tuple[int, int], ...]:
    r"""Return the ``(l, m)`` modes retained for a given symmetry.

    Modes that vanish identically under the stated symmetry are skipped,
    which reduces build cost and memory without affecting accuracy.

    - ``None``: every mode, :math:`-l \le m \le l`.
    - ``"spherical"``: ``(0, 0)`` only.
    - ``"axisymmetric"``: invariance under rotation about :math:`z` kills
      :math:`m \ne 0` and nothing else. Every :math:`l` with :math:`m = 0` is
      retained, odd :math:`l` included: axisymmetry says nothing about
      :math:`z \to -z`, so a density offset or lopsided along the axis keeps
      its odd-:math:`l` power. Callers who *also* have equatorial symmetry
      want ``"triaxial"`` (even :math:`l`, even :math:`m \ge 0`).
    - ``"triaxial"``: octant symmetry. :math:`x \to -x` kills odd-:math:`m`
      cosine terms, :math:`y \to -y` kills every sine term (:math:`m < 0`),
      and :math:`z \to -z` requires :math:`l + m` even — together, even
      :math:`l` and even :math:`m \ge 0`.

    Symmetry is never inferred from parameter values: that inference is only
    sound for concrete Python scalars and would silently degrade to ``None``
    under `jax.jit`, making the build depend on whether the caller traced.
    """
    if symmetry is None:
        return tuple((l, m) for l in range(l_max + 1) for m in range(-l, l + 1))
    if symmetry == "spherical":
        return ((0, 0),)
    if symmetry == "axisymmetric":
        # NB: upstream ``bfeax`` restricts this to even ``l``, conflating
        # axisymmetry with equatorial symmetry and silently dropping the
        # odd-``l`` power of any z-offset density. We keep every ``l``.
        return tuple((l, 0) for l in range(l_max + 1))
    if symmetry == "triaxial":
        return tuple((l, m) for l in range(0, l_max + 1, 2) for m in range(0, l + 1, 2))
    msg = f"Unknown symmetry {symmetry!r}. Use one of {SYMMETRIES} or None."
    raise ValueError(msg)


def real_ylm(
    l_max: int, keys: tuple[tuple[int, int], ...], uvec: gt.BtSz3, /
) -> Float[Array, "n_modes *batch"]:
    r"""Stack real orthonormal :math:`Y_{lm}` for ``keys``.

    `iter_Ylm` gives :math:`(\Re Y_l^m, \Im Y_l^m)` for :math:`m \ge 0`. The
    real orthonormal basis follows as

    .. math::

        Y_{l0} = \Re Y_l^0, \quad
        Y_{lm} = \sqrt{2}\,\Re Y_l^m, \quad
        Y_{l,-m} = \sqrt{2}\,\Im Y_l^m \quad (m > 0)

    matching ``bfeax.sph_harm.ylm_real`` to 7e-16; both carry the
    Condon-Shortley phase.

    ``uvec`` is a *Cartesian* unit direction, not :math:`(\theta, \phi)`.
    That is what keeps the result differentiable on the z-axis -- see
    `galax.potential._src.builtin.multipole.compute_Ylm`.
    """
    table = {(l, m): (cY, sY) for l, m, cY, sY in iter_Ylm(l_max, uvec)}
    out = []
    for l, m in keys:
        cY, sY = table[(l, abs(m))]
        if m == 0:
            out.append(cY)
        elif m > 0:
            out.append(SQRT2 * cY)
        else:
            out.append(SQRT2 * sY)
    return cast(Float[Array, "n_modes *batch"], jnp.stack(out))


def default_angular_resolution(l_max: int, /) -> tuple[int, int]:
    r"""Return the default angular quadrature resolution for ``l_max``.

    ``n_theta = 2 l_max + 2`` Gauss-Legendre nodes in :math:`\cos\theta` and
    ``n_phi = 4 l_max + 2`` uniform points in :math:`\phi`: roughly 2x the
    Nyquist minimum in each angle.

    ``bfeax``'s defaults (``l_max + 2``, ``2 l_max + 1``) are the minimum
    exact rule for a *band-limited* integrand -- one with no power above
    :math:`l_\max`. Real densities are not band-limited: a flattened halo has
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

    Pass ``n_theta`` / ``n_phi`` explicitly to `MultipoleProfilePotential`'s
    constructors to trade build cost against residual aliasing deliberately.
    """
    return 2 * l_max + 2, 4 * l_max + 2


def angular_grid(
    n_theta: int, n_phi: int, /
) -> tuple[Float[Array, "n_theta n_phi 3"], Float[Array, "n_theta n_phi"]]:
    r"""Gauss-Legendre :math:`\times` uniform-:math:`\phi` grid on the sphere.

    Returns Cartesian unit directions and quadrature weights such that
    :math:`\int f \, d\Omega \approx \sum_{ij} w_{ij} f(\hat{u}_{ij})`.

    The GL rule is applied in :math:`\cos\theta`, so its weights already carry
    the :math:`d(\cos\theta)` measure and no :math:`\sin\theta` Jacobian is
    needed. Nodes come from `numpy` at trace time -- they are static data, not
    traced values.
    """
    x, w = np.polynomial.legendre.leggauss(n_theta)
    cos_t = jnp.asarray(x, dtype=float)
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
    w_scaled = jnp.asarray(w, dtype=float)[:, None] * (2.0 * jnp.pi / n_phi)
    weights = jnp.broadcast_to(w_scaled, (n_theta, n_phi))
    return uvec, weights


@ft.partial(jax.jit, static_argnums=(0, 2, 3, 4, 5))
def project_density(
    rho_fn: Callable[[gt.BtSz3, gt.BBtSz0], Float[Array, "..."]],
    r_knots: Float[Array, "n_r"],
    l_max: int,
    keys: tuple[tuple[int, int], ...],
    n_theta: int,
    n_phi: int,
    t: gt.BBtSz0,
    /,
) -> Float[Array, "n_r n_modes"]:
    r"""Project a density onto real spherical harmonics on each shell.

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
