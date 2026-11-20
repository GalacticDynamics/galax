r"""Shared kernel factories for the per-potential modules.

The contract (see ``docs/superpowers/specs/2026-11-20-potential-modules-design.md``)
is that positions and parameters are bare arrays in the potential's unit system,
and ``p`` is the dict returned by ``pot.params(t)``. Kernels are single-point:
``r`` is a scalar, ``xyz`` has shape ``(3,)``, ``t`` is a scalar, and results are
scalars, ``(3,)`` or ``(3, 3)``. To batch, call `jax.vmap`.

======================  ============================  =============================
Potential symmetry      Radial quantities ``(p, r)``  Vector / tensor ``(p, xyz)``
======================  ============================  =============================
``Symmetry.SPHERICAL``  ``potential``, ``density``,   ``gradient``, ``hessian``
                        ``laplacian``,
                        ``mass_enclosed``,
                        ``dpotential_dr``,
                        ``d2potential_dr2``
anything else           none                          ``potential``, ``density``,
                                                      ``laplacian``, ``gradient``,
                                                      ``hessian``
======================  ============================  =============================

The factories here build the quantities that have no closed form from the ones
that do. Every returned function is `jax.jit`-ed. `spherical_gradient` and
`spherical_hessian` are written with ``axis=-1`` and ellipsis-safe ops, so they
also accept batches of points; that is not part of the contract.
"""

__all__ = (
    "cartesian_gradient",
    "cartesian_hessian",
    "cartesian_laplacian",
    "mass_from_dpotential",
    "poisson_density",
    "poisson_laplacian",
    "radial_gradient",
    "radial_second_derivative",
    "spherical_gradient",
    "spherical_hessian",
)

from collections.abc import Callable
from typing import Any

import jax

import quaxed.numpy as jnp

import galax.potential.custom_types as gt
from .utils import RadialFn, safe_vector_norm

# ==============================================================================
# Radial


def radial_gradient(potential: RadialFn, /) -> RadialFn:
    """`jax.grad` of ``potential(p, r)`` with respect to the scalar ``r``."""

    def dpotential_dr(p: gt.Params, r: gt.Sz0) -> gt.Sz0:
        """``dpotential_dr`` as ``jax.grad(potential, argnums=1)``."""
        return jax.grad(potential, argnums=1)(p, r)  # type: ignore[no-any-return]

    return jax.jit(dpotential_dr)


def radial_second_derivative(dpotential_dr: RadialFn, /) -> RadialFn:
    """`jax.grad` of ``dpotential_dr(p, r)`` with respect to the scalar ``r``."""

    def d2potential_dr2(p: gt.Params, r: gt.Sz0) -> gt.Sz0:
        """``d2potential_dr2`` as ``jax.grad(dpotential_dr, argnums=1)``."""
        return jax.grad(dpotential_dr, argnums=1)(p, r)  # type: ignore[no-any-return]

    return jax.jit(d2potential_dr2)


def mass_from_dpotential(dpotential_dr: RadialFn, /) -> RadialFn:
    """``mass_enclosed`` from Newton's shell theorem: ``r^2 Phi'(r) / G``."""

    def mass_enclosed(p: gt.Params, r: gt.BBtSz0) -> gt.BBtSz0:
        """``mass_enclosed = r**2 * dpotential_dr(p, r) / p["G"]``."""
        return r**2 * dpotential_dr(p, r) / p["G"]

    return jax.jit(mass_enclosed)


# ==============================================================================
# Spherical assembly


def spherical_gradient(
    dpotential_dr: RadialFn, /
) -> Callable[[gt.Params, gt.BBtSz3], gt.BBtSz3]:
    r"""Cartesian gradient of a spherical potential: $\Phi'(r)\,\vec{x}/r$.

    At the origin the gradient is exactly zero.
    """

    def gradient(p: gt.Params, xyz: gt.BBtSz3) -> gt.BBtSz3:
        """``gradient`` as ``dpotential_dr(p, r) * xyz / r``."""
        origin = jnp.sum(jnp.square(xyz), axis=-1) == 0
        # The inner `where` keeps the division away from r = 0 in the
        # primal, so no NaN or inf reaches a later gradient of this function.
        r = jnp.where(origin, 1.0, safe_vector_norm(xyz))
        out = (dpotential_dr(p, r) / r)[..., None] * xyz
        return jnp.where(origin[..., None], 0.0, out)  # type: ignore[no-any-return]

    return jax.jit(gradient)


def spherical_hessian(
    dpotential_dr: RadialFn, d2potential_dr2: RadialFn, /
) -> Callable[[gt.Params, gt.BBtSz3], Any]:
    r"""Cartesian Hessian of a spherical potential.

    $\Phi'/r\,\delta_{ij} + (\Phi'' - \Phi'/r)\,x_i x_j/r^2$. At the origin the
    first term is evaluated at the smallest normal radius, which for a cored
    profile is $\Phi''(0)$, and the second is dropped.
    """

    def hessian(p: gt.Params, xyz: gt.BBtSz3) -> Any:
        """``hessian`` as ``Phi'/r delta_ij + (Phi'' - Phi'/r) x_i x_j / r^2``."""
        origin = jnp.sum(jnp.square(xyz), axis=-1) == 0
        r = safe_vector_norm(xyz)
        # The second term is masked at the origin, with the inner `where`
        # keeping its radius away from 0 so that no inf * 0 = NaN arises.
        r1 = jnp.where(origin, 1.0, r)
        dphi_over_r = dpotential_dr(p, r1) / r1
        coeff = (d2potential_dr2(p, r1) - dphi_over_r) / r1**2
        coeff = jnp.where(origin, 0.0, coeff)
        outer = xyz[..., :, None] * xyz[..., None, :]
        eye = jnp.eye(3, dtype=xyz.dtype)
        iso = (dpotential_dr(p, r) / r)[..., None, None] * eye
        return iso + coeff[..., None, None] * outer

    return jax.jit(hessian)


# ==============================================================================
# Poisson


def poisson_laplacian(density: Callable[..., Any], /) -> Callable[..., Any]:
    r"""``laplacian`` from ``density`` by Poisson: $\nabla^2\Phi = 4\pi G\rho$."""

    def laplacian(p: gt.Params, x: Any, *rest: Any) -> Any:
        """``laplacian = 4 * pi * p["G"] * density(p, x, *rest)``."""
        return 4 * jnp.pi * p["G"] * density(p, x, *rest)

    return jax.jit(laplacian)


def poisson_density(laplacian: Callable[..., Any], /) -> Callable[..., Any]:
    r"""``density`` from ``laplacian`` by Poisson: $\rho = \nabla^2\Phi/(4\pi G)$."""

    def density(p: gt.Params, x: Any, *rest: Any) -> Any:
        """``density = laplacian(p, x, *rest) / (4 * pi * p["G"])``."""
        return laplacian(p, x, *rest) / (4 * jnp.pi * p["G"])

    return jax.jit(density)


# ==============================================================================
# Cartesian autodiff


def cartesian_gradient(potential: Callable[..., Any], /) -> Callable[..., Any]:
    """`jax.grad` of ``potential(p, xyz, *rest)`` with respect to ``xyz``."""

    def gradient(p: gt.Params, xyz: Any, *rest: Any) -> Any:
        """``gradient`` as ``jax.grad(potential, argnums=1)``."""
        return jax.grad(potential, argnums=1)(p, xyz, *rest)

    return jax.jit(gradient)


def cartesian_hessian(potential: Callable[..., Any], /) -> Callable[..., Any]:
    """`jax.hessian` of ``potential(p, xyz, *rest)`` with respect to ``xyz``."""

    def hessian(p: gt.Params, xyz: Any, *rest: Any) -> Any:
        """``hessian`` as ``jax.hessian(potential, argnums=1)``."""
        return jax.hessian(potential, argnums=1)(p, xyz, *rest)

    return jax.jit(hessian)


def cartesian_laplacian(potential: Callable[..., Any], /) -> Callable[..., Any]:
    """Trace of `cartesian_hessian` of ``potential(p, xyz, *rest)``."""
    hess = cartesian_hessian(potential)

    def laplacian(p: gt.Params, xyz: Any, *rest: Any) -> Any:
        """``laplacian`` as the trace of ``jax.hessian(potential, argnums=1)``."""
        return jnp.trace(hess(p, xyz, *rest))

    return jax.jit(laplacian)
