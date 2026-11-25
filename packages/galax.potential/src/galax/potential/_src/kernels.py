r"""Shared kernel factories for the per-potential modules.

The contract (see ``docs/potential_modules.rst``)
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
that do. Every returned function is `jax.jit`-ed. Keyword-only parameters of the
source function, such as the ``integration_order`` of a quadrature, stay
keyword-only on the built functions, are forwarded to the source and are static
under `jax.jit`. `spherical_gradient` and
`spherical_hessian` are written with ``axis=-1`` and ellipsis-safe ops, so they
also accept batches of points; that is not part of the contract.

`spherical_gradient` and `spherical_hessian` are tested in float32 with SI-scale
values when the parameters are traced arguments, for example when ``p`` is an
argument of the jitted function. With float32 and SI-scale values an
intermediate such as ``r**3`` can overflow. When the parameters are compile-time
constants closed over by `jax.jit`, XLA can fold the arithmetic into that form,
so the results are not guaranteed and some come out as 0. Use float64, or a unit
system in which typical values are of order 1, such as galactic. In float32
with SI-scale values the Hessian at exactly ``xyz = 0`` can also contain NaN,
because it is ``laplacian(p, 0) / 3`` and several potentials, NFW and Hernquist
among them, return NaN for ``laplacian(p, 0)`` there.
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

import inspect

from collections.abc import Callable
from typing import Any

import jax

import quaxed.numpy as jnp

import galax.potential.custom_types as gt
from .utils import RadialFn, safe_vector_norm


def _signed(source: Callable[..., Any], f: Callable[..., Any], /) -> Callable[..., Any]:
    """`jax.jit` of ``f`` that shows the parameters of ``source`` in ``help()``.

    The factories below return ``(p, x, *rest, **static)`` wrappers; the
    parameters are copied from ``source``. The positional ones become
    positional-only, since the wrapper's own parameter is ``x``, so that ``p``,
    ``r`` / ``xyz`` and ``t`` appear. The keyword-only ones of ``source``, such
    as an ``integration_order``, stay keyword-only, are forwarded to ``source``
    and are static under `jax.jit`.
    """
    sig = inspect.signature(source)
    kwonly = tuple(q.name for q in sig.parameters.values() if q.kind is q.KEYWORD_ONLY)
    f.__signature__ = sig.replace(  # type: ignore[attr-defined]
        parameters=[
            q if q.kind is q.KEYWORD_ONLY else q.replace(kind=q.POSITIONAL_ONLY)
            for q in sig.parameters.values()
        ],
        return_annotation=inspect.Signature.empty,
    )
    return jax.jit(f, static_argnames=kwonly)


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

    def mass_enclosed(p: gt.Params, r: gt.Sz0) -> gt.Sz0:
        """``mass_enclosed = r**2 * dpotential_dr(p, r) / p["G"]``."""
        return r**2 * dpotential_dr(p, r) / p["G"]

    return jax.jit(mass_enclosed)


# ==============================================================================
# Spherical assembly


def spherical_gradient(
    dpotential_dr: RadialFn, /
) -> Callable[[gt.Params, gt.BBtSz3], gt.BBtSz3]:
    r"""Cartesian gradient of a spherical potential: $\Phi'(r)\,\vec{x}/r$.

    At the origin the gradient is exactly zero. To differentiate at the origin
    use `spherical_hessian`: ``jax.jacfwd(gradient)`` at ``0`` is zero, because
    the outer `jnp.where` discards the tangent.
    """

    def gradient(p: gt.Params, xyz: gt.BBtSz3) -> gt.BBtSz3:
        """``gradient`` as ``dpotential_dr(p, r) * xyz / r``."""
        origin = jnp.sum(jnp.square(xyz), axis=-1) == 0
        # The inner `where` keeps the division away from r = 0 in the
        # primal, so no NaN or inf reaches a later gradient of this function.
        r = jnp.where(origin, 1.0, safe_vector_norm(xyz))
        # Times the unit vector: XLA rewrites (a / b) / c to a / (b * c), so a
        # `/ r` after a `G M / r**2` forms r**3, which overflows float32 in SI.
        out = dpotential_dr(p, r)[..., None] * (xyz / r[..., None])
        return jnp.where(origin[..., None], 0.0, out)  # type: ignore[no-any-return]

    return jax.jit(gradient)


def spherical_hessian(
    dpotential_dr: RadialFn, d2potential_dr2: RadialFn, laplacian: RadialFn, /
) -> Callable[[gt.Params, gt.BBtSz3], gt.BBtSz33]:
    r"""Cartesian Hessian of a spherical potential.

    $\Phi'/r\,\delta_{ij} + (\Phi'' - \Phi'/r)\,x_i x_j/r^2$. At the origin the
    Hessian is isotropic with trace ``laplacian(p, 0)``, so it is
    $\nabla^2\Phi(0)/3\,\delta_{ij}$: $\Phi''(0)\,I$ for a core, and $+\infty$
    on the diagonal (zero off it) for a cusp. It is never NaN, and a parameter
    gradient at an off-origin point stays finite for a cusp too.
    """

    def hessian(p: gt.Params, xyz: gt.BBtSz3) -> gt.BBtSz33:
        """``hessian`` as ``Phi'/r delta_ij + (Phi'' - Phi'/r) x_i x_j / r^2``."""
        origin = jnp.sum(jnp.square(xyz), axis=-1) == 0
        # The inner `where` keeps every radial kernel away from r = 0, where
        # shell-theorem profiles (M ~ r^3) underflow to 0 / 0.
        r1 = jnp.where(origin, 1.0, safe_vector_norm(xyz))
        # `* (1 / r1)`, not `/ r1`, so that no r**3 forms (see
        # `spherical_gradient`), and unit vectors in `outer`: a coefficient
        # divided by r**2 underflows float32 in SI.
        dphi_over_r = dpotential_dr(p, r1) * (1 / r1)
        coeff = jnp.where(origin, 0.0, d2potential_dr2(p, r1) - dphi_over_r)
        # r0 is 0 only at the origin, so a cusp's inf never reaches an
        # off-origin cotangent of `p`.
        r0 = jnp.where(origin, 0.0, r1)
        iso = jnp.where(origin, laplacian(p, r0) / 3, dphi_over_r)
        eye = jnp.eye(3, dtype=bool)
        rhat = xyz / r1[..., None]
        outer = rhat[..., :, None] * rhat[..., None, :]
        iso_eye = jnp.where(eye, iso[..., None, None], 0.0)
        return iso_eye + coeff[..., None, None] * outer  # type: ignore[no-any-return]

    return jax.jit(hessian)


# ==============================================================================
# Poisson


def poisson_laplacian(density: Callable[..., Any], /) -> Callable[..., Any]:
    r"""``laplacian`` from ``density`` by Poisson: $\nabla^2\Phi = 4\pi G\rho$."""

    def laplacian(p: gt.Params, x: Any, *rest: Any, **static: Any) -> Any:
        """``laplacian = 4 * pi * p["G"] * density(p, x, *rest, **static)``."""
        return 4 * jnp.pi * p["G"] * density(p, x, *rest, **static)

    return _signed(density, laplacian)


def poisson_density(laplacian: Callable[..., Any], /) -> Callable[..., Any]:
    r"""``density`` from ``laplacian`` by Poisson: $\rho = \nabla^2\Phi/(4\pi G)$."""

    def density(p: gt.Params, x: Any, *rest: Any, **static: Any) -> Any:
        """``density = laplacian(p, x, *rest, **static) / (4 * pi * p["G"])``."""
        return laplacian(p, x, *rest, **static) / (4 * jnp.pi * p["G"])

    return _signed(laplacian, density)


# ==============================================================================
# Cartesian autodiff


def cartesian_gradient(potential: Callable[..., Any], /) -> Callable[..., Any]:
    """`jax.grad` of ``potential(p, xyz, *rest)`` with respect to ``xyz``."""

    def gradient(p: gt.Params, xyz: Any, *rest: Any, **static: Any) -> Any:
        """``gradient`` as ``jax.grad(potential, argnums=1)``."""
        return jax.grad(potential, argnums=1)(p, xyz, *rest, **static)

    return _signed(potential, gradient)


def cartesian_hessian(potential: Callable[..., Any], /) -> Callable[..., Any]:
    """`jax.hessian` of ``potential(p, xyz, *rest)`` with respect to ``xyz``."""

    def hessian(p: gt.Params, xyz: Any, *rest: Any, **static: Any) -> Any:
        """``hessian`` as ``jax.hessian(potential, argnums=1)``."""
        return jax.hessian(potential, argnums=1)(p, xyz, *rest, **static)

    return _signed(potential, hessian)


def cartesian_laplacian(potential: Callable[..., Any], /) -> Callable[..., Any]:
    """Trace of `cartesian_hessian` of ``potential(p, xyz, *rest)``."""
    hess = cartesian_hessian(potential)

    def laplacian(p: gt.Params, xyz: Any, *rest: Any, **static: Any) -> Any:
        """``laplacian`` as the trace of ``jax.hessian(potential, argnums=1)``."""
        return jnp.trace(hess(p, xyz, *rest, **static))

    return _signed(potential, laplacian)
