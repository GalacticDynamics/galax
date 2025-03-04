"""galax: Galactic Dynamix in Jax."""

__all__: list[str] = []

from functools import partial
from typing import Any

import jax
from jaxtyping import Real
from plum import dispatch

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
from . import api
from .base import AbstractPotential
from .utils import parse_to_xyz_t
from galax.utils._shape import batched_shape, expand_arr_dims, expand_batch_dims

# =============================================================================
# Potential Energy


# ---------------------------
# Arrays


# TODO: consider "*#batch 1" for t
@dispatch  # special-case Array input to not return Quantity
@partial(jax.jit, inline=True)
def potential(
    pot: AbstractPotential, xyz: gt.XYZArrayLike, t: gt.BBtLikeSz0, /
) -> gt.BBtSz0:
    """Compute the potential energy at the given position(s).

    The position is in Cartesian coordinates and it and the time are assumed to
    be in the unit system of the potential.

    """
    xyz, t = parse_to_xyz_t(None, xyz, t)  # TODO: frame
    return pot._potential(xyz, t)  # noqa: SLF001


# TODO: consider "*#batch 1" for t
@dispatch  # special-case Array input to not return Quantity
@partial(jax.jit, inline=True)
def potential(
    pot: AbstractPotential, xyz: gt.XYZArrayLike, /, *, t: gt.BBtLikeSz0
) -> gt.BBtSz0:
    return api.potential(pot, xyz, t)


# ---------------------------


@dispatch
def potential(
    pot: AbstractPotential, tq: Any, /, *, t: Any = None
) -> Real[u.Quantity["specific energy"], "*#batch"]:
    """Compute from a q + t object."""
    return api.potential(pot, tq, t)


@dispatch
def potential(
    pot: AbstractPotential, q: Any, t: Any, /
) -> Real[u.Quantity["specific energy"], "*#batch"]:
    """Compute the potential energy at the given position(s)."""
    xyz, t = parse_to_xyz_t(None, q, t, ustrip=pot.units)  # TODO: frame
    phi = pot._potential(xyz, t)  # noqa: SLF001
    return u.Quantity(phi, pot.units["specific energy"])


# =============================================================================
# Gradient

# ---------------------------
# Arrays


# TODO: consider "*#batch 1" for t
@dispatch  # special-case Array input to not return Quantity
@partial(jax.jit, inline=True)
def gradient(
    pot: AbstractPotential, xyz: gt.XYZArrayLike, t: gt.BBtLikeSz0, /
) -> gt.BBtSz3:
    """Compute the gradient at the given position(s).

    The position is in Cartesian coordinates and it and the time are assumed to
    be in the unit system of the potential.

    """
    xyz, t = parse_to_xyz_t(None, xyz, t, dtype=float)  # TODO: frame
    return pot._gradient(xyz, t)  # noqa: SLF001


# TODO: consider "*#batch 1" for t
@dispatch  # special-case Array input to not return Quantity
@partial(jax.jit, inline=True)
def gradient(
    pot: AbstractPotential, xyz: gt.XYZArrayLike, /, *, t: gt.BBtLikeSz0
) -> gt.BBtSz3:
    return api.gradient(pot, xyz, t)


# ---------------------------
# Quantity


@dispatch
@partial(jax.jit, inline=True)
def gradient(
    pot: AbstractPotential, xyz: u.AbstractQuantity, /, *, t: u.AbstractQuantity
) -> Real[u.Quantity["acceleration"], "*#batch 3"]:
    """Compute from a q + t object."""
    xyz, t = parse_to_xyz_t(None, xyz, t, ustrip=pot.units, dtype=float)  # TODO: frame
    grad = pot._gradient(xyz, t)  # noqa: SLF001
    return u.Quantity.from_(grad, pot.units["acceleration"])


@dispatch
@partial(jax.jit, inline=True)
def gradient(
    pot: AbstractPotential, q: u.AbstractQuantity, t: u.AbstractQuantity, /
) -> Real[u.Quantity["acceleration"], "*#batch 3"]:
    """Compute the potential energy at the given position(s)."""
    xyz, t = parse_to_xyz_t(None, q, t, ustrip=pot.units, dtype=float)  # TODO: frame
    grad = pot._gradient(xyz, t)  # noqa: SLF001
    return u.Quantity.from_(grad, pot.units["acceleration"])


# ---------------------------


@dispatch
@partial(jax.jit, inline=True)
def gradient(
    pot: AbstractPotential, tq: Any, /, *, t: Any = None
) -> cx.vecs.CartesianAcc3D:
    """Compute from a q + t object."""
    xyz, t = parse_to_xyz_t(None, tq, t, ustrip=pot.units, dtype=float)  # TODO: frame
    grad = pot._gradient(xyz, t)  # noqa: SLF001
    return cx.vecs.CartesianAcc3D.from_(grad, pot.units["acceleration"])


@dispatch
def gradient(pot: AbstractPotential, q: Any, t: Any, /) -> cx.vecs.CartesianAcc3D:
    """Compute the potential energy at the given position(s)."""
    xyz, t = parse_to_xyz_t(None, q, t, ustrip=pot.units, dtype=float)  # TODO: frame
    grad = pot._gradient(xyz, t)  # noqa: SLF001
    return cx.vecs.CartesianAcc3D.from_(grad, pot.units["acceleration"])


# =============================================================================
# Laplacian

# ---------------------------
# Arrays


# TODO: consider "*#batch 1" for t
@dispatch  # special-case Array input to not return Quantity
@partial(jax.jit, inline=True)
def laplacian(
    pot: AbstractPotential, xyz: gt.XYZArrayLike, t: gt.BBtLikeSz0, /
) -> gt.BBtSz0:
    """Compute the laplacian at the given position(s).

    The position is in Cartesian coordinates and it and the time are assumed to
    be in the unit system of the potential.

    """
    xyz, t = parse_to_xyz_t(None, xyz, t, dtype=float)  # TODO: frame
    return pot._laplacian(xyz, t)  # noqa: SLF001


# TODO: consider "*#batch 1" for t
@dispatch  # special-case Array input to not return Quantity
@partial(jax.jit, inline=True)
def laplacian(
    pot: AbstractPotential, xyz: gt.XYZArrayLike, /, *, t: gt.BBtLikeSz0
) -> gt.BBtSz0:
    return api.laplacian(pot, xyz, t)


# ---------------------------


@dispatch
def laplacian(
    pot: AbstractPotential, tq: Any, /, *, t: Any = None
) -> Real[u.Quantity["frequency drift"], "*#batch"]:
    """Compute from a q + t object."""
    xyz, t = parse_to_xyz_t(None, tq, t, dtype=float, ustrip=pot.units)  # TODO: frame
    lapl = pot._laplacian(xyz, t)  # noqa: SLF001
    return u.Quantity(lapl, pot.units["frequency drift"])


@dispatch
def laplacian(
    pot: AbstractPotential, q: Any, t: Any, /
) -> Real[u.Quantity["frequency drift"], "*#batch"]:
    """Compute the laplacian energy at the given position(s)."""
    xyz, t = parse_to_xyz_t(None, q, t, dtype=float, ustrip=pot.units)  # TODO: frame
    lapl = pot._laplacian(xyz, t)  # noqa: SLF001
    return u.Quantity(lapl, pot.units["frequency drift"])


# =============================================================================
# Density

# ---------------------------
# Arrays


# TODO: consider "*#batch 1" for t
@dispatch  # special-case Array input to not return Quantity
@partial(jax.jit, inline=True)
def density(
    pot: AbstractPotential, xyz: gt.XYZArrayLike, t: gt.BBtLikeSz0, /
) -> gt.BBtSz0:
    """Compute the density at the given position(s).

    The position is in Cartesian coordinates and it and the time are assumed to
    be in the unit system of the potential.

    """
    xyz, t = parse_to_xyz_t(None, xyz, t, dtype=float)  # TODO: frame
    return pot._density(xyz, t)  # noqa: SLF001


# TODO: consider "*#batch 1" for t
@dispatch  # special-case Array input to not return Quantity
@partial(jax.jit, inline=True)
def density(
    pot: AbstractPotential, xyz: gt.XYZArrayLike, /, *, t: gt.BBtLikeSz0
) -> gt.BBtSz0:
    return api.density(pot, xyz, t)


# ---------------------------


@dispatch
def density(
    pot: AbstractPotential, tq: Any, /, *, t: Any = None
) -> Real[u.Quantity["mass density"], "*#batch"]:
    """Compute from a q + t object."""
    xyz, t = parse_to_xyz_t(None, tq, t, dtype=float, ustrip=pot.units)  # TODO: frame
    rho = pot._density(xyz, t)  # noqa: SLF001
    return u.Quantity(rho, pot.units["mass density"])


@dispatch
def density(
    pot: AbstractPotential, q: Any, t: Any, /
) -> Real[u.Quantity["mass density"], "*#batch"]:
    """Compute the density at the given position(s)."""
    xyz, t = parse_to_xyz_t(None, q, t, dtype=float, ustrip=pot.units)  # TODO: frame
    rho = pot._density(xyz, t)  # noqa: SLF001
    return u.Quantity(rho, pot.units["mass density"])


# =============================================================================
# Hessian

# ---------------------------
# Arrays


# TODO: consider "*#batch 1" for t
@dispatch  # special-case Array input to not return Quantity
@partial(jax.jit, inline=True)
def hessian(
    pot: AbstractPotential, xyz: gt.XYZArrayLike, t: gt.BBtLikeSz0, /
) -> gt.BBtSz33:
    """Compute the hessian at the given position(s).

    The position is in Cartesian coordinates and it and the time are assumed to
    be in the unit system of the potential.

    """
    xyz, t = parse_to_xyz_t(None, xyz, t, dtype=float)  # TODO: frame
    return pot._hessian(xyz, t)  # noqa: SLF001


# TODO: consider "*#batch 1" for t
@dispatch  # special-case Array input to not return Quantity
@partial(jax.jit, inline=True)
def hessian(
    pot: AbstractPotential, xyz: gt.XYZArrayLike, /, *, t: gt.BBtLikeSz0
) -> gt.BBtSz33:
    return api.hessian(pot, xyz, t)


# ---------------------------


@dispatch
def hessian(
    pot: AbstractPotential, tq: Any, /, *, t: Any = None
) -> Real[u.Quantity["frequency drift"], "*#batch 3 3"]:
    """Compute from a q + t object."""
    xyz, t = parse_to_xyz_t(None, tq, t, dtype=float, ustrip=pot.units)  # TODO: frame
    phi = pot._hessian(xyz, t)  # noqa: SLF001
    return u.Quantity(phi, pot.units["frequency drift"])


@dispatch
def hessian(
    pot: AbstractPotential, q: Any, t: Any, /
) -> Real[u.Quantity["frequency drift"], "*#batch 3 3"]:
    """Compute the potential energy at the given position(s)."""
    xyz, t = parse_to_xyz_t(None, q, t, dtype=float, ustrip=pot.units)  # TODO: frame
    phi = pot._hessian(xyz, t)  # noqa: SLF001
    return u.Quantity(phi, pot.units["frequency drift"])


# =============================================================================
# Acceleration


@dispatch
def acceleration(pot: AbstractPotential, /, *args: Any, **kwargs: Any) -> Any:
    """Compute the acceleration due to the potential at the given position(s).

    Parameters
    ----------
    pot : `~galax.potential.AbstractPotential`
        The potential to compute the acceleration of.
    *args : Any
        The phase-space + time position to compute the acceleration. See
        `~galax.potential.gradient` for more details.

    """
    return -api.gradient(pot, *args, **kwargs)


# =============================================================================
# Tidal Tensor


@dispatch
def tidal_tensor(
    pot: AbstractPotential, *args: Any, **kwargs: Any
) -> gt.BBtSz33 | gt.BBtQuSz33:
    """Compute the tidal tensor.

    See https://en.wikipedia.org/wiki/Tidal_tensor

    .. note::

        This is in cartesian coordinates with a Euclidean metric. Also, this
        isn't correct for GR.

    Parameters
    ----------
    pot : `~galax.potential.AbstractPotential`
        The potential to compute the tidal tensor of.
    *args, **kwargs : Any
        The arguments to pass to `galax.potential.hessian`.

    """
    J = api.hessian(pot, *args, **kwargs)  # (*batch, 3, 3)
    batch_shape, arr_shape = batched_shape(J, expect_ndim=2)  # (*batch), (3, 3)
    traced = (
        expand_batch_dims(jnp.eye(3), ndim=len(batch_shape))
        * expand_arr_dims(jnp.trace(J, axis1=-2, axis2=-1), ndim=len(arr_shape))
        / 3
    )
    return J - traced


# =============================================================================
# Local Circular Velocity


@dispatch
@partial(jax.jit)
def local_circular_velocity(
    pot: AbstractPotential, q: Any, t: Any, /
) -> gt.BBtSz0 | gt.BBtQuSz0:
    """Estimate the circular velocity at the given position."""
    xyz, t = parse_to_xyz_t(None, q, t, dtype=float)  # TODO: frame
    r = jnp.linalg.vector_norm(xyz, axis=-1)
    dPhi_dxyz = pot.gradient(xyz, t)
    dPhi_dr = jnp.sum(dPhi_dxyz * xyz / r[..., None], axis=-1)
    vcirc = jnp.sqrt(r * jnp.abs(dPhi_dr))
    return (
        u.Quantity.from_(vcirc, pot.units["velocity"])
        if u.quantity.is_any_quantity(xyz)
        else vcirc
    )


@dispatch
@partial(jax.jit)
def local_circular_velocity(
    pot: AbstractPotential, q: Any, /, *, t: Any = None
) -> gt.BBtSz0 | gt.BBtQuSz0:
    """Estimate the circular velocity at the given position."""
    return api.local_circular_velocity(pot, q, t)


# =============================================================================
# Radial derivative


@dispatch
@partial(jax.jit)
def dpotential_dr(pot: AbstractPotential, q: Any, t: Any, /) -> gt.BBtSz0 | gt.BBtQuSz0:
    xyz, t = parse_to_xyz_t(None, q, t, dtype=float)  # TODO: frame
    r_hat = cx.vecs.normalize_vector(xyz)
    grad = api.gradient(pot, xyz, t)
    dphi_dr = jnp.sum(grad * r_hat, axis=-1)
    return (
        u.Quantity.from_(dphi_dr, pot.units["acceleration"])
        if u.quantity.is_any_quantity(xyz)
        else dphi_dr
    )


@dispatch
@partial(jax.jit)
def dpotential_dr(
    pot: AbstractPotential, q: Any, /, *, t: Any = None
) -> gt.BBtSz0 | gt.BBtQuSz0:
    """Estimate the circular velocity at the given position."""
    return api.dpotential_dr(pot, q, t)


# =============================================================================
# 2nd Radial derivative


@dispatch
@partial(jax.jit)
def d2potential_dr2(
    pot: AbstractPotential, q: Any, t: Any, /
) -> gt.BBtSz0 | gt.BBtQuSz0:
    xyz, t = parse_to_xyz_t(None, q, t, dtype=float)  # TODO: frame
    rhat = cx.vecs.normalize_vector(xyz)
    # TODO: benchmark this vs the hessian approach commented out below
    # d2phi_dr2 = jnp.sum(jax.grad(pot.dpotential_dr)(xyz, t) * rhat)  # noqa: ERA001, E501
    H = pot.hessian(xyz, t)
    d2phi_dr2 = jnp.einsum("...i,...ij,...j -> ...", rhat, H, rhat)  # rhat · H · rhat
    return (
        u.Quantity.from_(d2phi_dr2, pot.units["frequency drift"])
        if u.quantity.is_any_quantity(xyz)
        else d2phi_dr2
    )


@dispatch
@partial(jax.jit)
def d2potential_dr2(
    pot: AbstractPotential, q: Any, /, *, t: Any = None
) -> gt.BBtSz0 | gt.BBtQuSz0:
    """Estimate the circular velocity at the given position."""
    return api.d2potential_dr2(pot, q, t)


# =============================================================================
# Enclosed mass


@dispatch
def spherical_mass_enclosed(
    pot: AbstractPotential, q: Any, t: Any, /
) -> gt.BBtSz0 | gt.BBtQuSz0:
    """Compute from `jax.Array`."""
    # Parse inputs
    q, t = parse_to_xyz_t(None, q, t, dtype=float)  # TODO: frame
    xyz, t = parse_to_xyz_t(None, q, t, ustrip=pot.units)
    # Compute mass
    r2 = jnp.sum(jnp.square(xyz), axis=-1)
    dPhi_dr = api.dpotential_dr(pot, xyz, t)
    m_encl = r2 * jnp.abs(dPhi_dr) / pot.constants["G"].value
    return (
        u.Quantity.from_(m_encl, pot.units["mass"])
        if u.quantity.is_any_quantity(q)
        else m_encl
    )


@dispatch
def spherical_mass_enclosed(
    pot: AbstractPotential, q: Any, /, *, t: Any = None
) -> gt.BBtSz0 | gt.BBtQuSz0:
    """Compute from `jax.Array`."""
    return api.spherical_mass_enclosed(pot, q, t)
