"""galax: Galactic Dynamix in Jax."""

__all__: list[str] = []

from functools import partial
from typing import Any, TypeAlias

import jax
from jaxtyping import Shaped
from plum import convert, dispatch

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.typing as gt
from . import api
from .base import AbstractPotential
from .utils import parse_to_quantity
from galax.utils._shape import batched_shape, expand_arr_dims, expand_batch_dims

# TODO: shape -> batch
HessianVec: TypeAlias = Shaped[u.Quantity["1/s^2"], "*#shape 3 3"]

# =============================================================================
# Potential Energy


@dispatch
def potential(
    pot: AbstractPotential,
    pspt: gc.AbstractOnePhaseSpacePosition | cx.FourVector,
    /,
) -> u.Quantity["specific energy"]:
    """Compute from a PSP or 4vec.

    Parameters
    ----------
    pot : `~galax.potential.AbstractPotential`
    pspt : `~galax.coordinates.AbstractOnePhaseSpacePosition`
        The phase-space + time position to compute the value of the
        potential.

    """
    q = parse_to_quantity(pspt.q, units=pot.units)
    return pot._potential(q, pspt.t)  # noqa: SLF001


@dispatch
def potential(
    pot: AbstractPotential, q: Any, t: Any, /
) -> u.Quantity["specific energy"]:
    """Compute the potential energy at the given position(s).

    Parameters
    ----------
    pot : `~galax.potential.AbstractPotential`
        The potential to compute the value of.
    q : Any
        The position to compute the value of the potential. See
        `parse_to_quantity` for more details.
    t : Any
        The time at which to compute the value of the potential. See
        :meth:`unxt.Quantity.from_` for more details.

    """
    q = parse_to_quantity(q, unit=pot.units["length"])
    t = u.Quantity.from_(t, pot.units["time"])
    return pot._potential(q, t)  # noqa: SLF001


@dispatch
def potential(
    pot: AbstractPotential, q: Any, /, *, t: Any
) -> u.Quantity["specific energy"]:
    """Compute the potential energy when `t` is keyword-only."""
    return api.potential(pot, q, t)


# =============================================================================
# Gradient


@dispatch
def gradient(
    pot: AbstractPotential,
    pspt: gc.AbstractOnePhaseSpacePosition | cx.FourVector,
    /,
) -> cx.vecs.CartesianAcc3D:
    """Compute the gradient of the potential at the given position(s).

    Parameters
    ----------
    pot : `~galax.potential.AbstractPotential`
        The potential to compute the gradient of.
    pspt : `~galax.coordinates.AbstractOnePhaseSpacePosition`,
    positional-only
        The phase-space + time position to compute the gradient.

    """
    q = parse_to_quantity(pspt.q, units=pot.units["length"]).astype(float)
    grad = pot._gradient(q, pspt.t)  # noqa: SLF001
    return cx.vecs.CartesianAcc3D.from_(grad)


@dispatch
def gradient(pot: AbstractPotential, q: Any, t: Any, /) -> cx.vecs.CartesianAcc3D:
    """Compute the gradient of the potential at the given position(s).

    Parameters
    ----------
    pot : `~galax.potential.AbstractPotential`
        The potential to compute the gradient of.
    q : Any
        The position to compute the gradient of the potential. See
        `parse_to_quantity` for more details.
    t : Any
        The time at which to compute the gradient of the potential. See
        :meth:`unxt.Quantity.from_` for more details.

    """
    q = parse_to_quantity(q, unit=pot.units["length"]).astype(float)
    t = u.Quantity.from_(t, pot.units["time"])
    grad = pot._gradient(q, t)  # noqa: SLF001
    return cx.vecs.CartesianAcc3D.from_(grad)


@dispatch
def gradient(pot: AbstractPotential, q: Any, /, *, t: Any) -> cx.vecs.CartesianAcc3D:
    """Compute the gradient at the given position(s).

    Parameters
    ----------
    pot : `~galax.potential.AbstractPotential`
        The potential to compute the gradient of.
    q : Any
        The position to compute the gradient of the potential.
    t : Any, keyword-only
        The time at which to compute the gradient of the potential.

    """
    return api.gradient(pot, q, t)


# =============================================================================
# Laplacian


@dispatch
def laplacian(
    pot: AbstractPotential,
    pspt: gc.AbstractOnePhaseSpacePosition | cx.FourVector,
    /,
) -> u.Quantity["1/s^2"]:
    """Compute the laplacian of the potential at the given position(s).

    Parameters
    ----------
    pot : `~galax.potential.AbstractPotential`
        The potential to compute the laplacian of.
    pspt : `~galax.coordinates.AbstractOnePhaseSpacePosition`,
    positional-only
        The phase-space + time position to compute the laplacian.

    """
    q = parse_to_quantity(pspt.q, units=pot.units).astype(float)
    return pot._laplacian(q, pspt.t)  # noqa: SLF001


@dispatch
def laplacian(pot: AbstractPotential, q: Any, t: Any, /) -> u.Quantity["1/s^2"]:
    """Compute the laplacian of the potential at the given position(s).

    Parameters
    ----------
    pot : `~galax.potential.AbstractPotential`
        The potential to compute the laplacian of.
    q : Any
        The position to compute the laplacian of the potential. See
        `parse_to_quantity` for more details.
    t : Any
        The time at which to compute the laplacian of the potential.  If
        unitless (i.e. is an `~jax.Array`), it is assumed to be in the unit
        system of the potential.

    """
    q = parse_to_quantity(q, unit=pot.units["length"]).astype(float)
    t = u.Quantity.from_(t, pot.units["time"])
    return pot._laplacian(q, t)  # noqa: SLF001


@dispatch
def laplacian(pot: AbstractPotential, q: Any, /, *, t: Any) -> u.Quantity["1/s^2"]:
    """Compute the laplacian at the given position(s) when `t` is keyword-only."""
    return laplacian(pot, q, t)


# =============================================================================
# Density


@dispatch
def density(
    pot: AbstractPotential, pspt: gc.AbstractOnePhaseSpacePosition | cx.FourVector, /
) -> u.Quantity["mass density"]:
    """Compute the density at the given position(s).

    Parameters
    ----------
    pot : `~galax.potential.AbstractPotential`
        The potential to compute the density of.
    pspt : `~galax.coordinates.AbstractOnePhaseSpacePosition`
        The phase-space + time position to compute the density.

    """
    q = parse_to_quantity(pspt.q, units=pot.units)
    return pot._density(q, pspt.t)  # noqa: SLF001


@dispatch
def density(pot: AbstractPotential, q: Any, t: Any, /) -> u.Quantity["mass density"]:
    """Compute the density at the given position(s).

    Parameters
    ----------
    q : Any
        The position to compute the density of the potential.
        See `parse_to_quantity` for more details.
    t : Any
        The time at which to compute the density of the potential.
        See :meth:`unxt.Quantity.from_` for more details.

    """
    q = parse_to_quantity(q, unit=pot.units["length"])
    t = u.Quantity.from_(t, pot.units["time"])
    return pot._density(q, t)  # noqa: SLF001


@dispatch
def density(pot: AbstractPotential, q: Any, /, *, t: Any) -> u.Quantity["mass density"]:
    """Compute the density when `t` is keyword-only."""
    return density(pot, q, t)


# =============================================================================
# Hessian


@dispatch
def hessian(
    pot: AbstractPotential,
    pspt: gc.AbstractOnePhaseSpacePosition | cx.FourVector,
    /,
) -> gt.BtQuSz33:
    """Compute the hessian of the potential at the given position(s)."""
    q = parse_to_quantity(pspt.q, units=pot.units).astype(float)
    return pot._hessian(q, pspt.t)  # noqa: SLF001


@dispatch
def hessian(pot: AbstractPotential, q: Any, t: Any, /) -> HessianVec:
    """Compute the hessian of the potential at the given position(s).

    Parameters
    ----------
    pot : `~galax.potential.AbstractPotential`
        The potential to compute the hessian of.
    q : Any
        The position to compute the hessian of the potential. See
        `parse_to_quantity` for more details.
    t : Any
        The time at which to compute the hessian of the potential. See
        :meth:`~unxt.array.Quantity.from_` for more details.

    """
    q = parse_to_quantity(q, unit=pot.units["length"]).astype(float)
    t = u.Quantity.from_(t, pot.units["time"])
    return pot._hessian(q, t)  # noqa: SLF001


@dispatch
def hessian(pot: AbstractPotential, q: Any, /, *, t: Any) -> HessianVec:
    """Compute the hessian when `t` is keyword-only."""
    return api.hessian(pot, q, t)


# =============================================================================
# Acceleration


@dispatch
def acceleration(
    pot: AbstractPotential,
    /,
    *args: Any,  # defer to `gradient`
    **kwargs: Any,  # defer to `gradient`
) -> cx.vecs.CartesianAcc3D:
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
def tidal_tensor(pot: AbstractPotential, *args: Any, **kwargs: Any) -> gt.BtQuSz33:
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
        The arguments to pass to `~galax.potential.hessian`.

    """
    J = hessian(pot, *args, **kwargs)  # (*batch, 3, 3)
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
@partial(jax.jit, inline=True)
def local_circular_velocity(
    pot: AbstractPotential, x: gt.LengthSz3, /, t: gt.TimeSz0
) -> gt.BBtRealQuSz0:
    """Estimate the circular velocity at the given position.

    Parameters
    ----------
    pot : AbstractPotential
        The Potential.
    x : Quantity[float, (*batch, 3), "length"]
        Position(s) to estimate the circular velocity.
    t : Quantity[float, (), "time"]
        Time at which to compute the circular velocity.

    """
    r = jnp.linalg.vector_norm(x, axis=-1)
    dPhi_dxyz = convert(pot.gradient(x, t=t), u.Quantity)
    dPhi_dr = jnp.sum(dPhi_dxyz * x / r[..., None], axis=-1)
    return jnp.sqrt(r * jnp.abs(dPhi_dr))


@dispatch
@partial(jax.jit, inline=True)
def local_circular_velocity(
    pot: AbstractPotential, q: cx.vecs.AbstractPos3D, t: gt.TimeSz0, /
) -> gt.BBtRealQuSz0:
    """Estimate the circular velocity at the given position."""
    return api.local_circular_velocity(pot, convert(q, u.Quantity), t)


@dispatch
@partial(jax.jit, inline=True)
def local_circular_velocity(
    pot: AbstractPotential, w: gc.AbstractOnePhaseSpacePosition, /
) -> gt.BBtRealQuSz0:
    """Estimate the circular velocity at the given position."""
    return api.local_circular_velocity(pot, w.q, w.t)


@dispatch
@partial(jax.jit, inline=True)
def local_circular_velocity(
    pot: AbstractPotential, x: Any, /, *, t: Any
) -> gt.BBtRealQuSz0:
    """Compute the local circular velocity when `t` is keyword-only."""
    return api.local_circular_velocity(pot, x, t)
