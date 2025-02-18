"""galax: Galactic Dynamix in Jax."""

__all__: list[str] = []

from functools import partial
from typing import Any

import equinox as eqx
import jax
from jaxtyping import Array, ArrayLike, ScalarLike
from plum import convert, dispatch

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import BareQuantity

import galax.coordinates as gc
import galax.typing as gt
from . import api
from .base import AbstractPotential
from .utils import parse_to_quantity_or_array
from galax.utils._shape import batched_shape, expand_arr_dims, expand_batch_dims
from galax.utils._unxt import AllowValue

frame = gc.frames.SimulationFrame()

# =============================================================================
# Potential Energy


@dispatch
@partial(jax.jit, inline=True)
def potential(pot: AbstractPotential, q: Any, /, *, t: Any) -> Any:
    """Compute the potential energy when `t` is keyword-only."""
    return api.potential(pot, q, t)


@dispatch
@partial(jax.jit, inline=True)
def potential(
    pot: AbstractPotential,
    wt: gc.AbstractPhaseSpaceCoordinate | cx.FourVector,
    /,
) -> u.Quantity["specific energy"]:
    """Compute from a q + t object."""
    q = parse_to_quantity_or_array(wt, dtype=None, units=pot.units)
    phi = pot._potential(q, wt.t.ustrip(pot.units["time"]))  # noqa: SLF001
    return u.Quantity(phi, pot.units["specific energy"])


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
    q = parse_to_quantity_or_array(q, dtype=None, unit=pot.units["length"])
    t = u.ustrip(AllowValue, pot.units["time"], t)
    phi = pot._potential(q, t)  # noqa: SLF001
    return u.Quantity(phi, pot.units["specific energy"])


@dispatch
@partial(jax.jit, inline=True)
def potential(
    pot: AbstractPotential, q: gt.BBtRealSz3, t: gt.BBtRealSz0 | ScalarLike | int, /
) -> gt.BBtRealSz0:
    """Compute the potential energy at the given position(s).

    Parameters
    ----------
    pot : `~galax.potential.AbstractPotential`
        The potential to compute the value of.
    q : Array[real, (*batch, 3)]
        The position to compute the value of the potential.
        Assumed to be in the unit system of the potential.
    t : Array[real, (*batch,)]
        The time at which to compute the value of the potential.
        Assumed to be in the unit system of the potential.

    """
    return pot._potential(q, t)  # noqa: SLF001


# =============================================================================
# Gradient


@dispatch
@partial(jax.jit, inline=True)
def gradient(pot: AbstractPotential, q: Any, /, *, t: Any) -> Any:
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


@dispatch
def gradient(
    pot: AbstractPotential, q: gt.BBtRealSz3, t: gt.BBtRealSz0, /
) -> gt.BBtRealSz3:
    """Compute the gradient of the potential at the given position(s).

    The Cartesian position and time are assumed to be in the unit system of the
    potential.

    """
    return pot._gradient(q.astype(float), t)  # noqa: SLF001


@dispatch
def gradient(pot: AbstractPotential, q: gt.BBtRealQuSz3, t: Any, /) -> gt.BBtRealQuSz3:
    """Compute the gradient of the potential at the given position(s).

    The position is assumed to be Cartesian.

    """
    q = parse_to_quantity_or_array(q, dtype=float, unit=pot.units["length"])
    t = u.ustrip(AllowValue, pot.units["time"], t)
    grad = pot._gradient(q, t)  # noqa: SLF001
    return u.Quantity(grad, pot.units["acceleration"])


@dispatch
def gradient(
    pot: AbstractPotential,
    wt: gc.AbstractPhaseSpaceCoordinate | cx.FourVector,
    /,
) -> cx.vecs.CartesianAcc3D:
    """Compute the gradient of the potential at the given coordinate(s)."""
    q = parse_to_quantity_or_array(wt, dtype=float, units=pot.units["length"])
    grad = pot._gradient(q, wt.t)  # noqa: SLF001
    return cx.vecs.CartesianAcc3D.from_(grad, pot.units["acceleration"])


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
    q = parse_to_quantity_or_array(q, dtype=float, unit=pot.units["length"])
    t = u.ustrip(AllowValue, pot.units["time"], t)
    grad = pot._gradient(q, t)  # noqa: SLF001
    return cx.vecs.CartesianAcc3D.from_(grad, pot.units["acceleration"])


# =============================================================================
# Laplacian


@dispatch
def laplacian(pot: AbstractPotential, q: Any, /, *, t: Any) -> Any:
    """Compute the laplacian at the given position(s) when `t` is keyword-only."""
    return laplacian(pot, q, t)


@dispatch
def laplacian(
    pot: AbstractPotential,
    wt: gc.AbstractPhaseSpaceCoordinate | cx.FourVector,
    /,
) -> u.Quantity["1/s^2"]:
    """Compute the laplacian of the potential at the given coordinate(s)."""
    q = parse_to_quantity_or_array(wt, dtype=float, units=pot.units)
    return u.Quantity(pot._laplacian(q, wt.t), pot.units["frequency drift"])  # noqa: SLF001


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
    q = parse_to_quantity_or_array(q, dtype=float, unit=pot.units["length"])
    t = u.ustrip(AllowValue, pot.units["time"], t)
    return u.Quantity(pot._laplacian(q, t), pot.units["frequency drift"])  # noqa: SLF001


@dispatch
def laplacian(
    pot: AbstractPotential, q: gt.BBtRealSz3, t: gt.BBtRealSz0, /
) -> gt.BtRealSz0:
    """Compute the laplacian of the potential at the given position(s).

    Parameters
    ----------
    pot : `~galax.potential.AbstractPotential`
        The potential to compute the laplacian of.
    q : Any
        The position to compute the laplacian of the potential.
        Assumed to be in the units of the potential.
    t : Any
        The time at which to compute the laplacian of the potential.
        Assumed to be in the units of the potential.

    """
    return pot._laplacian(q, t)  # noqa: SLF001


# =============================================================================
# Density


@dispatch
def density(
    pot: AbstractPotential, q: Any, /, *, t: Any
) -> u.Quantity["mass density"] | Array:
    """Compute the density when `t` is keyword-only."""
    return density(pot, q, t)


@dispatch
def density(
    pot: AbstractPotential, wt: gc.AbstractPhaseSpaceCoordinate | cx.FourVector, /
) -> u.Quantity["mass density"]:
    """Compute the density at the given coordinate(s)."""
    q = parse_to_quantity_or_array(wt.q, units=pot.units)
    rho = pot._density(q, wt.t)  # noqa: SLF001
    return u.Quantity(rho, pot.units["mass density"])


@dispatch
def density(pot: AbstractPotential, q: Any, t: Any, /) -> u.Quantity["mass density"]:
    """Compute the mass density at the given position(s).

    Parameters
    ----------
    q : Any
        The position to compute the density of the potential.
        See `parse_to_quantity` for more details.
    t : Any
        The time at which to compute the density of the potential.
        See :meth:`unxt.Quantity.from_` for more details.

    """
    q = parse_to_quantity_or_array(q, unit=pot.units["length"])
    t = u.ustrip(AllowValue, pot.units["time"], t)
    rho = pot._density(q, t)  # noqa: SLF001
    return u.Quantity(rho, pot.units["mass density"])


@dispatch
def density(pot: AbstractPotential, q: gt.BBtSz3, t: gt.BBtSz0, /) -> gt.BtRealSz0:
    """Compute the mass density at the given position(s).

    Parameters
    ----------
    q : Array[real, (*batch, 3)]
        The position to compute the density of the potential.
        Assumed to be in the units of the potential.
    t : Array[real, (*batch,)]
        The time at which to compute the density of the potential.
        Assumed to be in the units of the potential.

    """
    return pot._density(q, t)  # noqa: SLF001


# =============================================================================
# Hessian


@dispatch
def hessian(pot: AbstractPotential, q: Any, /, *, t: Any) -> Any:
    """Compute the hessian when `t` is keyword-only."""
    return api.hessian(pot, q, t)


@dispatch
def hessian(
    pot: AbstractPotential, q: gt.BBtRealSz3, t: gt.BBtRealSz0, /
) -> gt.BBtSz33:
    """Compute the hessian of the potential at the given position(s).

    The position is in Cartesian coordinates and it and the time are assumed to
    be in the unit system of the potential.

    """
    return pot._hessian(q.astype(float), t)  # noqa: SLF001


@dispatch
def hessian(
    pot: AbstractPotential, q: gt.BBtRealQuSz3, t: gt.BBtRealQuSz0, /
) -> gt.BtQuSz33:
    """Compute the hessian of the potential at the given position(s).

    The position is in Cartesian coordinates.

    """
    q = parse_to_quantity_or_array(q, dtype=float, unit=pot.units["length"])
    t = u.ustrip(AllowValue, pot.units["time"], t)
    hess = pot._hessian(q, t)  # noqa: SLF001
    return u.Quantity(hess, pot.units["frequency drift"])


@dispatch
def hessian(pot: AbstractPotential, q: Any, t: Any, /) -> gt.BtQuSz33:
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
    q = parse_to_quantity_or_array(q, dtype=float, unit=pot.units["length"])
    t = u.ustrip(AllowValue, pot.units["time"], t)
    hess = pot._hessian(q, t)  # noqa: SLF001
    return u.Quantity(hess, pot.units["frequency drift"])


@dispatch
def hessian(
    pot: AbstractPotential,
    wt: gc.AbstractPhaseSpaceCoordinate | cx.FourVector,
    /,
) -> gt.BtQuSz33:
    """Compute the hessian of the potential at the given position(s)."""
    q = parse_to_quantity_or_array(wt, dtype=float, units=pot.units)
    hess = pot._hessian(q, wt.t)  # noqa: SLF001
    return u.Quantity(hess, pot.units["frequency drift"])


# =============================================================================
# Acceleration


@dispatch
def acceleration(
    pot: AbstractPotential,
    /,
    *args: Any,  # defer to `gradient`
    **kwargs: Any,  # defer to `gradient`
) -> Any:
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
    pot: AbstractPotential, w: gc.AbstractPhaseSpaceCoordinate, /
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


# =============================================================================
# Radial derivative


@dispatch
@partial(jax.jit)
def dpotential_dr(
    pot: AbstractPotential, xyz: gt.BBtRealSz3, t: gt.BBtRealSz0, /
) -> gt.BBtRealSz0:
    """Compute the radial derivative of the potential at the given position."""
    r_hat: Array = cx.vecs.normalize_vector(xyz)
    grad = api.gradient(pot, xyz, t)
    return jnp.sum(grad * r_hat, axis=-1)


@dispatch
@partial(jax.jit)
def dpotential_dr(
    pot: AbstractPotential, x: gt.BBtRealQuSz3, t: gt.BBtRealQuSz0, /
) -> gt.BBtRealQuSz0:
    """Compute the radial derivative of the potential at the given position."""
    x, t = jnp.asarray(x), jnp.asarray(t)
    r_hat = cx.vecs.normalize_vector(x)
    grad = api.gradient(pot, x, t)
    return jnp.sum(grad * r_hat, axis=-1)


@dispatch
@partial(jax.jit)
def dpotential_dr(
    pot: AbstractPotential,
    x: cx.vecs.AbstractPos3D,
    t: u.AbstractQuantity | ArrayLike,
    /,
) -> gt.BBtRealQuSz0:
    return api.dpotential_dr(pot, convert(x, u.Quantity), t)


@dispatch
@partial(jax.jit)
def dpotential_dr(
    pot: AbstractPotential,
    w: cx.vecs.FourVector | gc.AbstractPhaseSpaceCoordinate,
) -> gt.BBtRealQuSz0:
    return api.dpotential_dr(pot, w.q, w.t)


@dispatch
def dpotential_dr(pot: AbstractPotential, x: Any, /, *, t: Any) -> gt.BBtRealQuSz0:
    """Compute the radial derivative of the potential when `t` is keyword-only."""
    return api.dpotential_dr(pot, x, t)


# =============================================================================
# 2nd Radial derivative


@dispatch
@partial(jax.jit)
def d2potential_dr2(
    pot: AbstractPotential, w: Any, /, *, t: Any
) -> gt.BBtRealQuSz0 | gt.BBtRealSz0:
    """Compute when `t` is keyword-only."""
    return api.d2potential_dr2(pot, w, t)


@dispatch
@partial(jax.jit)
def d2potential_dr2(
    pot: AbstractPotential, q: Any, t: Any, /
) -> gt.BBtRealQuSz0 | gt.BBtRealSz0:
    """Compute the second derivative of the potential at the position.

    Parameters
    ----------
    pot : `galax.potential.AbstractPotential`
        The gravitational potential.
    x: Quantity[Any, (*batch, 3,), 'length']
        3d position (x, y, z) in [kpc]
    t: Quantity[Any, (*#batch,), 'time']
        Time in [Myr]

    """
    xyz = parse_to_quantity_or_array(q, unit=pot.units["length"])
    rhat = cx.vecs.normalize_vector(xyz)
    H = pot.hessian(xyz, t)
    # vectorized dot product of rhat · H · rhat
    return jnp.einsum("...i,...ij,...j -> ...", rhat, H, rhat)


@dispatch
@partial(jax.jit, inline=True)
def d2potential_dr2(pot: AbstractPotential, w: cx.Space, t: Any, /) -> gt.BBtRealQuSz0:
    """Compute the second derivative of the potential at the position."""
    q3 = w["length"]
    q3 = eqx.error_if(
        q3,
        isinstance(q3, cx.vecs.FourVector)
        and jnp.logical_not(jnp.array_equal(q3.t, t)),
        "Got a FourVector, but t is not the same as the FourVector time",
    )
    return api.d2potential_dr2(pot, q3, t)


@dispatch
@partial(jax.jit, inline=True)
def d2potential_dr2(pot: AbstractPotential, w: cx.Space, /) -> gt.BBtRealQuSz0:
    """Compute the second derivative of the potential at the position."""
    q4 = w["length"]
    q4 = eqx.error_if(
        q4, not isinstance(q4, cx.vecs.FourVector), "Expected a FourVector"
    )
    return api.d2potential_dr2(pot, q4.q, q4.t)


@dispatch
@partial(jax.jit, inline=True)
def d2potential_dr2(
    pot: AbstractPotential, w: cx.frames.AbstractCoordinate, /
) -> gt.BBtRealQuSz0:
    # TODO: deal with frames
    return api.d2potential_dr2(pot, w.data)


@dispatch
@partial(jax.jit, inline=True)
def d2potential_dr2(
    pot: AbstractPotential, w: gc.PhaseSpacePosition, t: Any, /
) -> gt.BBtRealQuSz0:
    # TODO: deal with frames
    return api.d2potential_dr2(pot, w.q, t)


@dispatch.multi(
    (AbstractPotential, gc.AbstractPhaseSpaceCoordinate),
    (AbstractPotential, cx.vecs.FourVector),
)
@partial(jax.jit, inline=True)
def d2potential_dr2(
    pot: AbstractPotential, w: gc.AbstractPhaseSpaceCoordinate | cx.vecs.FourVector, /
) -> gt.BBtRealQuSz0:
    # TODO: deal with frames
    return api.d2potential_dr2(pot, w.q, w.t)


# =============================================================================
# Enclosed mass


@dispatch
def spherical_mass_enclosed(
    pot: AbstractPotential, x: gt.BBtRealSz3, t: gt.BBtRealSz0, /
) -> gt.BBtRealSz0:
    """Compute from `jax.Array`."""
    r2 = jnp.sum(jnp.square(x), axis=-1)
    dPhi_dr = api.dpotential_dr(pot, x, t)
    return r2 * jnp.abs(dPhi_dr) / pot.constants["G"].value


@dispatch
def spherical_mass_enclosed(
    pot: AbstractPotential, x: gt.BBtQuSz3, t: gt.BBtRealQuSz0, /
) -> gt.BBtRealQuSz0:
    """Compute from `unxt.Quantity`."""
    r2 = jnp.sum(jnp.square(x), axis=-1)
    dPhi_dr = api.dpotential_dr(pot, x, t)
    return r2 * jnp.abs(dPhi_dr) / pot.constants["G"]


@dispatch
def spherical_mass_enclosed(
    pot: AbstractPotential, q: cx.vecs.AbstractPos3D, t: gt.BBtRealQuSz0, /
) -> gt.BBtRealQuSz0:
    """Compute from `coordinax.vecs.AbstractPos3D`."""
    return api.spherical_mass_enclosed(pot, convert(q, BareQuantity), t)


@dispatch
def spherical_mass_enclosed(
    pot: AbstractPotential, qt: cx.vecs.FourVector, /
) -> gt.BBtRealQuSz0:
    """Compute from `coordinax.vecs.AbstractPos3D`."""
    return api.spherical_mass_enclosed(pot, qt.q, qt.t)


@dispatch
def spherical_mass_enclosed(
    pot: AbstractPotential, qt: cx.vecs.FourVector, t: gt.BBtRealQuSz0, /
) -> gt.BBtRealQuSz0:
    """Compute from `coordinax.vecs.AbstractPos3D`."""
    t = eqx.error_if(
        qt.t,
        jnp.logical_not(jnp.array_equal(qt.t, t)),
        msg="`qt.t` and `t` are not equal.",
    )
    return api.spherical_mass_enclosed(pot, qt.q, t)


@dispatch
def spherical_mass_enclosed(
    pot: AbstractPotential, space: cx.vecs.Space, /
) -> gt.BBtRealQuSz0:
    """Compute from `coordinax.vecs.AbstractPos3D`."""
    q = space["length"]
    q = eqx.error_if(
        q,
        not isinstance(q, cx.vecs.FourVector),
        msg="`space['length']` is not a FourVector.",
    )
    return api.spherical_mass_enclosed(pot, q)


@dispatch
def spherical_mass_enclosed(
    pot: AbstractPotential, space: cx.vecs.Space, t: gt.BBtRealQuSz0, /
) -> gt.BBtRealQuSz0:
    """Compute from `coordinax.vecs.AbstractPos3D`."""
    return api.spherical_mass_enclosed(pot, space["length"], t)


@dispatch
def spherical_mass_enclosed(
    pot: AbstractPotential, space: cx.frames.AbstractCoordinate, /
) -> gt.BBtRealQuSz0:
    """Compute from `coordinax.vecs.AbstractPos3D`."""
    return api.spherical_mass_enclosed(pot, space.data)


@dispatch
def spherical_mass_enclosed(
    pot: AbstractPotential, space: cx.frames.AbstractCoordinate, t: gt.BBtFloatQuSz0, /
) -> gt.BBtRealQuSz0:
    """Compute from `coordinax.vecs.AbstractPos3D`."""
    return api.spherical_mass_enclosed(pot, space.data, t)


@dispatch
def spherical_mass_enclosed(
    pot: AbstractPotential, coord: gc.PhaseSpacePosition, t: gt.BBtRealQuSz0, /
) -> gt.BBtRealQuSz0:
    """Compute from `coordinax.vecs.AbstractPos3D`."""
    return api.spherical_mass_enclosed(pot, coord.q, t)


@dispatch
def spherical_mass_enclosed(
    pot: AbstractPotential, wt: gc.AbstractPhaseSpaceCoordinate, /
) -> gt.BBtRealQuSz0:
    """Compute from `coordinax.vecs.AbstractPos3D`."""
    return api.spherical_mass_enclosed(pot, wt.q, wt.t)


@dispatch
def spherical_mass_enclosed(
    pot: AbstractPotential, wt: gc.AbstractPhaseSpaceCoordinate, t: gt.BBtFloatQuSz0, /
) -> gt.BBtRealQuSz0:
    """Compute from `coordinax.vecs.AbstractPos3D`."""
    t = eqx.error_if(
        wt.t,
        jnp.logical_not(jnp.array_equal(wt.t, t)),
        msg="`wt.t` and `t` are not equal.",
    )
    return api.spherical_mass_enclosed(pot, wt.q, t)
