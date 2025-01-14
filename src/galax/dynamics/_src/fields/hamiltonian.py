"""Dynamics Solvers.

This is private API.

"""

__all__ = ["HamiltonianField"]

from functools import partial
from typing import Any, final

import jax
from plum import convert, dispatch

import coordinax as cx
import unxt as u
from unxt.quantity import UncheckedQuantity as FastQ

import galax.coordinates as gc
import galax.potential as gp
import galax.typing as gt
from .base import AbstractDynamicsField


@final
class HamiltonianField(AbstractDynamicsField, strict=True):  # type: ignore[call-arg]
    r"""Dynamics field for Hamiltonian EoM.

    This is for Hamilton's equations for motion for a particle in a potential.

    .. math::

            \\dot{q} = \frac{dH}{dp} \\ \\dot{p} = -\frac{dH}{dq}

    .. note::

        Calling this object in a jit context will provide a significant speedup.

    .. warning::

        The call method currently returns a `tuple[Array[float, (3,)],
        Array[float, (3,)]]`. In future, when `unxt.Quantity` is registered with
        `quax.quaxify` for `diffrax.diffeqsolve` then this will return
        `tuple[Quantity[float, (3,), 'length'], Quantity[float, (3,),
        'speed']]`. Later, when `coordinax.AbstractVector` is registered with
        `quax.quaxify` for `diffrax.diffeqsolve` then this will return
        `tuple[CartesianPos3D, CartesianVel3D]`.

    """

    #: Potential.
    potential: gp.AbstractBasePotential

    @property
    def units(self) -> u.AbstractUnitSystem:
        return self.potential.units

    @dispatch.abstract
    def __call__(
        self, t: Any, qp: tuple[Any, Any], args: tuple[Any, ...], /
    ) -> tuple[Any, Any]:
        raise NotImplementedError  # pragma: no cover


# ---------------------------
# Call dispatches


@HamiltonianField.__call__.dispatch
@partial(jax.jit, inline=True)
def call(
    self: HamiltonianField,
    t: gt.RealQSz0,
    q: gt.BBtQ,
    p: gt.BBtP,
    args: tuple[Any, ...] | None,  # noqa: ARG001
    /,
) -> gt.BtPAarr:
    """Call with time, position, velocity quantity arrays.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> t = u.Quantity(0, "Myr")
    >>> q = u.Quantity([8., 0, 0], "kpc")
    >>> p = u.Quantity([0, 220, 0], "km/s")
    >>> field(t, q, p, None)
    (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
     Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    """
    # TODO: not require unit munging
    units = self.units
    a = -self.potential._gradient(q, t).ustrip(units["acceleration"])  # noqa: SLF001
    return p.ustrip(units["speed"]), a


@HamiltonianField.__call__.dispatch
@partial(jax.jit, inline=True)
def call(
    self: HamiltonianField,
    t: gt.RealQSz0,
    qp: gt.BBtQP,
    args: tuple[Any, ...] | None,
    /,
) -> gt.BtPAarr:
    """Call with time, (position, velocity) quantity arrays.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> t = u.Quantity(0, "Myr")
    >>> q = u.Quantity([8., 0, 0], "kpc")
    >>> p = u.Quantity([0, 220, 0], "km/s")
    >>> field(t, (q, p), None)
    (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
     Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    """
    return self(t, qp[0], qp[1], args)


@HamiltonianField.__call__.dispatch
@partial(jax.jit, inline=True)
def call(
    self: HamiltonianField,
    t: gt.BBtSz0,
    q: gt.BBtQarr,
    p: gt.BBtParr,
    args: tuple[Any, ...] | None,
    /,
) -> gt.BtPAarr:
    """Call with time, position, velocity arrays.

    The arrays are considered to be in the unit system of the field
    (`field.units`) Which can be checked with ``unitsystem["X"]`` for X in time,
    position, and speed.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> t = u.Quantity(0, "Myr").ustrip(pot.units)
    >>> q = u.Quantity([8., 0, 0], "kpc").ustrip(pot.units)
    >>> p = u.Quantity([0, 220, 0], "km/s").ustrip(pot.units)
    >>> field(t, q, p, None)
    (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
     Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    """
    units = self.units
    return self(
        FastQ(t, units["time"]),
        FastQ(q, units["length"]),
        FastQ(p, units["speed"]),
        args,
    )


# Note: this is the one called by DynamicsSolver, so it's special-cased
@HamiltonianField.__call__.dispatch
@partial(jax.jit, inline=True)
def call(
    self: HamiltonianField,
    t: gt.BBtSz0,
    qp: gt.BBtQParr,
    args: tuple[Any, ...] | None,  # noqa: ARG001
    /,
) -> gt.BtPAarr:
    """Call with time, (position, velocity) arrays.

    The arrays are considered to be in the unit system of the field
    (`field.units`) Which can be checked with ``unitsystem["X"]`` for X in time,
    position, and speed.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> t = u.Quantity(0, "Myr").ustrip(pot.units)
    >>> q = u.Quantity([8., 0, 0], "kpc").ustrip(pot.units)
    >>> p = u.Quantity([0, 220, 0], "km/s").ustrip(pot.units)
    >>> field(t, (q, p), None)
    (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
     Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    """
    # TODO: not require unit munging
    units = self.units
    a = -self.potential._gradient(  # noqa: SLF001
        FastQ(qp[0], units["length"]),
        FastQ(t, units["time"]),
    ).ustrip(units["acceleration"])
    return qp[1], a


@HamiltonianField.__call__.dispatch
@partial(jax.jit, inline=True)
def call(
    self: HamiltonianField,
    t: gt.BBtSz0,
    qp: gt.BBtSz6,
    args: tuple[Any, ...] | None,
    /,
) -> gt.BtPAarr:
    """Call with time, pos-vel 6 array.

    The arrays are considered to be in the unit system of the field
    (`field.units`) Which can be checked with ``unitsystem["X"]`` for X in time,
    position, and speed.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> t = u.Quantity(0, "Myr").ustrip(pot.units)
    >>> qp = jnp.concat([u.Quantity([8., 0, 0], "kpc").ustrip(pot.units),
    ...                  u.Quantity([0, 220, 0], "km/s").ustrip(pot.units)])

    >>> field(t, qp, None)
    (Array([0.        , 0.22499668, 0.        ], dtype=float64),
     Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    """
    return self(t, (qp[..., 0:3], qp[..., 3:6]), args)


@HamiltonianField.__call__.dispatch
@partial(jax.jit, inline=True)
def call(
    self: HamiltonianField,
    qp: gt.BBtSz7,
    args: tuple[Any, ...] | None,
    /,
) -> gt.BtPAarr:
    """Call with time, pos-vel 6 array.

    The arrays are considered to be in the unit system of the field
    (`field.units`) Which can be checked with ``unitsystem["X"]`` for X in time,
    position, and speed.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> tqp = jnp.concat([u.Quantity([0], "Myr").ustrip(pot.units),
    ...                   u.Quantity([8., 0, 0], "kpc").ustrip(pot.units),
    ...                   u.Quantity([0, 220, 0], "km/s").ustrip(pot.units)])

    >>> field(tqp, None)
    (Array([0.        , 0.22499668, 0.        ], dtype=float64),
     Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    """
    return self(qp[..., 0], (qp[..., 1:4], qp[..., 4:7]), args)


@HamiltonianField.__call__.dispatch
@partial(jax.jit, inline=True)
def call(
    self: HamiltonianField,
    t: gt.TimeBBtSz0,
    q: cx.vecs.AbstractPos3D,
    p: cx.vecs.AbstractVel3D,
    args: tuple[Any, ...] | None,
    /,
) -> gt.BtPAarr:
    """Call with time, `coordinax.vecs.AbstractPos3D`, `coordinax.vecs.AbstractVel3D`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> t = u.Quantity(0, "Myr")
    >>> q = cx.vecs.CartesianPos3D.from_([8., 0, 0], "kpc")
    >>> p = cx.vecs.CartesianVel3D.from_([0, 220, 0], "km/s")

    >>> field(t, q, p, None)
    (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
     Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    """
    return self(t, (convert(q, FastQ), convert(p, FastQ)), args)


@HamiltonianField.__call__.dispatch
@partial(jax.jit, inline=True)
def call(
    self: HamiltonianField,
    tq: cx.vecs.FourVector,
    p: cx.vecs.AbstractVel3D,
    args: tuple[Any, ...] | None,
    /,
) -> gt.BtPAarr:
    """Call with `coordinax.vecs.FourVector`, `coordinax.vecs.AbstractVel3D`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> q = cx.vecs.FourVector.from_([0, 8., 0, 0], "kpc")
    >>> p = cx.vecs.CartesianVel3D.from_([0, 220, 0], "km/s")

    >>> field(q, p, None)
    (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
     Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    """
    return self(tq.t, tq.q, p, args)


@HamiltonianField.__call__.dispatch
@partial(jax.jit, inline=True)
def call(
    self: HamiltonianField,
    space: cx.vecs.Space,
    args: tuple[Any, ...] | None,
    /,
) -> gt.BtPAarr:
    """Call with `coordinax.vecs.FourVector`, `coordinax.vecs.AbstractVel3D`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> space = cx.Space(length=cx.vecs.FourVector.from_([0, 8., 0, 0], "kpc"),
    ...                  speed=cx.vecs.CartesianVel3D.from_([0, 220, 0], "km/s"))

    >>> field(space, None)
    (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
     Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    """
    # TODO: better packing re 4Vec for "length"
    assert isinstance(space["length"], cx.vecs.FourVector)  # noqa: S101
    return self(space["length"], space["speed"], args)


@HamiltonianField.__call__.dispatch
@partial(jax.jit, inline=True)
def call(
    self: HamiltonianField,
    w: cx.frames.AbstractCoordinate,
    args: tuple[Any, ...] | None,
    /,
) -> gt.BtPAarr:
    """Call with `coordinax.AbstractCoordinate`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> w = cx.frames.Coordinate(
    ...     {"length": cx.vecs.FourVector.from_([0, 8., 0, 0], "kpc"),
    ...      "speed": cx.vecs.CartesianVel3D.from_([0, 220, 0], "km/s")},
    ...     gc.frames.SimulationFrame())

    >>> field(w, None)
    (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
     Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    """
    w = w.to_frame(gc.frames.SimulationFrame())  # TODO: enable other frames
    return self(w.data, args)


@HamiltonianField.__call__.dispatch
@partial(jax.jit, inline=True)
def call(
    self: HamiltonianField,
    w: gc.PhaseSpacePosition,
    args: tuple[Any, ...] | None,
    /,
) -> gt.BtPAarr:
    """Call with `galax.coordinates.PhaseSpacePosition`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> w = gc.PhaseSpacePosition(t=u.Quantity(0, "Gyr"),
    ...                           q=u.Quantity([8., 0, 0], "kpc"),
    ...                           p=u.Quantity([0, 220, 0], "km/s"))

    >>> field(w, None)
    (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
     Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    """
    assert w.t is not None  # noqa: S101
    w = w.to_frame(gc.frames.SimulationFrame())  # TODO: enable other frames
    return self(w.t, w._qp(units=self.units), args)  # noqa: SLF001
