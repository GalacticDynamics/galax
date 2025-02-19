"""Hamiltonian field."""

__all__ = ["HamiltonianField"]

from functools import partial
from typing import Any, TypeAlias, final

import diffrax as dfx
import equinox as eqx
import jax
from plum import convert, dispatch

import coordinax as cx
import coordinax.vecs as cxv
import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import BareQuantity as FastQ

import galax._custom_types as gt
import galax.coordinates as gc
import galax.dynamics._src.custom_types as gdt
import galax.potential as gp
from .field_base import AbstractDynamicsField
from galax.potential._src.utils import parse_to_xyz_t
from galax.utils._unxt import AllowValue


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

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import diffrax as dfx
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.HernquistPotential(m_tot=1e12, r_s=5, units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)
    >>> field
    HamiltonianField( potential=HernquistPotential( ... ) )

    The `.terms()` method returns a PyTree of `diffrax.AbstractTerm` objects
    that can be used to integrate the equations of motion with a `diffrax`
    solver (e.g. `diffrax.diffeqsolve`). The `.terms()` method is solver
    dependent, e.g. returning a `ODETerm` for `diffrax.Dopri8` and
    `tuple[ODETerm, ODETerm]` for `diffrax.SemiImplicitEuler`.

    >>> solver = dfx.Dopri8()
    >>> field.terms(solver)
    ODETerm(vector_field=<wrapped function __call__>)

    >>> solver = dfx.SemiImplicitEuler()
    >>> field.terms(solver)
    (ODETerm( ... ), ODETerm( ... ))

    Just to continue the example, we can use this field to integrate the
    equations of motion:

    >>> solver = gd.DynamicsSolver()  # defaults to Dopri8
    >>> w0 = gc.PhaseSpaceCoordinate(
    ...     q=u.Quantity([[8, 0, 9], [9, 0, 3]], "kpc"),
    ...     p=u.Quantity([0, 220, 0], "km/s"),
    ...     t=u.Quantity(0, "Gyr"))
    >>> t1 = u.Quantity(1, "Gyr")
    >>> soln = solver.solve(field, w0, t1)
    >>> soln
    Solution( t0=f64[], t1=f64[], ts=f64[1],
              ys=(f64[1,2,3], f64[1,2,3]),
              ... )

    >>> w = gc.PhaseSpaceCoordinate.from_(soln, frame=w0.frame, units=pot.units)
    >>> print(w, w.shape, sep='\n')
    PhaseSpaceCoordinate(
        q=<CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [[-5.151 -6.454 -5.795]
             [ 4.277  4.633  1.426]]>,
        p=<CartesianVel3D (x[kpc / Myr], y[kpc / Myr], z[kpc / Myr])
            [[ 0.225 -0.068  0.253]
             [-0.439 -0.002 -0.146]]>,
        t=Quantity['time'](Array(1000., dtype=float64), unit='Myr'),
        frame=SimulationFrame())
    (2,)

    The ``__call__`` is very flexible and can be called with many different
    combinations of arguments. See the ``__call__`` method for more information.

    """

    #: Potential.
    potential: gp.AbstractPotential

    @property
    def units(self) -> u.AbstractUnitSystem:
        return self.potential.units

    @dispatch.abstract
    def __call__(self, *_: Any, **__: Any) -> tuple[Any, Any]:
        """Evaluate the field at a position and time.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import unxt as u
        >>> import coordinax as cx
        >>> import galax.coordinates as gc
        >>> import galax.potential as gp
        >>> import galax.dynamics as gd

        >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")
        >>> field = gd.fields.HamiltonianField(pot)

        HamiltonianField can be called using many different combinations of
        arguments. Let's work up the type ladder:

        - `jax.Array` (assumed to be Cartesian coordinates and in the unit
          system of the field):

        >>> t = 0  # [Myr]
        >>> x = jnp.array([8., 0, 0])  # [kpc]
        >>> v = jnp.array([0, 0.22499668, 0])  # [kpc/Myr] (~220 km/s)

        >>> field(t, x, v)
        (Array([0.        , 0.22499668, 0.        ], dtype=float64),
         Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        >>> field(t, (x, v))
        (Array([0.        , 0.22499668, 0.        ], dtype=float64),
         Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        >>> xv = jnp.concat([x, v])
        >>> field(t, xv)
        (Array([0.        , 0.22499668, 0.        ], dtype=float64),
         Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        >>> txv = jnp.concat([jnp.array([t]), x, v])
        >>> field(txv)
        (Array([0.        , 0.22499668, 0.        ], dtype=float64),
         Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        - `unxt.Quantity` (assumed to be in Cartesian coordinates).

        >>> t = u.Quantity(0, "Gyr")
        >>> q = u.Quantity([8., 0, 0], "kpc")
        >>> p = u.Quantity([0, 220, 0], "km/s")

        >>> field(t, (q, p))
        (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
        Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        >>> field(t, q, p)
        (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
         Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        - `coordinax.vecs.AbstractVector`:

        >>> q = cx.CartesianPos3D.from_(q)
        >>> p = cx.CartesianVel3D.from_(p)

        >>> field(t, q, p)
        (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
         Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        >>> field(t, (q, p))
        (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
         Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        - `coordinax.vecs.FourVector`:

        >>> tq = cxv.FourVector(q=q, t=t)

        >>> field(tq, p)
        (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
         Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        >>> field(tq, p)
        (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
         Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        - `coordinax.vecs.Space`:

        >>> space = cx.Space(length=tq, speed=p)
        >>> field(space)
        (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
         Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        >>> space = cx.Space(length=q, speed=p)
        >>> field(t, space)
        (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
         Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        - `coordinax.frames.AbstractCoordinate`:

        >>> coord = cx.Coordinate(space, frame=gc.frames.SimulationFrame())
        >>> field(t, coord)
        (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
         Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        >>> coord = cx.Coordinate(cx.Space(length=tq, speed=p),
        ...                       frame=gc.frames.SimulationFrame())
        >>> field(coord)
        (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
         Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        >>> field(t, coord)
        (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
         Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        >>> coord = cx.Coordinate(cx.Space(length=tq, speed=p),
        ...                       frame=gc.frames.SimulationFrame())
        >>> field(coord)
        (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
         Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        - `galax.coordinates.PhaseSpacePosition`:

        >>> w = gc.PhaseSpacePosition(q=q, p=p)
        >>> field(t, w)
        (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
         Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        - `galax.coordinates.PhaseSpaceCoordinate`:

        >>> wt = gc.PhaseSpaceCoordinate(t=t, q=q, p=p)
        >>> field(wt)
        (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
         Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        >>> field(t, wt)
        (Array([0.        , 0.22499668, 0.        ], dtype=float64, ...),
         Array([-0.0702891, -0.       , -0.       ], dtype=float64))

        """
        raise NotImplementedError  # pragma: no cover

    # ---------------------------
    # Private API to support symplectic integrators. It would be good to figure
    # out a way to make these methods part of `__call__`.

    @eqx.filter_jit  # type: ignore[misc]
    def _dqdt(self, t: Any, p: gdt.BBtParr, args: Any, /) -> gdt.BBtParr:  # noqa: ARG002
        """Call with time, position quantity arrays."""
        return p

    @eqx.filter_jit  # type: ignore[misc]
    def _dpdt(self, t: gt.BBtSz0, q: gdt.BBtQarr, _: Any, /) -> gdt.BtAarr:
        """Call with time, velocity quantity arrays."""
        return -self.potential._gradient(q, t)  # noqa: SLF001


# ===============================================
# Terms dispatches


@AbstractDynamicsField.terms.dispatch  # type: ignore[misc]
def terms(
    self: HamiltonianField,
    _: dfx.SemiImplicitEuler,
    /,
) -> tuple[dfx.AbstractTerm, dfx.AbstractTerm]:
    r"""Return the AbstractTerm terms for the SemiImplicitEuler solver.

    Examples
    --------
    >>> import diffrax as dfx
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> solver = dfx.SemiImplicitEuler()

    >>> field.terms(solver)
    (ODETerm( ... ), ODETerm( ... ))

    For completeness we'll integrate the EoM.

    >>> dynamics_solver = gd.DynamicsSolver(solver,
    ...                                     stepsize_controller=dfx.ConstantStepSize())
    >>> w0 = gc.PhaseSpaceCoordinate(
    ...     q=u.Quantity([8., 0, 0], "kpc"),
    ...     p=u.Quantity([0, 220, 0], "km/s"),
    ...     t=u.Quantity(0, "Gyr"))
    >>> t1 = u.Quantity(200, "Myr")

    >>> soln = dynamics_solver.solve(field, w0, t1, dt0=0.001, max_steps=200_000)
    >>> w = gc.PhaseSpaceCoordinate.from_(soln, units=pot.units, frame=w0.frame)
    >>> print(w)
    PhaseSpaceCoordinate(
        q=<CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [ 7.645 -0.701  0.   ]>,
        p=<CartesianVel3D (x[kpc / Myr], y[kpc / Myr], z[kpc / Myr])
            [0.228 0.215 0.   ]>,
        t=Quantity['time'](Array(200., dtype=float64), unit='Myr'),
        frame=SimulationFrame())

    """
    return (dfx.ODETerm(self._dqdt), dfx.ODETerm(self._dpdt))


# ===============================================
# Call dispatches


Args: TypeAlias = tuple[Any, ...] | None


@HamiltonianField.__call__.dispatch
@partial(jax.jit)
def call(
    self: HamiltonianField,
    t: gt.RealQuSz0 | gt.BBtSz0 | gt.RealScalarLike,
    q: gdt.BBtQ | gdt.BBtQarr,
    p: gdt.BBtP | gdt.BBtParr,
    _: Args = None,
    /,
) -> gdt.BtPAarr:
    """Call with time, position, velocity quantity arrays."""
    xyz, t = parse_to_xyz_t(None, q, t, dtype=float, ustrip=self.units)
    p = u.ustrip(AllowValue, self.units["speed"], p)
    a = -self.potential._gradient(xyz, t)  # noqa: SLF001
    return p, a


@HamiltonianField.__call__.dispatch
@partial(jax.jit)
def call(
    self: HamiltonianField,
    t: gt.RealQuSz0 | gt.BBtSz0 | gt.RealScalarLike,
    qp: gdt.BBtQP | gdt.BBtQParr | tuple[cxv.AbstractPos3D, cxv.AbstractVel3D],
    args: Args = None,
    /,
) -> gdt.BtPAarr:
    """Call with time, (position, velocity) quantity arrays."""
    return self(t, qp[0], qp[1], args)


# ---------------------------


@HamiltonianField.__call__.dispatch
@partial(jax.jit)
def call(
    self: HamiltonianField, tq: gt.BBtSz4, p: gt.BBtSz3, args: Args = None, /
) -> gdt.BtPAarr:
    """Call with time-pos 4 array, vel 3 array."""
    t, q = tq[..., 0], tq[..., 1:4]
    return self(t, q, p, args)


@HamiltonianField.__call__.dispatch
@partial(jax.jit)
def call(
    self: HamiltonianField,
    t: gt.BBtSz0 | gt.RealScalarLike,
    tq: gt.BBtSz4,
    p: gt.BBtSz3,
    args: Args = None,
    /,
) -> gdt.BtPAarr:
    """Call with time-pos 4 array, vel 3 array."""
    t = eqx.error_if(
        t, jnp.logical_not(jnp.array_equal(t, tq[..., 0])), "t != tq[...,0]"
    )
    q = tq[..., 1:4]
    return self(t, q, p, args)


# ---------------------------


@HamiltonianField.__call__.dispatch
@partial(jax.jit)
def call(
    self: HamiltonianField,
    t: gt.RealQuSz0 | gt.BBtSz0 | gt.RealScalarLike,
    qp: gt.BBtSz6,
    args: Args = None,
    /,
) -> gdt.BtPAarr:
    """Call with time, pos-vel 6 array."""
    return self(t, (qp[..., 0:3], qp[..., 3:6]), args)


# ---------------------------


@HamiltonianField.__call__.dispatch
@partial(jax.jit)
def call(self: HamiltonianField, qp: gt.BBtSz7, args: Args = None, /) -> gdt.BtPAarr:
    """Call with time, pos-vel 7 array."""
    return self(qp[..., 0], (qp[..., 1:4], qp[..., 4:7]), args)


@HamiltonianField.__call__.dispatch
@partial(jax.jit)
def call(
    self: HamiltonianField,
    t: gt.BBtSz0 | gt.RealScalarLike,
    qp: gt.BBtSz7,
    args: Args = None,
    /,
) -> gdt.BtPAarr:
    """Call with time, pos-vel 7 array."""
    t = eqx.error_if(
        t, jnp.logical_not(jnp.array_equal(t, qp[..., 0])), "t != qp[...,0]"
    )
    return self(t, qp[..., 1:4], qp[..., 4:7], args)


# ---------------------------


@HamiltonianField.__call__.dispatch
@partial(jax.jit)
def call(
    self: HamiltonianField,
    t: gt.RealQuSz0 | gt.BBtSz0 | gt.RealScalarLike,
    q: cxv.AbstractPos3D,
    p: cxv.AbstractVel3D,
    args: Args = None,
    /,
) -> gdt.BtPAarr:
    """Call with time and `coordinax` ``AbstractPos3D`` and ``AbstractVel3D``."""
    return self(t, (convert(q, FastQ), convert(p, FastQ)), args)


# ---------------------------


@HamiltonianField.__call__.dispatch
@partial(jax.jit)
def call(
    self: HamiltonianField,
    tq: cxv.FourVector,
    p: cxv.AbstractVel3D,
    args: Args = None,
    /,
) -> gdt.BtPAarr:
    """Call with `coordinax.vecs.FourVector`, `coordinax.vecs.AbstractVel3D`."""
    return self(tq.t, tq.q, p, args)


@HamiltonianField.__call__.dispatch
@partial(jax.jit)
def call(
    self: HamiltonianField,
    t: Any,
    tq: cxv.FourVector,
    p: cxv.AbstractVel3D,
    args: Args = None,
    /,
) -> gdt.BtPAarr:
    """Call with `coordinax.vecs.FourVector`, `coordinax.vecs.AbstractVel3D`."""
    t = eqx.error_if(t, jnp.logical_not(jnp.array_equal(t, tq.t)), "t != tq.t")
    return self(t, tq.q, p, args)


# ---------------------------


@HamiltonianField.__call__.dispatch
@partial(jax.jit)
def call(
    self: HamiltonianField, t: Any, space: cxv.Space, args: Args = None, /
) -> gdt.BtPAarr:
    """Call with `coordinax.vecs.Space`."""
    return self(t, space["length"], space["speed"], args)


@HamiltonianField.__call__.dispatch
@partial(jax.jit)
def call(self: HamiltonianField, space: cxv.Space, args: Args = None, /) -> gdt.BtPAarr:
    """Call with `coordinax.vecs.Space`."""
    q = space["length"]
    q = eqx.error_if(
        q, not isinstance(q, cxv.FourVector), "space['length'] is not a FourVector"
    )
    return self(q, space["speed"], args)


# ---------------------------


@HamiltonianField.__call__.dispatch
@partial(jax.jit)
def call(
    self: HamiltonianField, w: cx.frames.AbstractCoordinate, args: Args = None, /
) -> gdt.BtPAarr:
    """Call with `coordinax.AbstractCoordinate`."""
    w = w.to_frame(gc.frames.simulation_frame)  # TODO: enable other frames
    return self(w.data, args)


@HamiltonianField.__call__.dispatch
@partial(jax.jit)
def call(
    self: HamiltonianField,
    t: Any,
    w: cx.frames.AbstractCoordinate,
    args: Args = None,
    /,
) -> gdt.BtPAarr:
    """Call with `coordinax.AbstractCoordinate`."""
    w = w.to_frame(gc.frames.simulation_frame)  # TODO: enable other frames
    return self(t, w.data, args)


# ---------------------------


@HamiltonianField.__call__.dispatch
@partial(jax.jit)
def call(
    self: HamiltonianField, t: Any, w: gc.PhaseSpacePosition, args: Args = None, /
) -> gdt.BtPAarr:
    """Call with `galax.coordinates.PhaseSpacePosition`."""
    w = w.to_frame(gc.frames.simulation_frame)  # TODO: enable other frames
    return self(t, w.q, w.p, args)


# ---------------------------


@HamiltonianField.__call__.dispatch
@partial(jax.jit)
def call(
    self: HamiltonianField, w: gc.PhaseSpaceCoordinate, args: Args = None, /
) -> gdt.BtPAarr:
    """Call with `galax.coordinates.PhaseSpaceCoordinate`."""
    return self(w.t, w.q, w.p, args)


@HamiltonianField.__call__.dispatch
@partial(jax.jit)
def call(
    self: HamiltonianField, t: Any, w: gc.PhaseSpaceCoordinate, args: Args = None, /
) -> gdt.BtPAarr:
    """Call with `galax.coordinates.PhaseSpaceCoordinate`."""
    t = eqx.error_if(t, jnp.logical_not(jnp.array_equal(t, w.t)), "t != w.t")
    return self(t, w.q, w.p, args)
