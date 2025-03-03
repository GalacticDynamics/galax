"""Hamiltonian field."""

__all__ = ["HamiltonianField"]

from functools import partial
from typing import Any, final

import diffrax as dfx
import jax

import unxt as u

import galax._custom_types as gt
import galax.dynamics._src.custom_types as gdt
import galax.potential as gp
from .field_base import AbstractOrbitField
from galax.dynamics._src.utils import parse_to_t_y


@final
class HamiltonianField(AbstractOrbitField, strict=True):  # type: ignore[call-arg]
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
    >>> import coordinax as cx
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
    ODETerm(...)

    >>> solver = dfx.SemiImplicitEuler()
    >>> field.terms(solver)
    (ODETerm(...), ODETerm(...))

    Just to continue the example, we can use this field to integrate the
    equations of motion:

    >>> solver = gd.OrbitSolver()  # defaults to Dopri8
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
    combinations of arguments. Let's work up the type ladder:

    >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    `galax.dynamics.fields.HamiltonianField` can be called using many
    different combinations of arguments. Let's work up the type ladder:

    - `jax.Array` (assumed to be Cartesian coordinates and in the unit
        system of the field):

    >>> t = 0  # [Myr]
    >>> x = jnp.array([8., 0, 0])  # [kpc]
    >>> v = jnp.array([0, 0.22499668, 0])  # [kpc/Myr] (~220 km/s)

    >>> field(t, x, v, None)
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
    (Array([0.        , 0.22499668, 0.        ], dtype=float64),
        Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    >>> field(t, q, p)
    (Array([0.        , 0.22499668, 0.        ], dtype=float64),
        Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    - `coordinax.vecs.AbstractVector`:

    >>> q = cx.CartesianPos3D.from_(q)
    >>> p = cx.CartesianVel3D.from_(p)

    >>> field(t, q, p)
    (Array([0.        , 0.22499668, 0.        ], dtype=float64),
        Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    >>> field(t, (q, p))
    (Array([0.        , 0.22499668, 0.        ], dtype=float64),
        Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    - `coordinax.vecs.FourVector`:

    >>> tq = cx.vecs.FourVector(q=q, t=t)

    >>> field(tq, p)
    (Array([0.        , 0.22499668, 0.        ], dtype=float64),
        Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    >>> field(tq, p)
    (Array([0.        , 0.22499668, 0.        ], dtype=float64),
        Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    - `coordinax.vecs.Space`:

    >>> space = cx.Space(length=tq, speed=p)
    >>> field(space)
    (Array([0.        , 0.22499668, 0.        ], dtype=float64),
        Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    >>> space = cx.Space(length=q, speed=p)
    >>> field(t, space)
    (Array([0.        , 0.22499668, 0.        ], dtype=float64),
        Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    - `coordinax.frames.AbstractCoordinate`:

    >>> coord = cx.Coordinate(space, frame=gc.frames.SimulationFrame())
    >>> field(t, coord)
    (Array([0.        , 0.22499668, 0.        ], dtype=float64),
        Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    >>> coord = cx.Coordinate(cx.Space(length=tq, speed=p),
    ...                       frame=gc.frames.SimulationFrame())
    >>> field(coord)
    (Array([0.        , 0.22499668, 0.        ], dtype=float64),
        Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    >>> field(t, coord)
    (Array([0.        , 0.22499668, 0.        ], dtype=float64),
        Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    >>> coord = cx.Coordinate(cx.Space(length=tq, speed=p),
    ...                       frame=gc.frames.SimulationFrame())
    >>> field(coord)
    (Array([0.        , 0.22499668, 0.        ], dtype=float64),
        Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    - `galax.coordinates.PhaseSpacePosition`:

    >>> w = gc.PhaseSpacePosition(q=q, p=p)
    >>> field(t, w)
    (Array([0.        , 0.22499668, 0.        ], dtype=float64),
        Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    - `galax.coordinates.PhaseSpaceCoordinate`:

    >>> wt = gc.PhaseSpaceCoordinate(t=t, q=q, p=p)
    >>> field(wt)
    (Array([0.        , 0.22499668, 0.        ], dtype=float64),
        Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    >>> field(t, wt)
    (Array([0.        , 0.22499668, 0.        ], dtype=float64),
        Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    """

    #: Potential.
    potential: gp.AbstractPotential

    @property
    def units(self) -> u.AbstractUnitSystem:
        return self.potential.units

    # ---------------------------
    # Symplectic integration terms
    # TODO: enable full gamut of inputs

    @jax.jit  # type: ignore[misc]
    def dx_dt(self, t: Any, v_xyz: gdt.BBtParr, args: Any, /) -> gdt.BBtParr:  # noqa: ARG002
        """Call with time, position quantity arrays."""
        return v_xyz

    @jax.jit  # type: ignore[misc]
    def dv_dt(self, t: gt.BBtSz0, xyz: gdt.BBtQarr, _: Any, /) -> gdt.BtAarr:
        """Call with time, velocity quantity arrays."""
        return -self.potential._gradient(xyz, t)  # noqa: SLF001


# ===============================================
# Terms dispatches


@AbstractOrbitField.terms.dispatch  # type: ignore[misc]
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
    (ODETerm(...), ODETerm(...))

    For completeness we'll integrate the EoM.

    >>> dynamics_solver = gd.OrbitSolver(solver,
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
    return (dfx.ODETerm(self.dx_dt), dfx.ODETerm(self.dv_dt))


# ===============================================
# Call dispatches


@AbstractOrbitField.__call__.dispatch
@partial(jax.jit)
def call(self: HamiltonianField, tqp: Any, args: gt.OptArgs = None, /) -> gdt.BtPAarr:
    """Call with time, position, velocity quantity arrays."""
    t, (xyz, v_xyz) = parse_to_t_y(None, tqp, ustrip=self.units)
    return self.dx_dt(t, v_xyz, args), self.dv_dt(t, xyz, args)


@AbstractOrbitField.__call__.dispatch
@partial(jax.jit)
def call(
    self: HamiltonianField, tq: Any, qp: Any, args: gt.OptArgs = None, /
) -> gdt.BtPAarr:
    """Call with time, position, velocity quantity arrays."""
    t, (xyz, v_xyz) = parse_to_t_y(None, tq, qp, ustrip=self.units)
    return self.dx_dt(t, v_xyz, args), self.dv_dt(t, xyz, args)


@AbstractOrbitField.__call__.dispatch
@partial(jax.jit)
def call(
    self: HamiltonianField, t: Any, q: Any, p: Any, args: gt.OptArgs = None, /
) -> gdt.BtPAarr:
    """Call with time, position, velocity quantity arrays."""
    t, (xyz, v_xyz) = parse_to_t_y(None, t, q, p, ustrip=self.units)
    return self.dx_dt(t, v_xyz, args), self.dv_dt(t, xyz, args)
