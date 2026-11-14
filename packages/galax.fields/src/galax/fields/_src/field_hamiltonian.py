"""Hamiltonian field."""

__all__ = ["HamiltonianField"]

import functools as ft

from typing import Any, final

import diffrax as dfx
import jax

import unxt as u

import galax.fields.custom_types as gt
import galax.potential as gp
from .field_base import AbstractOrbitField
from .utils import parse_to_t_y


@final
class HamiltonianField(AbstractOrbitField):
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
    >>> import diffraxtra as dfxtra
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.fields as gf

    >>> pot = gp.HernquistPotential(m_tot=1e12, r_s=5, units="galactic")
    >>> field = gf.HamiltonianField(pot)
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
    equations of motion with `galax.fields.integrate_field`:

    >>> solver = dfxtra.DiffEqSolver(
    ...     dfx.Dopri8(), stepsize_controller=dfx.PIDController(rtol=1e-7, atol=1e-7))
    >>> y0 = (u.Q([[8., 0, 9], [9, 0, 3]], "kpc"),
    ...       u.Q([[0., 220, 0], [0, 220, 0]], "km/s"))
    >>> ts = u.Q([0., 1], "Gyr")
    >>> soln = gf.integrate_field(field, y0, ts, solver=solver)
    >>> soln
    Solution( t0=f64[], t1=f64[], ts=f64[2],
              ys=(f64[2,2,3], f64[2,2,3]),
              ... )

    The solution has the initial and final states of the two orbits, in the
    units of the field (kpc and kpc/Myr).

    >>> soln.ys[0][-1].round(3)
    Array([[-5.151, -6.454, -5.795],
           [ 4.277,  4.633,  1.426]], dtype=float64)

    >>> soln.ys[1][-1].round(3)
    Array([[ 0.225, -0.068,  0.253],
           [-0.439, -0.002, -0.146]], dtype=float64)

    The ``__call__`` is very flexible and can be called with many different
    combinations of arguments. Let's work up the type ladder:

    >>> pot = gp.KeplerPotential(m_tot=1e11, units="galactic")
    >>> field = gf.HamiltonianField(pot)

    `galax.fields.HamiltonianField` can be called using many
    different combinations of arguments. Let's work up the type ladder:

    - `jax.Array` (assumed to be Cartesian coordinates and in the unit
        system of the field):

    >>> t = 0  # [Myr]
    >>> x = jnp.array([8., 0, 0])  # [kpc]
    >>> v = jnp.array([0, 0.22499668, 0])  # [kpc/Myr] (~220 km/s)

    >>> field(t, x, v, None)
    (Array([0.         , 0.22499668, 0.        ], dtype=float64),
     Array([-0.00702891, -0.        , -0.      ], dtype=float64))

    >>> field(t, (x, v))
    (Array([0.         , 0.22499668, 0.        ], dtype=float64),
     Array([-0.00702891, -0.        , -0.      ], dtype=float64))

    >>> xv = jnp.concat([x, v])
    >>> field(t, xv)
    (Array([0.         , 0.22499668, 0.        ], dtype=float64),
     Array([-0.00702891, -0.        , -0.      ], dtype=float64))

    >>> txv = jnp.concat([jnp.array([t]), x, v])
    >>> field(txv)
    (Array([0.         , 0.22499668, 0.        ], dtype=float64),
     Array([-0.00702891, -0.        , -0.      ], dtype=float64))

    - `unxt.Quantity` (assumed to be in Cartesian coordinates).

    >>> t = u.Q(0, "Gyr")
    >>> q = u.Q([8., 0, 0], "kpc")
    >>> p = u.Q([0, 220, 0], "km/s")

    >>> field(t, (q, p))
    (Array([0.         , 0.22499668, 0.        ], dtype=float64),
     Array([-0.00702891, -0.        , -0.      ], dtype=float64))

    >>> field(t, q, p)
    (Array([0.         , 0.22499668, 0.        ], dtype=float64),
     Array([-0.00702891, -0.        , -0.      ], dtype=float64))

    - `coordinax.vecs.AbstractVector`:

    >>> q = cx.CartesianPos3D.from_(q)
    >>> p = cx.CartesianVel3D.from_(p)

    >>> field(t, q, p)
    (Array([0.         , 0.22499668, 0.        ], dtype=float64),
     Array([-0.00702891, -0.        , -0.      ], dtype=float64))

    >>> field(t, (q, p))
    (Array([0.         , 0.22499668, 0.        ], dtype=float64),
     Array([-0.00702891, -0.        , -0.      ], dtype=float64))

    - `coordinax.vecs.FourVector`:

    >>> tq = cx.vecs.FourVector(q=q, t=t)

    >>> field(tq, p)
    (Array([0.         , 0.22499668, 0.        ], dtype=float64),
     Array([-0.00702891, -0.        , -0.      ], dtype=float64))

    >>> field(tq, p)
    (Array([0.         , 0.22499668, 0.        ], dtype=float64),
     Array([-0.00702891, -0.        , -0.      ], dtype=float64))

    - `coordinax.vecs.KinematicSpace`:

    >>> space = cx.KinematicSpace(length=tq, speed=p)
    >>> field(space)
    (Array([0.         , 0.22499668, 0.        ], dtype=float64),
     Array([-0.00702891, -0.        , -0.      ], dtype=float64))

    >>> space = cx.KinematicSpace(length=q, speed=p)
    >>> field(t, space)
    (Array([0.         , 0.22499668, 0.        ], dtype=float64),
     Array([-0.00702891, -0.        , -0.      ], dtype=float64))

    - `coordinax.frames.AbstractCoordinate`:

    >>> coord = cx.Coordinate(space, frame=gc.frames.SimulationFrame())
    >>> field(t, coord)
    (Array([0.         , 0.22499668, 0.        ], dtype=float64),
     Array([-0.00702891, -0.        , -0.      ], dtype=float64))

    >>> coord = cx.Coordinate(cx.KinematicSpace(length=tq, speed=p),
    ...                       frame=gc.frames.SimulationFrame())
    >>> field(coord)
    (Array([0.         , 0.22499668, 0.        ], dtype=float64),
     Array([-0.00702891, -0.        , -0.      ], dtype=float64))

    >>> field(t, coord)
    (Array([0.         , 0.22499668, 0.        ], dtype=float64),
     Array([-0.00702891, -0.        , -0.      ], dtype=float64))

    >>> coord = cx.Coordinate(cx.KinematicSpace(length=tq, speed=p),
    ...                       frame=gc.frames.SimulationFrame())
    >>> field(coord)
    (Array([0.         , 0.22499668, 0.        ], dtype=float64),
     Array([-0.00702891, -0.        , -0.      ], dtype=float64))

    - `galax.coordinates.PhaseSpacePosition`:

    >>> w = gc.PhaseSpacePosition(q=q, p=p)
    >>> field(t, w)
    (Array([0.         , 0.22499668, 0.        ], dtype=float64),
     Array([-0.00702891, -0.        , -0.      ], dtype=float64))

    - `galax.coordinates.PhaseSpaceCoordinate`:

    >>> wt = gc.PhaseSpaceCoordinate(t=t, q=q, p=p)
    >>> field(wt)
    (Array([0.         , 0.22499668, 0.        ], dtype=float64),
     Array([-0.00702891, -0.        , -0.      ], dtype=float64))

    >>> field(t, wt)
    (Array([0.         , 0.22499668, 0.        ], dtype=float64),
     Array([-0.00702891, -0.        , -0.      ], dtype=float64))

    """

    #: Potential.
    potential: gp.AbstractPotential

    @property  # type: ignore[misc]
    def units(self) -> u.AbstractUnitSystem:
        return self.potential.units

    # ---------------------------
    # Symplectic integration terms
    # TODO: enable full gamut of inputs

    @jax.jit
    def dx_dt(self, t: Any, v_xyz: gt.BBtParr, args: Any, /) -> gt.BBtParr:  # noqa: ARG002
        """Call with time, position quantity arrays."""
        return v_xyz

    @jax.jit
    def dv_dt(self, t: gt.BBtSz0, xyz: gt.BBtQarr, _: Any, /) -> gt.BtAarr:
        """Call with time, velocity quantity arrays."""
        grad = self.potential._gradient(xyz, t)  # noqa: SLF001
        return -grad  # type: ignore[no-any-return]


# ===============================================
# Terms dispatches


@AbstractOrbitField.terms.dispatch
def terms(
    self: HamiltonianField,
    _: dfx.SemiImplicitEuler,
    /,
) -> tuple[dfx.ODETerm, dfx.ODETerm]:
    r"""Return the AbstractTerm terms for the SemiImplicitEuler solver.

    See also `galax.dynamics.experimental.Leapfrog`, another symplectic solver,
    for which the analogous dispatch is registered by `galax.dynamics` (in
    ``galax/dynamics/_src/experimental/leapfrog.py``), because `galax.fields`
    does not depend on `galax.dynamics`.

    Examples
    --------
    >>> import diffrax as dfx
    >>> import diffraxtra as dfxtra
    >>> import quaxed.numpy as jnp
    >>> import galax.potential as gp
    >>> import galax.fields as gf

    >>> pot = gp.KeplerPotential(m_tot=1e11, units="galactic")
    >>> field = gf.HamiltonianField(pot)

    >>> solver = dfx.SemiImplicitEuler()

    >>> field.terms(solver)
    (ODETerm(...), ODETerm(...))

    For completeness we'll integrate the EoM. The initial conditions are in the
    units of the field (kpc and kpc/Myr).

    >>> dynamics_solver = dfxtra.DiffEqSolver(
    ...     solver, stepsize_controller=dfx.ConstantStepSize(), max_steps=200_000)
    >>> y0 = (jnp.asarray([8., 0, 0]), jnp.asarray([0, 0.22499668, 0]))

    >>> soln = dynamics_solver(field, 0., 200., 0.001, y0)  # t0, t1, dt0 [Myr]
    >>> soln.ys[0].round(3), soln.ys[1].round(3)
    (Array([[7.091, 3.504, 0.   ]], dtype=float64),
     Array([[-0.111,  0.199,  0.   ]], dtype=float64))

    """
    return (dfx.ODETerm(self.dx_dt), dfx.ODETerm(self.dv_dt))


# ===============================================
# Call dispatches


@AbstractOrbitField.__call__.dispatch
@ft.partial(jax.jit)
def call(self: HamiltonianField, tqp: Any, args: gt.OptArgs = None, /) -> gt.BtPAarr:
    """Call with time, position, velocity quantity arrays."""
    t, (xyz, v_xyz) = parse_to_t_y(None, tqp, ustrip=self.units)
    return self.dx_dt(t, v_xyz, args), self.dv_dt(t, xyz, args)


@AbstractOrbitField.__call__.dispatch
@ft.partial(jax.jit)
def call(
    self: HamiltonianField, tq: Any, qp: Any, args: gt.OptArgs = None, /
) -> gt.BtPAarr:
    """Call with time, position, velocity quantity arrays."""
    t, (xyz, v_xyz) = parse_to_t_y(None, tq, qp, ustrip=self.units)
    return self.dx_dt(t, v_xyz, args), self.dv_dt(t, xyz, args)


@AbstractOrbitField.__call__.dispatch
@ft.partial(jax.jit)
def call(
    self: HamiltonianField, t: Any, q: Any, p: Any, args: gt.OptArgs = None, /
) -> gt.BtPAarr:
    """Call with time, position, velocity quantity arrays."""
    t, (xyz, v_xyz) = parse_to_t_y(None, t, q, p, ustrip=self.units)
    return self.dx_dt(t, v_xyz, args), self.dv_dt(t, xyz, args)
