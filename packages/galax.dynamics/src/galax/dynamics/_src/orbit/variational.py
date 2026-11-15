"""Variational equations for orbit fields.

This is private API.

The augmented state carries ``k`` deviation vectors next to the orbit, in the
layout ``((x, dX), (v, dV))`` with ``x, v`` of shape ``(3,)`` and ``dX, dV`` of
shape ``(3, k)``. The deviations obey the linearized equations of motion, whose
right-hand side is the Jacobian of the field applied to each column.

"""

__all__ = ["compute_variational", "tangent_terms"]

from jaxtyping import Array, Float
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.tree as jtu
from plum import dispatch

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax.coordinates as gc
import galax.potential as gp
from .field_base import AbstractOrbitField
from .field_hamiltonian import HamiltonianField
from .field_nbody import NBodyField
from .orbit import Orbit
from .solver import OrbitSolver
from galax.dynamics._src.utils import parse_to_t_y


def _linearize_columns(
    fn: Any, primals: tuple[Any, ...], tangents: tuple[Any, ...]
) -> tuple[Any, Any]:
    """Evaluate ``fn`` and apply its Jacobian to each column of ``tangents``.

    The function is linearized once and the linear map is mapped over the last
    axis of the tangents, so ``k`` deviation vectors cost one evaluation of the
    field plus ``k`` Jacobian-vector products.
    """
    out, lin = jax.linearize(fn, *primals)
    d_out = jax.vmap(lin, in_axes=-1, out_axes=-1)(*tangents)
    return out, d_out


@dispatch
def tangent_terms(term: dfx.ODETerm, /) -> dfx.ODETerm:
    """Augment an ODE term on ``(x, v)`` with the variational equations.

    The returned term acts on ``((x, dX), (v, dV))`` and returns
    ``((dx/dt, d(dX)/dt), (dv/dt, d(dV)/dt))``.

    Examples
    --------
    >>> import diffrax as dfx
    >>> import jax.numpy as jnp
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd
    >>> from galax.dynamics._src.orbit.variational import tangent_terms

    >>> pot = gp.HarmonicOscillatorPotential(omega=0.1, units="galactic")
    >>> term = gd.fields.HamiltonianField(pot).terms(dfx.Dopri8())
    >>> aug = tangent_terms(term)
    >>> x, v, eye = jnp.array([1.0, 0, 0]), jnp.zeros(3), jnp.eye(6)
    >>> (xd, dXd), (vd, dVd) = aug.vf(0.0, ((x, eye[:3]), (v, eye[3:])), None)
    >>> dVd[0, 0]  # -omega^2
    Array(-0.01, dtype=float64)

    """

    def vf(t: Any, y: Any, args: Any) -> Any:
        (x, dx), (v, dv) = y
        (xd, vd), (dxd, dvd) = _linearize_columns(
            lambda x, v: term.vf(t, (x, v), args), (x, v), (dx, dv)
        )
        return (xd, dxd), (vd, dvd)

    return dfx.ODETerm(vf)


@dispatch
def tangent_terms(
    terms: tuple[dfx.ODETerm, dfx.ODETerm], /
) -> tuple[dfx.ODETerm, dfx.ODETerm]:
    """Augment split ``(drift, kick)`` terms with the variational equations.

    The drift acts on ``(v, dV)`` and the kick on ``(x, dX)``. Because a
    symplectic solver applies the same sub-steps to the deviations as to the
    orbit, the deviations follow the exact tangent map of the integrator at the
    end of each step. Between steps the solver's interpolation applies.

    Examples
    --------
    >>> import diffrax as dfx
    >>> import jax.numpy as jnp
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd
    >>> from galax.dynamics._src.orbit.variational import tangent_terms

    >>> pot = gp.HarmonicOscillatorPotential(omega=0.1, units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)
    >>> drift, kick = tangent_terms(field.terms(dfx.SemiImplicitEuler()))
    >>> x, eye = jnp.array([1.0, 0, 0]), jnp.eye(3)
    >>> a, dA = kick.vf(0.0, (x, eye), None)
    >>> dA[0, 0]  # -omega^2
    Array(-0.01, dtype=float64)

    """
    return tuple(_tangent_split(term) for term in terms)


def _tangent_split(term: dfx.ODETerm, /) -> dfx.ODETerm:
    def vf(t: Any, y: Any, args: Any) -> Any:
        z, dz = y
        return _linearize_columns(lambda z: term.vf(t, z, args), (z,), (dz,))

    return dfx.ODETerm(vf)


# ===================================================================


def _solve_augmented(
    terms: Any,
    solver: OrbitSolver,
    t0: Any,
    t1: Any,
    y0: Any,
    saveat: dfx.SaveAt,
    dt0: float | None,
) -> Any:
    """Integrate an augmented state with the solver's settings; return ``ys``.

    ``dt0`` is taken as a magnitude and given the sign of ``t1 - t0``, so one
    ``dt0`` serves forward, backward, and mixed-direction stages.
    """
    if dt0 is not None:
        dt0 = jnp.where(t1 >= t0, 1, -1) * jnp.abs(dt0)
    soln = solver(terms, t0=t0, t1=t1, dt0=dt0, y0=y0, saveat=saveat)
    return soln.ys


def _prepare(
    field: AbstractOrbitField | gp.AbstractPotential,
    solver: OrbitSolver | None,
    dt0: Any,
) -> tuple[AbstractOrbitField, OrbitSolver, u.AbstractUnitSystem, Any]:
    """Check and normalize the field, solver, and ``dt0`` arguments.

    A potential is wrapped in a `HamiltonianField`, N-body fields are
    rejected, and a `unxt.Quantity` ``dt0`` is converted to the field's time
    unit.
    """
    if isinstance(field, NBodyField):
        msg = "N-body fields are not supported; pass a single-particle field."
        raise TypeError(msg)
    thefield = (
        field if isinstance(field, AbstractOrbitField) else HamiltonianField(field)  # type: ignore[call-arg]
    )
    solver = OrbitSolver() if solver is None else solver
    units = thefield.units
    if dt0 is not None:
        dt0 = u.ustrip(AllowValue, units["time"], dt0)
    return thefield, solver, units, dt0


@eqx.filter_jit
def compute_variational(
    field: AbstractOrbitField | gp.AbstractPotential,
    w0: gc.AbstractPhaseSpaceCoordinate,
    ts: Any,
    /,
    *,
    dw0: Float[Array, "6 k"] | None = None,
    solver: OrbitSolver | None = None,
    dt0: Any = None,
) -> tuple[Orbit, Float[Array, "*batch time 6 k"]]:
    r"""Integrate an orbit together with its deviation vectors.

    The deviations :math:`\delta w` obey the variational equations
    :math:`d\delta w / dt = J(t)\, \delta w`, with :math:`J` the Jacobian of
    the field along the orbit. With ``dw0=None`` the deviations start as the
    identity, so the result is the state-transition matrix
    :math:`M(t) = \partial w(t) / \partial w_0`, where :math:`w_0` is the
    phase-space position ``w0`` at its time ``w0.t``.

    Parameters
    ----------
    field
        A single-particle orbit field. A potential is wrapped in a
        `galax.dynamics.fields.HamiltonianField`. N-body fields are not
        supported.
    w0
        The initial phase-space coordinate, with any batch shape.
    ts
        The output times. A `unxt.Quantity` is converted to the field's units.
    dw0
        The ``k`` initial deviation vectors, as the columns of a ``(6, k)``
        array in Cartesian ``(x, y, z, v_x, v_y, v_z)`` order and the field's
        units. Defaults to the ``(6, 6)`` identity.
    solver
        The solver. Defaults to `galax.dynamics.OrbitSolver`. Its error control
        includes the deviations, so the returned orbit can differ slightly from
        `galax.dynamics.orbit.compute_orbit` at the same tolerances. A solver
        ``event`` sees the augmented state ``((x, dX), (v, dV))``.
    dt0
        Initial step size, needed for constant-step solvers. A plain number is
        in the field's time unit, and a `unxt.Quantity` is converted. Only its
        size is used; the direction of integration sets its sign.

    Returns
    -------
    orbit
        The orbit at ``ts``.
    D
        The deviations at ``ts``, shape ``(*batch, time, 6, k)``, as a plain
        array in Cartesian order and the field's units.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.KeplerPotential(m_tot=1e11, units="galactic")
    >>> w0 = gc.PhaseSpaceCoordinate(q=u.Q([8.0, 0, 0], "kpc"),
    ...                              p=u.Q([0, 220, 0], "km/s"),
    ...                              t=u.Q(0, "Myr"))
    >>> ts = u.Q(jnp.linspace(0, 100, 5), "Myr")
    >>> orbit, M = gd.orbit.compute_variational(pot, w0, ts)
    >>> M.shape
    (5, 6, 6)
    >>> bool(jnp.allclose(M[0], jnp.eye(6)))
    True
    >>> bool(jnp.allclose(jnp.linalg.det(M), 1.0))
    True

    """
    thefield, solver, units, dt0 = _prepare(field, solver, dt0)
    ts = jnp.atleast_1d(u.ustrip(AllowValue, units["time"], ts))
    dws = jnp.eye(6) if dw0 is None else jnp.asarray(dw0)
    if dws.ndim != 2 or dws.shape[0] != 6:
        msg = f"dw0 must have shape (6, k), got {dws.shape}."
        raise ValueError(msg)
    terms = tangent_terms(thefield.terms(solver.solver))

    t0, (x0, v0) = parse_to_t_y(None, w0, ustrip=units)
    batch = x0.shape[:-1]
    t0 = jnp.broadcast_to(t0, batch).reshape(-1)
    x0, v0 = x0.reshape(-1, 3), v0.reshape(-1, 3)

    def single(t0: Any, x0: Any, v0: Any) -> Any:
        y0 = ((x0, dws[:3]), (v0, dws[3:]))
        # Integrate from `w0.t` to `ts[0]`, then save along `ts`.
        ys = _solve_augmented(terms, solver, t0, ts[0], y0, dfx.SaveAt(ts=ts[:1]), dt0)
        if ts.shape != (1,):
            y1 = jtu.map(lambda y: y[0], ys)
            ys = _solve_augmented(
                terms, solver, ts[0], ts[-1], y1, dfx.SaveAt(ts=ts), dt0
            )
        return ys

    (x, dx), (v, dv) = jax.vmap(single)(t0, x0, v0)
    x, v = x.reshape(*batch, *x.shape[1:]), v.reshape(*batch, *v.shape[1:])
    D = jnp.concat([dx, dv], axis=-2).reshape(*batch, len(ts), 6, dws.shape[-1])

    orbit = Orbit(
        q=u.Q(x, units["length"]),
        p=u.Q(v, units["speed"]),
        t=u.Q(ts, units["time"]),
        frame=w0.frame,
    )
    return orbit, D
