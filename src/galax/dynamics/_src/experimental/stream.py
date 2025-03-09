"""Experimental dynamics."""

__all__: list[str] = []

from functools import partial
from typing import Any, TypeAlias, final
from typing_extensions import Unpack

import diffrax as dfx
import jax
import jax.random as jr
from jax.tree_util import register_dataclass
from jaxtyping import Array, PRNGKeyArray, Real
from plum import dispatch

import diffraxtra as dfxtra
import quaxed.numpy as jnp
from dataclassish.converters import dataclass

import galax._custom_types as gt
import galax.dynamics._src.custom_types as gdt
import galax.potential as gp
import galax.utils.loop_strategies as lstrat
from .df import AbstractKinematicDF, Fardal2015DF
from .integrate import integrate_orbit

SzTime3: TypeAlias = Real[Array, "time 3"]
SzN3: TypeAlias = Real[Array, "N 3"]
SzN6: TypeAlias = Real[Array, "N 6"]

default_solver = dfxtra.DiffEqSolver(
    solver=dfx.Dopri5(scan_kind="bounded"),
    stepsize_controller=dfx.PIDController(
        rtol=1e-7, atol=1e-7, dtmin=0.3, dtmax=None, force_dtmin=True, jump_ts=None
    ),
    max_steps=10_000,
    # adjoint=ForwardMode(),  # noqa: ERA001
)

default_kinematic_df = Fardal2015DF()


##############################################################################


@final
@partial(
    register_dataclass,
    data_fields=["release_times", "prog_mass", "qp_lead", "qp_trail"],
    meta_fields=[],
)
@dataclass
class StreamICs:
    """Initial conditions for the stream particles."""

    release_times: gt.SzTime
    prog_mass: gt.SzTime
    qp_lead: tuple[SzTime3, SzTime3]
    qp_trail: tuple[SzTime3, SzTime3]


# =========================================================


ICSScanIn: TypeAlias = tuple[gt.Sz0, gdt.Qarr, gdt.Parr, gt.Sz0]  # t, x, v, Msat
ICSScanOut: TypeAlias = tuple[gdt.Qarr, gdt.Parr, gdt.Qarr, gdt.Parr]  # x/v_l1, x/v_l2
ICSScanCarry: TypeAlias = tuple[PRNGKeyArray, Unpack[ICSScanOut]]


# TODO: rename to Fardal-like, since it can have different parameters?
@final
@partial(
    register_dataclass,
    data_fields=[],
    meta_fields=[],
)
@dataclass
class StreamSimulator:
    """Simulate a stellar stream.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax.random as jr
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.HernquistPotential(1e12, 10, units="galactic")
    >>> qp = (jnp.array([15.0, 0.0, 0.0]), jnp.array([0.0, 0.225, 0.0]))
    >>> stripping_times = jnp.linspace(-4_000, -150, 2_000)

    >>> stream_simulator = gd.experimental.stream.StreamSimulator()
    >>> prog_ics

    """

    @partial(jax.jit, static_argnames=("solver", "solver_kwargs"))
    def init(
        self,
        pot: gp.AbstractPotential,
        release_times: gt.SzTime,
        prog_w0: gdt.QParr,
        /,
        Msat: gt.LikeSz0 | gt.SzTime,  # can be time-dependent by matching release_times
        kinematic_df: AbstractKinematicDF | None = None,
        *,
        key: PRNGKeyArray,
        solver: dfxtra.AbstractDiffEqSolver = default_solver,
        solver_kwargs: dict[str, Any] | None = None,
    ) -> StreamICs:
        """Generate the initial conditions for the stream particles.

        This function generates the initial conditions for the stream particles
        given the progenitor's orbit, the release times, and the progenitor's
        mass.

        """
        # Integrate the progenitor's orbit to get the stream progenitor's positions
        # and velocities at the stream particle release times.
        prog_xs, prog_vs = integrate_orbit(
            pot,
            prog_w0,
            saveat=release_times,
            solver=solver,
            solver_kwargs=solver_kwargs,
        ).ys

        w0_df = default_kinematic_df if kinematic_df is None else kinematic_df

        # Define the scan function for generating the stream particle's initial
        # conditions given the release time, position, velocity, and Msat.
        def scan_fn(
            carry: ICSScanCarry, x: ICSScanIn
        ) -> tuple[ICSScanCarry, ICSScanOut]:
            key, subkey = jr.split(carry[0])
            xv_l12_new = w0_df.sample(subkey, pot, t=x[0], x=x[1], v=x[2], Msat=x[3])
            new_carry = (key, *xv_l12_new)
            return new_carry, xv_l12_new

        # Initial carry is [key, x_l1, v_l1, x_l2, v_l2]
        init_carry = (key, jnp.zeros(3), jnp.zeros(3), jnp.zeros(3), jnp.zeros(3))
        Msat = Msat * jnp.ones(len(release_times))  # shape match for scanning

        # Scan over the release times/xs/vs/ms to generate the stream particle's
        # initial conditions.
        _, all_states = jax.lax.scan(
            scan_fn, init_carry, (release_times, prog_xs, prog_vs, Msat)
        )
        return StreamICs(
            release_times,
            prog_mass=Msat,
            qp_lead=all_states[0:2],
            qp_trail=all_states[2:4],
        )

    @dispatch.abstract
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Simulate a stellar stream.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import galax.potential as gp
        >>> import galax.dynamics as gd

        >>> pot = gp.HernquistPotential(1e12, 10, units="galactic")
        >>> qp = (jnp.array([15.0, 0.0, 0.0]), jnp.array([0.0, 0.225, 0.0]))
        >>> ts = jnp.linspace(-4_000, -150, 2_000)
        >>> t1 = 0.0
        >>> Msat = 1e5

        >>> stream_simulator = gd.experimental.stream.StreamSimulator()

        >>> stream_lead, stream_trail = stream_simulator.run(
        ...     pot, qp, release_times=ts, t1=t1, Msat=Msat, key=jr.key(0))

        """


# ---------------------------
# auto-determine


@StreamSimulator.run.dispatch
def run(
    self: StreamSimulator,
    pot: gp.AbstractPotential,
    prog: StreamICs,
    /,
    *,
    t1: gt.LikeSz0,
    dense: bool = False,
    solver: dfxtra.AbstractDiffEqSolver = default_solver,
    solver_kwargs: dict[str, Any] | None = None,
) -> Any:
    return self.run(
        lstrat.Determine,
        pot,
        prog,
        t1=t1,
        dense=dense,
        solver=solver,
        solver_kwargs=solver_kwargs,
    )


@StreamSimulator.run.dispatch
def run(
    self: StreamSimulator,
    loop_strategy: type[lstrat.Determine],  # noqa: ARG001
    pot: gp.AbstractPotential,
    prog: StreamICs,
    /,
    *,
    t1: gt.LikeSz0,
    dense: bool = False,
    solver: dfxtra.AbstractDiffEqSolver = default_solver,
    solver_kwargs: dict[str, Any] | None = None,
) -> Any:
    # Determine the loop strategy
    loop_strat = lstrat.VMap  # TODO: an actual heuristic
    return self.run(
        loop_strat,
        pot,
        prog,
        t1=t1,
        dense=dense,
        solver=solver,
        solver_kwargs=solver_kwargs,
    )


# ---------------------------


StreamScanOut: TypeAlias = tuple[gdt.Qarr, gdt.Parr, gdt.Qarr, gdt.Parr]
StreamCarry: TypeAlias = tuple[int, Unpack[StreamScanOut]]


@StreamSimulator.run.dispatch
@partial(
    jax.jit,
    static_argnums=(1,),
    static_argnames=("dense", "solver", "solver_kwargs"),
)
def run(
    self: StreamSimulator,  # noqa: ARG001
    loop_strategy: type[lstrat.Scan],  # noqa: ARG001
    pot: gp.AbstractPotential,
    prog: StreamICs,
    /,
    *,
    t1: gt.LikeSz0,
    dense: bool = False,
    solver: dfxtra.AbstractDiffEqSolver = default_solver,
    solver_kwargs: dict[str, Any] | None = None,
) -> tuple[tuple[SzN3, SzN3], tuple[SzN3, SzN3]] | dfx.Solution:
    """Generate stellar stream.

    By scanning over the release model/integration. Better for CPU usage.

    """
    t1 = jnp.asarray(t1)
    x0s_l1, v0s_l1 = prog.qp_lead
    x0s_l2, v0s_l2 = prog.qp_trail

    @partial(jax.jit)
    @partial(jax.vmap, in_axes=((0, 0), None))  # map over stream arms
    def integrate_orbits(xv0: gdt.QParr, t0: gt.Sz0) -> gdt.QParr:
        soln: dfx.Solution = integrate_orbit(
            pot, xv0, t0, t1, dense=dense, solver=solver, solver_kwargs=solver_kwargs
        )
        return soln if dense else (soln.ys[0][-1], soln.ys[1][-1])  # return final xv

    @partial(jax.jit)  # scan over particles (release times)
    def scan_fun(
        carry: StreamCarry, _: int
    ) -> tuple[StreamCarry, StreamScanOut | tuple[dfx.Solution]]:
        i, x0_l1_i, v0_l1_i, x0_l2_i, v0_l2_i = carry

        xv0s_i = jnp.vstack([x0_l1_i, x0_l2_i]), jnp.vstack([v0_l1_i, v0_l2_i])
        soln = integrate_orbits(xv0s_i, prog.release_times[i])

        if dense:
            new_state = (soln,)
        else:
            xs_i, vs_i = soln
            new_state = (xs_i[0], vs_i[0], xs_i[1], vs_i[1])
        j = i + 1
        new_carry = (j, x0s_l1[j, :], v0s_l1[j, :], x0s_l2[j, :], v0s_l2[j, :])
        return new_carry, new_state

    init_carry = (0, x0s_l1[0, :], v0s_l1[0, :], x0s_l2[0, :], v0s_l2[0, :])
    idxs = jnp.arange(len(prog.release_times))
    _, all_states = jax.lax.scan(scan_fun, init_carry, idxs)

    if dense:
        out = all_states[0]
    else:
        q_lead, v_lead, q_trail, v_trail = all_states
        out = (q_lead, v_lead), (q_trail, v_trail)

    return out


@StreamSimulator.run.dispatch
@partial(
    jax.jit,
    static_argnums=(1,),
    static_argnames=("dense", "solver", "solver_kwargs"),
)
def run(
    self: StreamSimulator,  # noqa: ARG001
    loop_strategy: type[lstrat.VMap],  # noqa: ARG001
    pot: gp.AbstractPotential,
    prog: StreamICs,
    /,
    *,
    t1: gt.LikeSz0,
    dense: bool = False,
    solver: dfxtra.AbstractDiffEqSolver = default_solver,
    solver_kwargs: dict[str, Any] | None = None,
) -> tuple[tuple[SzN3, SzN3], tuple[SzN3, SzN3]] | dfx.Solution:
    t1 = jnp.asarray(t1)
    x0s_l1, v0s_l1 = prog.qp_lead
    x0s_l2, v0s_l2 = prog.qp_trail

    @partial(jax.jit)
    @partial(jax.vmap, in_axes=((0, 0), 0))  # map over particles
    def integrate_particle_orbit(
        xv0: gdt.QParr, t0: gt.Sz0
    ) -> dfx.Solution | gdt.QParr:
        soln = integrate_orbit(
            pot, xv0, t0, t1, solver=solver, solver_kwargs=solver_kwargs, dense=dense
        )
        return soln if dense else (soln.ys[0][-1], soln.ys[1][-1])  # return final xv

    x0s = jnp.stack([x0s_l1, x0s_l2], axis=0)
    v0s = jnp.stack([v0s_l1, v0s_l2], axis=0)
    integrate_particles = jax.vmap(integrate_particle_orbit, in_axes=(0, None))
    result = integrate_particles((x0s, v0s), prog.release_times)

    if dense:
        out = result
    else:
        w_lead = (result[0][0], result[1][0])  # x, v
        w_trail = (result[0][1], result[1][1])  # x, v
        out = (w_lead, w_trail)

    return out
