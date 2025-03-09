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
    >>> qp0 = (jnp.array([15.0, 0.0, 0.0]), jnp.array([0.0, 0.225, 0.0]))
    >>> t0 = 0.0  # now
    >>> release_times = jnp.linspace(-4_000, -150, 2_000)

    >>> stream_simulator = gd.experimental.stream.StreamSimulator()
    >>> prog_ics = stream_simulator.init(pot, qp0, t0,
    ...     release_times=release_times, Msat=1e5, key=jr.key(0))
    >>> prog_ics
    StreamICs(release_times=Array([-4000. , ...  -150. ], dtype=float64),
        prog_mass=Array([100000., ... 100000.], dtype=float64),
        qp_lead=(Array([[-1.07924439e+01, -7.37489800e+00, -1.80585858e-02],
                        ...
                        [-4.72861830e+00,  1.40355376e+01, -2.59068091e-02]], dtype=float64),
                 Array([[ 4.88701470e-02, -2.75920181e-01, -3.55040709e-04],
                        ...
                        [-2.10197865e-01, -8.18185909e-02, -1.29460838e-04]], dtype=float64)),
        qp_trail=(Array([[-1.09735894e+01, -7.49868179e+00, -1.80585858e-02],
                        ...
                        [-4.84009594e+00,  1.43664267e+01, -2.59068091e-02]], dtype=float64),
                 Array([[ 4.96114690e-02, -2.77005033e-01, -3.55040709e-04],
                        ...
                        [-2.09998407e-01, -8.17513929e-02, -1.29460838e-04]],  dtype=float64)))

    >>> stream_lead, stream_trail = stream_simulator.run(pot, prog_ics, t1=t0)
    >>> stream_lead
    (Array([[ 8.17363908e+00,  7.06167200e+00,  1.57614710e-02],
            ...,
            [ 1.48119877e+01,  3.65839156e-01,  1.69433893e-02]],      dtype=float64),
     Array([[-3.23020402e-01,  1.29335442e-01, -4.61335351e-05],
            ...,
            [-1.35276009e-02,  2.24964959e-01, -3.41789771e-04]],      dtype=float64))

    """  # noqa: E501

    @partial(jax.jit, static_argnames=("solver", "solver_kwargs"))
    def init(
        self,
        pot: gp.AbstractPotential,
        prog_w0: gdt.QParr,
        prog_t0: gt.LikeSz0,
        /,
        release_times: gt.SzTime,
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
        # Sort the stripping times in ascending order.
        release_times = jnp.sort(release_times)

        # Integrate the progenitor from `t0` to the start of the release times.
        # Then when it's integrated over the release times, it will be at the
        # correct position. Note: diffrax is fine to integrate with t0 = t1
        prog_w0 = integrate_orbit(
            pot,
            prog_w0,
            t0=prog_t0,
            t1=release_times[0],
            solver=solver,
            solver_kwargs=solver_kwargs,
        ).ys
        prog_w0 = (prog_w0[0][-1], prog_w0[1][-1])  # rm the extra t batch dim

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
        >>> qp0 = (jnp.array([15.0, 0.0, 0.0]), jnp.array([0.0, 0.225, 0.0]))
        >>> t0 = 0.0
        >>> release_times = jnp.linspace(-4_000, -150, 2_000)
        >>> Msat = 1e5

        >>> stream_simulator = gd.experimental.stream.StreamSimulator()
        >>> prog_ics = stream_simulator.init(pot, qp0, t0,
        ...     release_times=release_times, Msat=1e5, key=jr.key(0))

        >>> stream_lead, stream_trail = stream_simulator.run(pot, prog_ics, t1=t0)

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
