"""Experimental dynamics."""

__all__: list[str] = []

from functools import partial
from typing import Any, TypeAlias, cast
from typing_extensions import Unpack

import diffrax as dfx
import jax
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, Real

import coordinax as cx
import diffraxtra as dfxtra
import quaxed.numpy as jnp

import galax._custom_types as gt
import galax.dynamics._src.custom_types as gdt
import galax.potential as gp
from .integrate import integrate_orbit
from galax.dynamics._src.api import omega
from galax.dynamics._src.cluster.api import tidal_radius

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


@partial(jax.jit)
def fardal2015_release_model(
    key: PRNGKeyArray,
    pot: gp.AbstractPotential,
    t: gt.LikeSz0,
    x: gt.Sz3,
    v: gt.Sz3,
    Msat: gt.LikeSz0,
    kval_arr: Real[Array, "8"] | gt.Sz0 | float = 1.0,
) -> tuple[gt.Sz3, gt.Sz3, gt.Sz3, gt.Sz3]:
    # ---------------------------------
    # If kval_arr is a scalar, then we assume the default values of kvals

    def true_func() -> Real[Array, "8"]:
        return jnp.array([2.0, 0.3, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])

    def false_func() -> Real[Array, "8"]:
        return jnp.ones(8) * kval_arr

    kr_bar, kvphi_bar, kz_bar, kvz_bar, sigma_kr, sigma_kvphi, sigma_kz, sigma_kvz = (
        jax.lax.cond(jnp.isscalar(kval_arr), true_func, false_func)
    )

    # ---------------------------------

    key1, key2, key3, key4 = jr.split(key, 4)

    Omega = omega(x, v)  # orbital angular frequency about the origin
    r_tidal = tidal_radius(pot, x, v, mass=Msat, t=t)  # tidal radius
    v_circ = Omega * r_tidal  # relative velocity

    # unit vectors
    r_hat = cx.vecs.normalize_vector(x)
    z_hat = cx.vecs.normalize_vector(jnp.linalg.cross(x, v))
    phi_vec = v - jnp.sum(v * r_hat) * r_hat
    phi_hat = cx.vecs.normalize_vector(phi_vec)

    # k vals
    shape = r_tidal.shape
    kr_samp = kr_bar + jr.normal(key1, shape) * sigma_kr
    kvphi_samp = kr_samp * (kvphi_bar + jr.normal(key2, shape) * sigma_kvphi)
    kz_samp = kz_bar + jr.normal(key3, shape) * sigma_kz
    kvz_samp = kvz_bar + jr.normal(key4, shape) * sigma_kvz

    # Leading arm
    x_lead = x - r_tidal * (kr_samp * r_hat - kz_samp * z_hat)
    v_lead = v - v_circ * (kvphi_samp * phi_hat - kvz_samp * z_hat)

    # Trailing arm
    x_trail = x + r_tidal * (kr_samp * r_hat + kz_samp * z_hat)
    v_trail = v + v_circ * (kvphi_samp * phi_hat + kvz_samp * z_hat)

    return x_lead, v_lead, x_trail, v_trail


##############################################################################

ICSScanIn: TypeAlias = tuple[gt.Sz0, gdt.Qarr, gdt.Parr, gt.Sz0]  # t, x, v, Msat
ICSScanOut: TypeAlias = tuple[gdt.Qarr, gdt.Parr, gdt.Qarr, gdt.Parr]  # x/v_l1, x/v_l2
ICSScanCarry: TypeAlias = tuple[PRNGKeyArray, Unpack[ICSScanOut]]


@partial(jax.jit, static_argnames=("solver", "solver_kwargs"))
def generate_stream_ics(
    pot: gp.AbstractPotential,
    ts: gt.SzTime,
    prog_w0: gdt.QParr,
    /,
    Msat: gt.LikeSz0 | gt.SzTime,
    kval_arr: Real[Array, "8"] | gt.Sz0 | float = 1.0,
    *,
    key: PRNGKeyArray,
    solver: dfxtra.AbstractDiffEqSolver = default_solver,
    solver_kwargs: dict[str, Any] | None = None,
) -> tuple[SzN3, SzN3, SzN3, SzN3]:
    # Integrate the progenitor's orbit to get the stream progenitor's positions
    # and velocities at the stream particle release times.
    prog_xs, prog_vs = integrate_orbit(
        pot,
        prog_w0,
        saveat=ts,
        solver=solver,
        solver_kwargs=solver_kwargs,
    ).ys

    # Define the scan function for generating the stream particle's initial
    # conditions given the release time, position, velocity, and Msat.
    def scan_fn(carry: ICSScanCarry, x: ICSScanIn) -> tuple[ICSScanCarry, ICSScanOut]:
        key, subkey = jr.split(carry[0])
        xv_l12_new = fardal2015_release_model(
            subkey, pot, t=x[0], x=x[1], v=x[2], Msat=x[3], kval_arr=kval_arr
        )
        new_carry = (key, *xv_l12_new)
        return new_carry, xv_l12_new

    # Initial carry is [key, x_l1, v_l1, x_l2, v_l2]
    init_carry = (key, jnp.zeros(3), jnp.zeros(3), jnp.zeros(3), jnp.zeros(3))
    Msat = Msat * jnp.ones(len(ts))  # shape match for scanning

    # Scan over the release times/xs/vs/ms to generate the stream particle's
    # initial conditions.
    _, all_states = jax.lax.scan(scan_fn, init_carry, (ts, prog_xs, prog_vs, Msat))
    return cast(ICSScanOut, all_states)


##############################################################################


StreamScanOut: TypeAlias = tuple[gdt.Qarr, gdt.Parr, gdt.Qarr, gdt.Parr]
StreamCarry: TypeAlias = tuple[int, Unpack[StreamScanOut]]


@partial(jax.jit, static_argnames=("solver", "solver_kwargs"))
def simulate_stream_scan(
    pot: gp.AbstractPotential,
    prog_w0: gdt.QParr,
    /,
    release_times: gt.SzTime,
    t1: gt.LikeSz0,
    Msat: gt.LikeSz0,
    kval_arr: Real[Array, "8"] | gt.Sz0 | float = 1.0,
    *,
    key: PRNGKeyArray,
    solver: dfxtra.AbstractDiffEqSolver = default_solver,
    solver_kwargs: dict[str, Any] | None = None,
) -> tuple[tuple[SzN3, SzN3], tuple[SzN3, SzN3]]:
    """Generate stellar stream.

    By scanning over the release model/integration. Better for CPU usage.

    """
    t1 = jnp.asarray(t1)
    x0s_l1, v0s_l1, x0s_l2, v0s_l2 = generate_stream_ics(  # x/v_l1/2 shape (N, 3)
        pot,
        jnp.concatenate([release_times, t1[None]]),
        prog_w0,
        Msat=Msat,
        kval_arr=kval_arr,
        key=key,
        solver=solver,
        solver_kwargs=solver_kwargs,
    )

    @partial(jax.jit)
    @partial(jax.vmap, in_axes=((0, 0), None))  # map over stream arms
    def integrate_orbits(xv0: gdt.QParr, t0: gt.Sz0) -> gdt.QParr:
        ys: gdt.QParr = integrate_orbit(
            pot, xv0, t0, t1, solver=solver, solver_kwargs=solver_kwargs
        ).ys
        return (ys[0][-1], ys[1][-1])  # return final xv

    @partial(jax.jit)  # scan over particles (release times)
    def scan_fun(carry: StreamCarry, _: int) -> tuple[StreamCarry, StreamScanOut]:
        i, x0_l1_i, v0_l1_i, x0_l2_i, v0_l2_i = carry

        xv0s_i = jnp.vstack([x0_l1_i, x0_l2_i]), jnp.vstack([v0_l1_i, v0_l2_i])
        xs_i, vs_i = integrate_orbits(xv0s_i, release_times[i])

        j = i + 1
        new_carry = (j, x0s_l1[j, :], v0s_l1[j, :], x0s_l2[j, :], v0s_l2[j, :])
        return new_carry, (xs_i[0], vs_i[0], xs_i[1], vs_i[1])

    init_carry = (0, x0s_l1[0, :], v0s_l1[0, :], x0s_l2[0, :], v0s_l2[0, :])
    idxs = jnp.arange(len(release_times))
    _, all_states = jax.lax.scan(scan_fun, init_carry, idxs)
    q_lead, v_lead, q_trail, v_trail = all_states

    return (q_lead, v_lead), (q_trail, v_trail)


@partial(jax.jit, static_argnames=("solver", "solver_kwargs"))
def simulate_stream_vmap(
    pot: gp.AbstractPotential,
    prog_w0: gdt.QParr,
    /,
    release_times: gt.SzTime,
    t1: gt.LikeSz0,
    Msat: gt.LikeSz0,
    kval_arr: Real[Array, "8"] | gt.Sz0 | float = 1.0,
    *,
    key: PRNGKeyArray,
    solver: dfxtra.AbstractDiffEqSolver = default_solver,
    solver_kwargs: dict[str, Any] | None = None,
) -> tuple[tuple[SzN3, SzN3], tuple[SzN3, SzN3]]:
    t1 = jnp.array(t1)
    x0s_l1, v0s_l1, x0s_l2, v0s_l2 = generate_stream_ics(  # x/v_l1/2 shape (N, 3)
        pot,
        jnp.concatenate([release_times, t1[None]]),
        prog_w0,
        Msat=Msat,
        kval_arr=kval_arr,
        key=key,
        solver=solver,
        solver_kwargs=solver_kwargs,
    )

    @partial(jax.jit)
    @partial(jax.vmap, in_axes=((0, 0), 0))  # map over particles
    def integrate_particle_orbit(xv0: gdt.QParr, t0: gt.Sz0) -> gdt.QParr:
        ys: gdt.QParr = integrate_orbit(
            pot, xv0, t0, t1, solver=solver, solver_kwargs=solver_kwargs, dense=False
        ).ys
        return (ys[0][-1], ys[1][-1])  # return final xv

    x0s = jnp.stack([x0s_l1[:-1], x0s_l2[:-1]], axis=0)
    v0s = jnp.stack([v0s_l1[:-1], v0s_l2[:-1]], axis=0)
    integrate_particles = jax.vmap(integrate_particle_orbit, in_axes=(0, None))
    xs, vs = integrate_particles((x0s, v0s), release_times)
    w_lead, w_trail = (xs[0], vs[0]), (xs[1], vs[1])

    return w_lead, w_trail
