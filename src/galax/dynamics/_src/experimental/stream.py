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
def gen_stream_ics(
    pot: gp.AbstractPotential,
    ts: gt.SzTime,
    prog_w0: gdt.QParr,
    /,
    Msat: gt.LikeSz0,
    kval_arr: Real[Array, "8"] | gt.Sz0 | float = 1.0,
    *,
    key: PRNGKeyArray,
    solver: dfxtra.AbstractDiffEqSolver = default_solver,
    solver_kwargs: dict[str, Any] | None = None,
) -> tuple[SzN3, SzN3, SzN3, SzN3]:
    prog_xs, prog_vs = integrate_orbit(
        pot,
        prog_w0,
        saveat=ts,
        solver=solver,
        solver_kwargs=solver_kwargs,
    ).ys
    Msat = Msat * jnp.ones(len(ts))

    def scan_fn(carry: ICSScanCarry, x: ICSScanIn) -> tuple[ICSScanCarry, ICSScanOut]:
        key, subkey = jr.split(carry[0])
        xv_l12_new = fardal2015_release_model(
            subkey, pot, t=x[0], x=x[1], v=x[2], Msat=x[3], kval_arr=kval_arr
        )
        new_carry = (key, *xv_l12_new)
        return new_carry, xv_l12_new

    init_carry = (key, jnp.zeros(3), jnp.zeros(3), jnp.zeros(3), jnp.zeros(3))
    _, all_states = jax.lax.scan(scan_fn, init_carry, (ts, prog_xs, prog_vs, Msat))
    return cast(ICSScanOut, all_states)


##############################################################################


@partial(jax.jit, static_argnames=("solver", "solver_kwargs"))
def gen_stream_scan(
    pot: gp.AbstractPotential,
    prog_w0: gt.Sz6,
    ts: gt.SzTime,
    /,
    Msat: gt.LikeSz0,
    seed_num: int,
    kval_arr: Real[Array, "8"] | gt.Sz0 | float = 1.0,
    solver: dfxtra.AbstractDiffEqSolver = default_solver,
    solver_kwargs: dict[str, Any] | None = None,
) -> tuple[SzN6, SzN6]:
    """Generate stellar stream.

    By scanning over the release model/integration. Better for CPU usage.

    """
    x_close_arr, v_close_arr, x_far_arr, v_far_arr = gen_stream_ics(
        pot,
        ts,
        (prog_w0[..., :3], prog_w0[..., 3:]),
        Msat=Msat,
        kval_arr=kval_arr,
        key=jr.key(seed_num),
        solver=solver,
        solver_kwargs=solver_kwargs,
    )

    def orb_integrator(w0: gt.Sz6, ts: gt.SzTime) -> SzN6:
        ys = integrate_orbit(
            pot,
            (w0[..., :3], w0[..., 3:]),
            saveat=ts,
            solver=solver,
            solver_kwargs=solver_kwargs,
            dense=False,
        ).ys
        return jnp.concat((ys[0][-1, :], ys[1][-1, :]), axis=-1)

    orb_integrator_mapped = jax.jit(jax.vmap(orb_integrator, in_axes=(0, None)))

    Carry: TypeAlias = tuple[int, gt.Sz3, gt.Sz3, gt.Sz3, gt.Sz3]
    State: TypeAlias = tuple[gt.Sz6, gt.Sz6]

    @partial(jax.jit)
    def scan_fun(carry: Carry, _: int) -> tuple[Carry, State]:
        i, x0_close_i, x0_far_i, v0_close_i, v0_far_i = carry
        curr_particle_w0_close = jnp.hstack([x0_close_i, v0_close_i])
        curr_particle_w0_far = jnp.hstack([x0_far_i, v0_far_i])

        ts_arr = jnp.array([ts[i], ts[-1]])
        curr_particle_loc = jnp.vstack([curr_particle_w0_close, curr_particle_w0_far])
        w_particle = orb_integrator_mapped(curr_particle_loc, ts_arr)

        w_particle_close = w_particle[0]
        w_particle_far = w_particle[1]

        new_carry = (
            i + 1,
            x_close_arr[i + 1, :],
            x_far_arr[i + 1, :],
            v_close_arr[i + 1, :],
            v_far_arr[i + 1, :],
        )
        new_state = (w_particle_close, w_particle_far)
        return new_carry, new_state

    init_carry = (
        0,
        x_close_arr[0, :],
        x_far_arr[0, :],
        v_close_arr[0, :],
        v_far_arr[0, :],
    )
    # Particle ids is one less than len(ts): ts[-1] defines final time to
    # integrate up to the observed time
    particle_ids = jnp.arange(len(x_close_arr) - 1)
    _, all_states = jax.lax.scan(scan_fun, init_carry, particle_ids)
    lead_arm, trail_arm = all_states

    return lead_arm, trail_arm
