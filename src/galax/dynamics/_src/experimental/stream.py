"""Experimental dynamics."""

__all__: list[str] = []

from functools import partial
from typing import Any, TypeAlias

import diffrax as dfx
import equinox as eqx
import jax
from jaxtyping import Array, Real

import diffraxtra as dfxtra
import quaxed.numpy as jnp

import galax._custom_types as gt
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


@partial(eqx.filter_jit)
def release_model(
    pot: gp.AbstractPotential,
    x: gt.Sz3,
    v: gt.Sz3,
    Msat: gt.LikeSz0,
    i: int,
    t: gt.LikeSz0,
    seed_num: int,
    kval_arr: Real[Array, "8"] | gt.Sz0 | float = 1.0,
) -> tuple[gt.Sz3, gt.Sz3, gt.Sz3, gt.Sz3]:
    # ---------------------------------
    # if kval_arr is a scalar, then we assume the default values of kvals
    pred = jnp.isscalar(kval_arr)

    def true_func() -> Real[Array, "8"]:
        return jnp.array([2.0, 0.3, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])

    def false_func() -> Real[Array, "8"]:
        return jnp.ones(8) * kval_arr

    kval_arr = jax.lax.cond(pred, true_func, false_func)
    kr_bar, kvphi_bar, kz_bar, kvz_bar, sigma_kr, sigma_kvphi, sigma_kz, sigma_kvz = (
        kval_arr
    )

    # ---------------------------------

    key_master = jax.random.PRNGKey(seed_num)
    random_ints = jax.random.randint(key=key_master, shape=(5,), minval=0, maxval=1000)

    keya = jax.random.PRNGKey(i * random_ints[0])
    keyb = jax.random.PRNGKey(i * random_ints[1])

    keyc = jax.random.PRNGKey(i * random_ints[2])
    keyd = jax.random.PRNGKey(i * random_ints[3])

    omega_val = omega(x, v)

    r = jnp.linalg.norm(x)
    r_hat = x / r
    r_tidal = tidal_radius(pot, x, v, mass=Msat, t=t)
    rel_v = omega_val * r_tidal  # relative velocity

    # circlar_velocity
    v_circ = rel_v  # jnp.sqrt( r*dphi_dr )

    L_vec = jnp.cross(x, v)
    z_hat = L_vec / jnp.linalg.norm(L_vec)

    phi_vec = v - jnp.sum(v * r_hat) * r_hat
    phi_hat = phi_vec / jnp.linalg.norm(phi_vec)

    kr_samp = kr_bar + jax.random.normal(keya, shape=(1,)) * sigma_kr
    kvphi_samp = kr_samp * (
        kvphi_bar + jax.random.normal(keyb, shape=(1,)) * sigma_kvphi
    )
    kz_samp = kz_bar + jax.random.normal(keyc, shape=(1,)) * sigma_kz
    kvz_samp = kvz_bar + jax.random.normal(keyd, shape=(1,)) * sigma_kvz

    ## Trailing arm
    pos_trail = x + kr_samp * r_hat * (r_tidal)
    pos_trail = pos_trail + z_hat * kz_samp * (r_tidal / 1.0)
    v_trail = v + (0.0 + kvphi_samp * v_circ * (1.0)) * phi_hat
    v_trail = v_trail + (kvz_samp * v_circ * (1.0)) * z_hat

    # Leading arm
    pos_lead = x + kr_samp * r_hat * (-r_tidal)  # nudge in
    pos_lead = pos_lead + z_hat * kz_samp * (-r_tidal / 1.0)
    v_lead = v + (0.0 + kvphi_samp * v_circ * (-1.0)) * phi_hat
    v_lead = v_lead + (kvz_samp * v_circ * (-1.0)) * z_hat

    return pos_lead, pos_trail, v_lead, v_trail


@partial(jax.jit, static_argnames=("solver", "solver_kwargs"))
def gen_stream_ics(
    pot: gp.AbstractPotential,
    ts: gt.SzTime,
    prog_w0: gt.Sz6,
    /,
    Msat: gt.LikeSz0,
    seed_num: int,
    kval_arr: Real[Array, "8"] | gt.Sz0 | float = 1.0,
    solver: dfxtra.AbstractDiffEqSolver = default_solver,
    solver_kwargs: dict[str, Any] | None = None,
) -> tuple[SzN3, SzN3, SzN3, SzN3]:
    ws_jax = integrate_orbit(
        pot,
        (prog_w0[..., :3], prog_w0[..., 3:]),
        ts,
        solver=solver,
        solver_kwargs=solver_kwargs,
    ).ys
    Msat = Msat * jnp.ones(len(ts))

    Carry: TypeAlias = tuple[int, gt.Sz3, gt.Sz3, gt.Sz3, gt.Sz3]
    State: TypeAlias = tuple[gt.Sz3, gt.Sz3, gt.Sz3, gt.Sz3]

    def scan_fun(carry: Carry, t: gt.Sz0) -> tuple[Carry, State]:
        i = carry[0]
        x_L1_new, x_L2_new, v_L1_new, v_L2_new = release_model(
            pot,
            ws_jax[0][i],
            ws_jax[1][i],
            Msat=Msat[i],
            i=i,
            t=t,
            seed_num=seed_num,
            kval_arr=kval_arr,
        )
        new_carry = (i + 1, x_L1_new, x_L2_new, v_L1_new, v_L2_new)
        state = (x_L1_new, x_L2_new, v_L1_new, v_L2_new)
        return new_carry, state

    init_carry: Carry = (
        0,
        jnp.zeros(3, dtype=float),
        jnp.zeros(3, dtype=float),
        jnp.zeros(3, dtype=float),
        jnp.zeros(3, dtype=float),
    )
    _, all_states = jax.lax.scan(scan_fun, init_carry, ts)
    x_close_arr, x_far_arr, v_close_arr, v_far_arr = all_states

    return x_close_arr, x_far_arr, v_close_arr, v_far_arr


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
    x_close_arr, x_far_arr, v_close_arr, v_far_arr = gen_stream_ics(
        pot,
        ts,
        prog_w0,
        Msat=Msat,
        seed_num=seed_num,
        kval_arr=kval_arr,
        solver=solver,
        solver_kwargs=solver_kwargs,
    )

    def orb_integrator(w0: gt.Sz6, ts: gt.SzTime) -> SzN6:
        ys = integrate_orbit(
            pot,
            (w0[..., :3], w0[..., 3:]),
            ts,
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
