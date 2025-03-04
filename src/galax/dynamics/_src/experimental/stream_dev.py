"""Experimental dynamics."""

__all__: list[str] = []

from functools import partial

import jax
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, Real

import quaxed.numpy as jnp

import galax._custom_types as gt
import galax.potential as gp
from galax.dynamics._src.api import omega
from galax.dynamics._src.cluster.api import tidal_radius


@partial(jax.jit)
def release_model(
    rng: PRNGKeyArray,  # consumed
    pot: gp.AbstractPotential,
    /,
    x: gt.Sz3,
    v: gt.Sz3,
    Msat: gt.LikeSz0,
    t: gt.LikeSz0,
    kval_arr: Real[Array, "8"] | gt.Sz0 | float = 1.0,
) -> tuple[gt.Sz3, gt.Sz3, gt.Sz3, gt.Sz3]:
    # -------------------
    # If kval_arr is a scalar, then we assume the default values of kvals
    pred = jnp.isscalar(kval_arr)

    def true_func() -> Real[Array, "8"]:
        return jnp.array([2.0, 0.3, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])

    def false_func() -> Real[Array, "8"]:
        return jnp.ones(8) * kval_arr

    kval_arr = jax.lax.cond(pred, true_func, false_func)
    kr_bar, kvphi_bar, kz_bar, kvz_bar, sigma_kr, sigma_kvphi, sigma_kz, sigma_kvz = (
        kval_arr
    )

    # -------------------

    skeys = jr.split(rng, 4)

    omega_val = omega(x, v)

    r = jnp.linalg.norm(x)
    r_hat = x / r
    r_tidal = tidal_radius(pot, x, v, mass=Msat, t=t)
    rel_v = omega_val * r_tidal  # relative velocity

    # circular
    v_circ = rel_v  # jnp.sqrt( r*dphi_dr ) ?

    L_vec = jnp.cross(x, v)
    z_hat = L_vec / jnp.linalg.norm(L_vec)

    phi_vec = v - jnp.sum(v * r_hat) * r_hat
    phi_hat = phi_vec / jnp.linalg.norm(phi_vec)

    kr_samp = kr_bar + jr.normal(skeys[0], shape=(1,)) * sigma_kr
    kvphi_samp = kr_samp * (kvphi_bar + jr.normal(skeys[1], shape=(1,)) * sigma_kvphi)
    kz_samp = kz_bar + jr.normal(skeys[2], shape=(1,)) * sigma_kz
    kvz_samp = kvz_bar + jr.normal(skeys[3], shape=(1,)) * sigma_kvz

    # Trailing arm
    pos_trail = x + kr_samp * r_hat * (r_tidal)  # nudge out
    pos_trail = pos_trail + z_hat * kz_samp * (r_tidal / 1.0)
    v_trail = v + (0.0 + kvphi_samp * v_circ * (1.0)) * phi_hat
    v_trail = v_trail + (kvz_samp * v_circ * (1.0)) * z_hat

    # Leading arm
    pos_lead = x + kr_samp * r_hat * (-r_tidal)  # nudge in
    pos_lead = pos_lead + z_hat * kz_samp * (-r_tidal / 1.0)
    v_lead = v + (0.0 + kvphi_samp * v_circ * (-1.0)) * phi_hat
    v_lead = v_lead + (kvz_samp * v_circ * (-1.0)) * z_hat

    return pos_lead, pos_trail, v_lead, v_trail
