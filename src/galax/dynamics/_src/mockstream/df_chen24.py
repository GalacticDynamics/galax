"""galax: Galactic Dynamix in Jax."""

__all__ = ["Chen24StreamDF"]


import warnings
from functools import partial
from typing import final

import jax
import jax.random as jr
from jaxtyping import PRNGKeyArray

import coordinax as cx
import quaxed.numpy as jnp

import galax._custom_types as gt
import galax.potential as gp
from .df_base import AbstractStreamDF
from galax.dynamics._src.cluster.radius import tidal_radius
from galax.dynamics._src.register_api import specific_angular_momentum

# ============================================================
# Constants

mean = jnp.array([1.6, -30, 0, 1, 20, 0])

cov = jnp.array(
    [
        [0.1225, 0, 0, 0, -4.9, 0],
        [0, 529, 0, 0, 0, 0],
        [0, 0, 144, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [-4.9, 0, 0, 0, 400, 0],
        [0, 0, 0, 0, 0, 484],
    ]
)

# ============================================================


@final
class Chen24StreamDF(AbstractStreamDF):
    """Chen Stream Distribution Function.

    A class for representing the Chen+2024 distribution function for
    generating stellar streams based on Chen et al. 2024
    https://ui.adsabs.harvard.edu/abs/2024arXiv240801496C/abstract
    """

    def __init__(self) -> None:
        super().__init__()
        warnings.warn(
            'Currently only the "no progenitor" version '
            "of the Chen+24 model is supported!",
            RuntimeWarning,
            stacklevel=1,
        )

    @partial(jax.jit, inline=True)
    def sample(
        self,
        key: PRNGKeyArray,
        potential: gp.AbstractPotential,
        x: gt.BBtQuSz3,
        v: gt.BBtQuSz3,
        prog_mass: gt.BBtFloatQuSz0,
        t: gt.BBtFloatQuSz0,
    ) -> tuple[gt.BtQuSz3, gt.BtQuSz3, gt.BtQuSz3, gt.BtQuSz3]:
        """Generate stream particle initial conditions."""
        # Random number generation

        # x_new-hat
        r = jnp.linalg.vector_norm(x, axis=-1, keepdims=True)
        x_new_hat = x / r

        # z_new-hat
        L_vec = specific_angular_momentum(x, v)
        z_new_hat = cx.vecs.normalize_vector(L_vec)

        # y_new-hat
        phi_vec = v - jnp.sum(v * x_new_hat, axis=-1, keepdims=True) * x_new_hat
        y_new_hat = cx.vecs.normalize_vector(phi_vec)

        r_tidal = tidal_radius(potential, x, v, mass=prog_mass, t=t)

        # Bill Chen: method="cholesky" doesn't work here!
        posvel = jr.multivariate_normal(
            key, mean, cov, shape=r_tidal.shape, method="svd"
        )

        Dr = posvel[:, 0] * r_tidal

        v_esc = jnp.sqrt(2 * potential.constants["G"] * prog_mass / Dr)
        Dv = posvel[:, 3] * v_esc

        # convert degrees to radians
        phi = posvel[:, 1] * 0.017453292519943295
        theta = posvel[:, 2] * 0.017453292519943295
        alpha = posvel[:, 4] * 0.017453292519943295
        beta = posvel[:, 5] * 0.017453292519943295

        ctheta, stheta = jnp.cos(theta), jnp.sin(theta)
        cphi, sphi = jnp.cos(phi), jnp.sin(phi)
        calpha, salpha = jnp.cos(alpha), jnp.sin(alpha)
        cbeta, sbeta = jnp.cos(beta), jnp.sin(beta)

        # Trailing arm
        x_trail = (
            x
            + (Dr * ctheta * cphi)[:, None] * x_new_hat
            + (Dr * ctheta * sphi)[:, None] * y_new_hat
            + (Dr * stheta)[:, None] * z_new_hat
        )
        v_trail = (
            v
            + (Dv * cbeta * calpha)[:, None] * x_new_hat
            + (Dv * cbeta * salpha)[:, None] * y_new_hat
            + (Dv * sbeta)[:, None] * z_new_hat
        )

        # Leading arm
        x_lead = (
            x
            - (Dr * ctheta * cphi)[:, None] * x_new_hat
            - (Dr * ctheta * sphi)[:, None] * y_new_hat
            + (Dr * stheta)[:, None] * z_new_hat
        )
        v_lead = (
            v
            - (Dv * cbeta * calpha)[:, None] * x_new_hat
            - (Dv * cbeta * salpha)[:, None] * y_new_hat
            + (Dv * sbeta)[:, None] * z_new_hat
        )

        return x_lead, v_lead, x_trail, v_trail
