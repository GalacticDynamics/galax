"""galax: Galactic Dynamix in Jax."""

__all__ = ["ChenStreamDF"]


import functools as ft

from jaxtyping import PRNGKeyArray
from typing import final

import jax
import jax.random as jr

import coordinax as cx
import quaxed.numpy as jnp

import galax.dynamics.custom_types as gt
import galax.potential as gp
from .base import AbstractStreamDF
from galax.dynamics._src.cluster.radius import tidal_radius
from galax.orbits import specific_angular_momentum

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
class ChenStreamDF(AbstractStreamDF):
    """Chen Stream Distribution Function.

    A class for representing the Chen+2024 distribution function for
    generating stellar streams based on Chen et al. 2024
    https://ui.adsabs.harvard.edu/abs/2024arXiv240801496C/abstract

    The paper's default model integrates the particles in the host potential
    plus the progenitor's: pass ``progenitor_potential`` to
    `~galax.dynamics.MockStreamGenerator`, e.g.
    ``PlummerPotential(m_tot=prog_mass, r_s=u.Q(4, "pc"), units=...)`` as in
    the authors' notebooks (github.com/ybillchen/particle_spray). The paper
    uses a point mass and finds that a scale radius of 0-4 pc doesn't
    noticeably change the streams. Without a progenitor potential the stream
    is the paper's "no progenitor" variant, and
    `~galax.dynamics.MockStreamGenerator.run` warns.
    """

    @ft.partial(jax.jit, inline=True)
    def _sample(
        self,
        key: PRNGKeyArray,
        potential: gp.AbstractPotential,
        x: gt.BBtQuSz3,
        v: gt.BBtQuSz3,
        prog_mass: gt.BBtFloatQuSz0,
        t: gt.BBtFloatQuSz0,
    ) -> tuple[gt.BBtQuSz3, gt.BBtQuSz3, gt.BBtQuSz3, gt.BBtQuSz3]:
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

        def offsets(key: PRNGKeyArray, /) -> tuple[gt.BBtQuSz3, gt.BBtQuSz3]:
            """Position and velocity offsets (along x_new, y_new, z_new)."""
            # Bill Chen: method="cholesky" doesn't work here!
            posvel = jr.multivariate_normal(
                key, mean, cov, shape=r_tidal.shape, method="svd"
            )

            Dr = posvel[:, 0] * r_tidal

            v_esc = jnp.sqrt(2 * potential.constants["G"] * prog_mass / Dr)
            Dv = posvel[:, 3] * v_esc

            # Eq. 4 gives the angles in degrees.
            phi, theta = jnp.deg2rad(posvel[:, 1]), jnp.deg2rad(posvel[:, 2])
            alpha, beta = jnp.deg2rad(posvel[:, 4]), jnp.deg2rad(posvel[:, 5])

            ctheta, stheta = jnp.cos(theta), jnp.sin(theta)
            cphi, sphi = jnp.cos(phi), jnp.sin(phi)
            calpha, salpha = jnp.cos(alpha), jnp.sin(alpha)
            cbeta, sbeta = jnp.cos(beta), jnp.sin(beta)

            dx = jnp.stack([Dr * ctheta * cphi, Dr * ctheta * sphi, Dr * stheta], -1)
            dv = jnp.stack([Dv * cbeta * calpha, Dv * cbeta * salpha, Dv * sbeta], -1)
            return dx, dv

        # Each arm gets its own draw, as in the authors' code (and gala, galpy).
        key_trail, key_lead = jr.split(key)

        # Trailing arm
        dx, dv = offsets(key_trail)
        x_trail = (
            x + dx[:, 0:1] * x_new_hat + dx[:, 1:2] * y_new_hat + dx[:, 2:3] * z_new_hat
        )
        v_trail = (
            v + dv[:, 0:1] * x_new_hat + dv[:, 1:2] * y_new_hat + dv[:, 2:3] * z_new_hat
        )

        # Leading arm: the same DF rotated by 180 deg about z_new
        dx, dv = offsets(key_lead)
        x_lead = (
            x - dx[:, 0:1] * x_new_hat - dx[:, 1:2] * y_new_hat + dx[:, 2:3] * z_new_hat
        )
        v_lead = (
            v - dv[:, 0:1] * x_new_hat - dv[:, 1:2] * y_new_hat + dv[:, 2:3] * z_new_hat
        )

        return x_lead, v_lead, x_trail, v_trail
