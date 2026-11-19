"""Energy diagnostics for N-Body dynamics."""

__all__ = ["total_energy"]

from jaxtyping import Array, Float
from typing import Any

import jax.numpy as jnp

import galax.potential as gp
from .field import NBodyField, _sources_targets


def total_energy(
    field: NBodyField, t: Any, q: Float[Array, "N 3"], p: Float[Array, "N 3"]
) -> Float[Array, ""]:
    r"""Return the total energy of the active particles in ``field``.

    The energy is the kinetic energy ``1/2 sum m |p|^2``, plus the self
    potential energy ``1/2 sum_i m_i Phi_self,i``, plus the external energy
    ``sum m_i Phi_ext(q_i, t)``. Here ``p`` is the velocity. ``t``, ``q`` and
    ``p`` are plain arrays in the field's units. Massless tracers and inactive
    particles contribute nothing.

    The energy is conserved only with a static external potential, and
    leapfrog keeps it constant only to O(dt^2).
    """
    nm = field.n_massive
    m = field.masses * field.active
    q, sources, targets = _sources_targets(field, q)
    backend = field.backend
    # G multiplies outside the backend, as in `nbody_acceleration`.
    phi_self = field._G * backend.potential(  # noqa: SLF001
        backend.prepare(sources), targets, G=1.0
    )
    phi_ext = jnp.asarray(gp.potential(field.external_potential, q, t))
    # Inactive rows may hold NaN or inf, and 0 * NaN is NaN, so mask them.
    p = jnp.where(field.active[:, None], p, 0.0)
    kinetic = 0.5 * jnp.sum(m * jnp.sum(p**2, axis=-1))
    self_pot = 0.5 * jnp.sum(m[:nm] * phi_self[:nm])
    ext = jnp.where(field.active, m * phi_ext, 0.0)
    return kinetic + self_pot + jnp.sum(ext)
