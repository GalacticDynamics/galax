"""Energy diagnostics for N-Body dynamics."""

__all__ = ["total_energy"]

from jaxtyping import Array, Float
from typing import Any

import jax.numpy as jnp

import galax.potential as gp
from .field import NBodyField, _sources_targets
from .particles import Targets


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
    # Tracers are massless, so only the massive rows enter. They are dropped
    # before any arithmetic, so a NaN or inf tracer cannot reach the value or
    # its gradient (0 * NaN is NaN in both).
    q, sources, targets = _sources_targets(field, q)
    targets = Targets(
        q=targets.q[:nm], h=targets.h[:nm], self_index=targets.self_index[:nm]
    )
    active = field.active[:nm]
    m = sources.m  # massive masses, zero where inactive
    backend = field.backend
    # G multiplies outside the backend, as in `nbody_acceleration`.
    phi_self = field._G * backend.potential(  # noqa: SLF001
        backend.prepare(sources), targets, G=1.0
    )
    # `_sources_targets` already moved inactive rows to safe points.
    phi_ext = jnp.asarray(gp.potential(field.external_potential, q[:nm], t))
    # Inactive rows may hold NaN or inf velocities, so mask them first.
    p = jnp.where(active[:, None], p[:nm], 0.0)
    kinetic = 0.5 * jnp.sum(m * jnp.sum(p**2, axis=-1))
    self_pot = 0.5 * jnp.sum(m * phi_self)
    ext = m * phi_ext
    return kinetic + self_pot + jnp.sum(ext)
