"""Tests for `galax.nbody.NBodyField`."""

import jax.numpy as jnp

import unxt as u

import galax.nbody as gn
import galax.potential as gp


def test_pairwise_accelerations_conserve_momentum() -> None:
    """With no external potential, the mass-weighted accelerations sum to zero."""
    field = gn.NBodyField(
        masses=u.Q([1.0, 2.0, 3.0], "Msun"),
        eps=u.Q(1e-3, "AU"),
        external_potential=gp.NullPotential(units="solarsystem"),
    )
    q = u.Q([[0.0, 0, 0], [1, 0, 0], [0, 2, 1]], "AU")
    p = u.Q(jnp.zeros((3, 3)), "km/s")

    v, a = field(0.0, (q, p), None)

    assert jnp.allclose(v, 0)
    assert jnp.allclose(jnp.sum(jnp.asarray([1.0, 2, 3])[:, None] * a, axis=0), 0)
    # The two closest bodies pull toward each other.
    assert a[0, 0] > 0
    assert a[1, 0] < 0
