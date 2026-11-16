"""Tests for `galax.dynamics.cluster.ReleaseTimeSampler`."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import unxt as u

import galax.dynamics.cluster as gdc


class ExpMassLoss(gdc.AbstractMassRateField):
    """dM/dt = -M / (10 Myr), written in yr."""

    units: u.AbstractUnitSystem = eqx.field(
        default=u.unitsystem("solarsystem"), converter=u.unitsystem, static=True
    )

    def __call__(self, t, M, args, /, **kw):
        return -1e-7 * M


def test_sample_reads_mass_history_in_field_units():
    """The sampler evaluates m(t) in the units it was solved in.

    The field is in yr and the sampler in galactic units (Myr). M(t) ∝ exp(-t/τ)
    with τ = 10 Myr, so the escape rate is too, and its inverse CDF is
    t = -τ ln(1 - U (1 - exp(-T/τ))) for T = 10 Myr.
    """
    field = ExpMassLoss()
    t0, t1 = u.Q(0.0, "Myr"), u.Q(10.0, "Myr")
    m_of_t = gdc.MassSolver().solve(
        field, u.Q(1e4, "Msun"), t0, t1, dense=True, vectorize_interpolation=True
    )
    sampler = gdc.ReleaseTimeSampler(dm_dt=field, m_of_t=m_of_t, units="galactic")

    key = jr.PRNGKey(0)
    got = sampler.sample(key, t0, t1, n_stars=5, mass_params={})

    U = jr.uniform(key, shape=(5,))
    want = jnp.sort(-10.0 * jnp.log1p(-U * (1 - jnp.exp(-1.0))))
    assert jnp.allclose(got.ustrip("Myr"), want, atol=1e-2)


def test_sample_accepts_a_custom_field():
    """A `CustomMassRateField` holding a Python function can be sampled."""
    field = gdc.CustomMassRateField(lambda _t, M, _args: -0.1 * M, units="galactic")
    t0, t1 = u.Q(0.0, "Myr"), u.Q(10.0, "Myr")
    m_of_t = gdc.MassSolver().solve(
        field, u.Q(1e4, "Msun"), t0, t1, dense=True, vectorize_interpolation=True
    )
    sampler = gdc.ReleaseTimeSampler(dm_dt=field, m_of_t=m_of_t, units="galactic")

    key = jr.PRNGKey(0)
    got = sampler.sample(key, t0, t1, n_stars=5, mass_params={})

    U = jr.uniform(key, shape=(5,))
    want = jnp.sort(-10.0 * jnp.log1p(-U * (1 - jnp.exp(-1.0))))
    assert jnp.allclose(got.ustrip("Myr"), want, atol=1e-2)


def test_sample_accepts_a_numpy_int_n_stars():
    """A numpy or whole-float ``n_stars`` sets a static shape; 5.5 is rejected."""
    field = gdc.CustomMassRateField(lambda _t, M, _args: -0.1 * M, units="galactic")
    t0, t1 = u.Q(0.0, "Myr"), u.Q(10.0, "Myr")
    m_of_t = gdc.MassSolver().solve(
        field, u.Q(1e4, "Msun"), t0, t1, dense=True, vectorize_interpolation=True
    )
    sampler = gdc.ReleaseTimeSampler(dm_dt=field, m_of_t=m_of_t, units="galactic")

    got = sampler.sample(jr.PRNGKey(0), t0, t1, n_stars=np.int64(5), mass_params={})
    assert got.shape == (5,)

    got = sampler.sample(jr.PRNGKey(0), t0, t1, n_stars=5.0, mass_params={})
    assert got.shape == (5,)

    with pytest.raises(TypeError):
        sampler.sample(jr.PRNGKey(0), t0, t1, n_stars=5.5, mass_params={})
