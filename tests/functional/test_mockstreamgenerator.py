"""Test :mod:`galax.dynamics.mockstream.mockstreamgenerator`."""

from typing import Any

import jax
import jax.random as jr
import jax.tree_util as jtu
import pytest
from jaxtyping import PRNGKeyArray

import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp
import galax.typing as gt

usys = u.unitsystem("kpc", "Myr", "Msun", "radian")
df = gd.FardalStreamDF()


@jax.jit
def compute_loss(
    params: dict[str, Any],
    rng: PRNGKeyArray,
    ts: gt.QSzTime,
    w0: gt.Sz6,
    M_sat: gt.FloatQSz0,
) -> gt.FloatSz0:
    # Generate mock stream
    pot = gp.MilkyWayPotential(**params, units=usys)
    mockgen = gd.MockStreamGenerator(df, pot)
    stream, _ = mockgen.run(rng, ts, w0, M_sat)
    trail_arm, lead_arm = stream["lead"], stream["trail"]
    # Generate "observed" stream from mock
    lead_arm_obs = jax.lax.stop_gradient(lead_arm)
    trail_arm_obs = jax.lax.stop_gradient(trail_arm)
    # Compute loss
    return -jnp.sum(
        (lead_arm.w(units=usys) - lead_arm_obs.w(units=usys)) ** 2
        + (trail_arm.w(units=usys) - trail_arm_obs.w(units=usys)) ** 2
    )


@jax.jit
def compute_derivative(
    params: dict[str, Any],
    rng: PRNGKeyArray,
    ts: gt.QSzTime,
    w0: gc.PhaseSpacePosition,
    M_sat: gt.FloatSz0,
) -> dict[str, Any]:
    return jax.jacfwd(compute_loss, argnums=0)(params, rng, ts, w0, M_sat)


@pytest.mark.array_compare(file_format="text", reference_dir="reference")
def test_first_deriv() -> None:
    """Test the first derivative of the mockstream."""
    # Inputs
    params = {
        "disk": {
            "m_tot": u.Quantity(5.0e10, "Msun"),
            "a": u.Quantity(3.0, "kpc"),
            "b": u.Quantity(0.25, "kpc"),
        },
        "halo": {
            "m": u.Quantity(1.0e12, "Msun"),
            "r_s": u.Quantity(15.0, "kpc"),
        },
    }

    ts = u.Quantity(jnp.linspace(0.0, 4.0, 10_000), "Gyr")
    w0 = gc.PhaseSpacePosition(
        q=u.Quantity([30.0, 10, 20], "kpc"),
        p=u.Quantity([10.0, -150, -20], "km / s"),
        t=None,
    )
    M_sat = u.Quantity(1.0e4, "Msun")

    # Compute the first derivative
    rng = jr.key(12)
    first_deriv = compute_derivative(params, rng, ts, w0, M_sat)

    # Test
    return jnp.asarray(jtu.tree_flatten(first_deriv)[0])


@pytest.mark.slow
@pytest.mark.array_compare(file_format="text", reference_dir="reference")
def test_second_deriv() -> None:
    # Inputs
    params = {
        "disk": {
            "m_tot": u.Quantity(5.0e10, "Msun"),
            "a": u.Quantity(3.0, "kpc"),
            "b": u.Quantity(0.25, "kpc"),
        },
        "halo": {
            "m": u.Quantity(1.0e12, "Msun"),
            "r_s": u.Quantity(15.0, "kpc"),
        },
    }

    ts = u.Quantity(jnp.linspace(0.0, 4.0, 10_000), "Gyr")
    w0 = gc.PhaseSpacePosition(
        q=u.Quantity([30.0, 10, 20], "kpc"),
        p=u.Quantity([10.0, -150, -20], "km / s"),
        t=None,
    )
    M_sat = u.Quantity(1.0e4, "Msun")

    # Compute the second derivative
    rng = jr.key(12)
    second_deriv = jax.jacfwd(jax.jacfwd(compute_loss, argnums=0))(
        params, rng, ts, w0, M_sat
    )

    # Test
    return jnp.asarray(jtu.tree_flatten(second_deriv)[0])
