"""Test :mod:`galax.dynamics.mockstream.mockstreamgenerator`."""

from typing import Any

import astropy.units as u
import jax
import pytest

import array_api_jax_compat as xp

from galax.dynamics import FardalStreamDF, MockStreamGenerator
from galax.potential import MilkyWayPotential
from galax.typing import FloatScalar, Vec6, VecTime
from galax.units import UnitSystem

usys = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian)
df = FardalStreamDF()
seed_num = 12


@jax.jit
def compute_loss(
    params: dict[str, Any], ts: VecTime, w0: Vec6, M_sat: FloatScalar
) -> FloatScalar:
    # Generate mock stream
    pot = MilkyWayPotential(**params, units=usys)
    mockgen = MockStreamGenerator(df, pot)
    stream, _ = mockgen.run(ts, w0, M_sat, seed_num=seed_num)
    trail_arm, lead_arm = stream[::2], stream[1::2]
    # Generate "observed" stream from mock
    lead_arm_obs = jax.lax.stop_gradient(lead_arm)
    trail_arm_obs = jax.lax.stop_gradient(trail_arm)
    # Compute loss
    return -xp.sum(
        (lead_arm.w(units=usys) - lead_arm_obs.w(units=usys)) ** 2
        + (trail_arm.w(units=usys) - trail_arm_obs.w(units=usys)) ** 2
    )


@jax.jit
def compute_derivative(
    params: dict[str, Any], ts: VecTime, w0: Vec6, M_sat: FloatScalar
) -> dict[str, Any]:
    return jax.jacfwd(compute_loss, argnums=0)(params, ts, w0, M_sat)


@pytest.mark.array_compare(file_format="text", reference_dir="reference")
def test_first_deriv() -> None:
    """Test the first derivative of the mockstream."""
    # Inputs
    params = {
        "disk": {"m": 5.0e10 * u.Msun, "a": 3.0, "b": 0.25},
        "halo": {"m": 1.0e12 * u.Msun, "r_s": 15.0},
    }

    ts = xp.linspace(0.0, 4_000, 10_000, dtype=float)  # [Myr]

    q0 = ((30, 10, 20) * u.Unit("kpc")).decompose(usys).value
    p0 = ((10, -150, -20) * u.Unit("km / s")).decompose(usys).value
    w0 = xp.asarray([*q0, *p0], dtype=float)

    M_sat = 1.0e4 * u.Msun

    # Compute the first derivative
    first_deriv = compute_derivative(params, ts, w0, M_sat)

    # Test
    return xp.asarray(jax.tree_util.tree_flatten(first_deriv)[0])


@pytest.mark.slow()
@pytest.mark.array_compare(file_format="text", reference_dir="reference")
def test_second_deriv() -> None:
    # Inputs
    params = {
        "disk": {"m": 5.0e10 * u.Msun, "a": 3.0, "b": 0.25},
        "halo": {"m": 1.0e12 * u.Msun, "r_s": 15.0},
    }
    ts = xp.linspace(0.0, 4_000, 10_000, dtype=float)  # [Myr]

    q0 = ((30, 10, 20) * u.Unit("kpc")).decompose(usys).value
    p0 = ((10, -150, -20) * u.Unit("km / s")).decompose(usys).value
    w0 = xp.asarray([*q0, *p0], dtype=float)

    M_sat = 1.0e4 * u.Msun

    # Compute the second derivative
    second_deriv = jax.jacfwd(jax.jacfwd(compute_loss, argnums=0))(
        params, ts, w0, M_sat
    )

    # Test
    return xp.asarray(jax.tree_util.tree_flatten(second_deriv)[0])
