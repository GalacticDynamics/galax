"""Test :mod:`galax.dynamics.mockstream.mockstreamgenerator`."""

from typing import Any

import astropy.units as u
import jax
import pytest
import quax.examples.prng as jr

import quaxed.array_api as xp
from unxt import Quantity, UnitSystem

from galax.dynamics import FardalStreamDF, MockStreamGenerator
from galax.potential import MilkyWayPotential
from galax.typing import FloatQScalar, FloatScalar, QVecTime, Vec6

usys = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian)
df = FardalStreamDF()


@jax.jit
def compute_loss(
    params: dict[str, Any], rng: jr.PRNG, ts: QVecTime, w0: Vec6, M_sat: FloatQScalar
) -> FloatScalar:
    # Generate mock stream
    pot = MilkyWayPotential(**params, units=usys)
    mockgen = MockStreamGenerator(df, pot)
    stream, _ = mockgen.run(rng, ts, w0, M_sat)
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
    params: dict[str, Any], rng: jr.PRNG, ts: QVecTime, w0: Vec6, M_sat: FloatScalar
) -> dict[str, Any]:
    return jax.jacfwd(compute_loss, argnums=0)(params, rng, ts, w0, M_sat)


@pytest.mark.array_compare(file_format="text", reference_dir="reference")
def test_first_deriv() -> None:
    """Test the first derivative of the mockstream."""
    # Inputs
    params = {
        "disk": {
            "m_tot": Quantity(5.0e10, "Msun"),
            "a": Quantity(3.0, "kpc"),
            "b": Quantity(0.25, "kpc"),
        },
        "halo": {"m": Quantity(1.0e12, "Msun"), "r_s": Quantity(15.0, "kpc")},
    }

    ts = Quantity(xp.linspace(0.0, 4_000, 10_000, dtype=float), "Myr")

    q0 = ((30, 10, 20) * u.Unit("kpc")).decompose(usys).value
    p0 = ((10, -150, -20) * u.Unit("km / s")).decompose(usys).value
    w0 = xp.asarray([*q0, *p0], dtype=float)

    M_sat = Quantity(1.0e4, "Msun")

    # Compute the first derivative
    rng = jr.ThreeFry(12)
    first_deriv = compute_derivative(params, rng, ts, w0, M_sat)

    # Test
    return xp.asarray(jax.tree_util.tree_flatten(first_deriv)[0])


@pytest.mark.slow()
@pytest.mark.array_compare(file_format="text", reference_dir="reference")
def test_second_deriv() -> None:
    # Inputs
    params = {
        "disk": {
            "m_tot": Quantity(5.0e10, "Msun"),
            "a": Quantity(3.0, "kpc"),
            "b": Quantity(0.25, "kpc"),
        },
        "halo": {"m": Quantity(1.0e12, "Msun"), "r_s": Quantity(15.0, "kpc")},
    }
    ts = Quantity(xp.linspace(0.0, 4_000, 10_000, dtype=float), "Myr")

    q0 = ((30, 10, 20) * u.Unit("kpc")).decompose(usys).value
    p0 = ((10, -150, -20) * u.Unit("km / s")).decompose(usys).value
    w0 = xp.asarray([*q0, *p0], dtype=float)

    M_sat = Quantity(1.0e4, "Msun")

    # Compute the second derivative
    rng = jr.ThreeFry(12)
    second_deriv = jax.jacfwd(jax.jacfwd(compute_loss, argnums=0))(
        params, rng, ts, w0, M_sat
    )

    # Test
    return xp.asarray(jax.tree_util.tree_flatten(second_deriv)[0])
