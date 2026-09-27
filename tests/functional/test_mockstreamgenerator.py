"""Test :mod:`galax.dynamics.mockstream.mockstreamgenerator`."""

from jaxtyping import PRNGKeyArray
from typing import Any

import jax
import jax.random as jr
import jax.tree_util as jtu
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.dynamics as gd
import galax.dynamics.custom_types as gt
import galax.potential as gp

usys = u.unitsystem("kpc", "Myr", "Msun", "radian")
df = gd.FardalStreamDF()


@jax.jit
def compute_loss(
    params: dict[str, Any],
    rng: PRNGKeyArray,
    ts: gt.QuSzTime,
    w0: gt.Sz6,
    M_sat: gt.FloatQuSz0,
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
    ts: gt.QuSzTime,
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
            "m_tot": u.Q(5.0e10, "Msun"),
            "a": u.Q(3.0, "kpc"),
            "b": u.Q(0.25, "kpc"),
        },
        "halo": {
            "m": u.Q(1.0e12, "Msun"),
            "r_s": u.Q(15.0, "kpc"),
        },
    }

    ts = u.Q(jnp.linspace(0.0, 4.0, 10_000), "Gyr")
    w0 = gc.PhaseSpacePosition(
        q=u.Q([30.0, 10, 20], "kpc"), p=u.Q([10.0, -150, -20], "km / s")
    )
    M_sat = u.Q(1.0e4, "Msun")

    # Compute the first derivative
    rng = jr.key(12)
    first_deriv = compute_derivative(params, rng, ts, w0, M_sat)

    # Test
    return jnp.asarray(jtu.tree_flatten(first_deriv)[0])


# NOTE: `rtol` is deliberately far looser than `pytest-arraydiff`'s 1e-7
# default. These values come from `jax.jacfwd` twice through an adaptive Dopri8
# (`PIDController(rtol=atol=1e-7)`), whose step sequence is chosen by the primal
# and is not itself differentiated, so the second derivative inherits an error
# much larger than the integrator's own tolerance. Measured on this pipeline:
# tightening the solver to 1e-8 or 1e-9 shifts these values by ~2e-3 relative,
# i.e. the quantity is only converged to ~3 significant figures at the default
# solver tolerance. A platform that reorders floating-point ops enough to pick a
# different step sequence lands within that same ~2e-3 band (macOS arm64 differed
# from the old reference by 1e-3), so anything tighter than 1e-2 fails on a
# perfectly healthy build. `atol` covers the entries that are zero up to
# round-off; without it they would have to match to 1% of ~1e-16.
@pytest.mark.array_compare(
    file_format="text", reference_dir="reference", rtol=1e-2, atol=1e-12
)
def test_second_deriv() -> None:
    # Inputs
    params = {
        "disk": {
            "m_tot": u.Q(5.0e10, "Msun"),
            "a": u.Q(3.0, "kpc"),
            "b": u.Q(0.25, "kpc"),
        },
        "halo": {
            "m": u.Q(1.0e12, "Msun"),
            "r_s": u.Q(15.0, "kpc"),
        },
    }

    # Unlike `test_first_deriv` these values are not identically zero, so the
    # stream length is baked into the reference. The loss is a sum over release
    # times, so they are linear in `len(ts)` to ~2e-3: 10_000 buys no coverage
    # over 100, only cost (~700s vs ~25s, plus the memory blowup that made the
    # first-derivative test shrink). Regenerate `reference/test_second_deriv.txt`
    # if this count changes.
    ts = u.Q(jnp.linspace(0.0, 4.0, 100), "Gyr")
    w0 = gc.PhaseSpacePosition(
        q=u.Q([30.0, 10, 20], "kpc"), p=u.Q([10.0, -150, -20], "km / s")
    )
    M_sat = u.Q(1.0e4, "Msun")

    # Compute the second derivative
    rng = jr.key(12)
    second_deriv = jax.jacfwd(jax.jacfwd(compute_loss, argnums=0))(
        params, rng, ts, w0, M_sat
    )

    # Test
    return jnp.asarray(jtu.tree_flatten(second_deriv)[0])
