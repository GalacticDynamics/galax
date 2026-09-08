"""Benchmark ``ZhaoPotential`` against a closed-form potential (``Plummer``).

The Zhao (1996) double power-law potential computes its mass normalization and
enclosed-mass terms from `hyp2f1`/`betainc`, which are much more expensive to
evaluate -- and (for `potential`, which differentiates through them) especially
to differentiate -- than a closed-form potential like Plummer's. See
https://github.com/GalacticDynamics/galax/pull/761#issuecomment-3185820797 for
wall-clock measurements motivating this suite.

`ZhaoPotential.gradient` is overridden with an analytic (shell-theorem) formula
that only evaluates the enclosed mass forward, rather than differentiating
through `potential`, so it should track `Plummer.gradient` far more closely
than `potential`-vs-`potential` does.

`potential`/`gradient` are already jitted internally (`AbstractSinglePotential`
wraps `_potential`/`_gradient` in `jax.jit`), so calling them directly -- after
one warm-up call per input shape to prime the compilation cache -- benchmarks
steady-state execution rather than re-tracing/compiling.
"""

import jax
import pytest

import unxt as u

import galax.potential as gp

N_POINTS = [1, 1_000, 100_000]

_xyz_all = jax.random.uniform(
    jax.random.key(0), (max(N_POINTS), 3), minval=-50.0, maxval=50.0
)
_t = u.Quantity(0.0, "Myr")

potentials = {
    "Zhao": gp.ZhaoPotential(
        m=u.Quantity(1e12, "Msun"),
        r_s=u.Quantity(10.0, "kpc"),
        alpha=1.0,
        beta=4.0,
        gamma=1.0,
        units="galactic",
    ),
    "Plummer": gp.PlummerPotential(
        m_tot=u.Quantity(1e12, "Msun"), r_s=u.Quantity(10.0, "kpc"), units="galactic"
    ),
}


@pytest.mark.parametrize("name", potentials)
@pytest.mark.parametrize("n", N_POINTS, ids=lambda n: f"n={n}")
@pytest.mark.benchmark(group="potential.zhao.potential")
def test_potential_eval(name, n):
    """Wall-clock cost of evaluating the potential value at `n` points."""
    pot = potentials[name]
    xyz = u.Quantity(_xyz_all[:n], "kpc")
    pot.potential(xyz, _t)  # warm up / compile, outside the timed region
    jax.block_until_ready(pot.potential(xyz, _t))


@pytest.mark.parametrize("name", potentials)
@pytest.mark.parametrize("n", N_POINTS, ids=lambda n: f"n={n}")
@pytest.mark.benchmark(group="potential.zhao.gradient")
def test_gradient_eval(name, n):
    """Wall-clock cost of evaluating the force (gradient) at `n` points."""
    pot = potentials[name]
    xyz = u.Quantity(_xyz_all[:n], "kpc")
    pot.gradient(xyz, _t)  # warm up / compile, outside the timed region
    jax.block_until_ready(pot.gradient(xyz, _t))
