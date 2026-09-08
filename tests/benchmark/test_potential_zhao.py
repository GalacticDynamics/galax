"""Benchmark `ZhaoPotential` against a closed-form potential (`Plummer`).

The Zhao (1996) model's potential, enclosed mass and normalization are all
incomplete beta functions (Eqs. 7, 15, 44), so it can never be quite as cheap
as a potential like Plummer's that is a couple of arithmetic operations. What
it should *not* be is asymptotically worse: see
https://github.com/GalacticDynamics/galax/pull/761#issuecomment-3185820797 for
the measurements that motivated this suite.

`gradient`, `laplacian` and `hessian` are written analytically (from Eq. 15
and Poisson's equation) rather than by autodiff of `potential`, so they should
cost no more than `potential` itself -- `laplacian` in particular is just
`4 pi G rho` and so should track Plummer.

`ZhaoInterp` is `InterpolatedZhaoPotential`: the same model with Eq. 43 fitted
rather than summed, which should land near Plummer for every method.

Everything is jitted internally (`AbstractSinglePotential` wraps `_potential`
and friends in `jax.jit`), so each case is warmed up at import, below, and the
timed body is a single call.
"""

import jax
import pytest

import unxt as u

import galax.potential as gp

N_POINTS = [1, 1_000, 100_000]
METHODS = ["potential", "gradient", "laplacian", "hessian"]

_xyz_all = jax.random.uniform(
    jax.random.key(0), (max(N_POINTS), 3), minval=-50.0, maxval=50.0
)
_t = u.Quantity(0.0, "Myr")

ZHAO_KW = {
    "m": u.Quantity(1e12, "Msun"),
    "r_s": u.Quantity(10.0, "kpc"),
    "alpha": 1.0,
    "beta": 4.0,
    "gamma": 1.0,
    "units": "galactic",
}

potentials = {
    "Zhao": gp.ZhaoPotential(
        m=u.Quantity(1e12, "Msun"),
        r_s=u.Quantity(10.0, "kpc"),
        alpha=1.0,
        beta=4.0,
        gamma=1.0,
        units="galactic",
    ),
    "ZhaoInterp": gp.InterpolatedZhaoPotential(**ZHAO_KW),
    "Plummer": gp.PlummerPotential(
        m_tot=u.Quantity(1e12, "Msun"), r_s=u.Quantity(10.0, "kpc"), units="galactic"
    ),
}

xyzs = {n: u.Quantity(_xyz_all[:n], "kpc") for n in N_POINTS}

# Compile every case once, so the timed bodies below measure execution only.
for _pot in potentials.values():
    for _method in METHODS:
        for _xyz in xyzs.values():
            jax.block_until_ready(getattr(_pot, _method)(_xyz, _t))


@pytest.mark.parametrize("name", potentials)
@pytest.mark.parametrize("n", N_POINTS, ids=lambda n: f"n={n}")
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.benchmark(group="potential.zhao")
def test_eval(method: str, n: int, name: str) -> None:
    """Wall-clock cost of one `method` evaluation at `n` points."""
    jax.block_until_ready(getattr(potentials[name], method)(xyzs[n], _t))
