"""Tests for the MW + LMC example field."""

import jax
import jax.numpy as jnp
import pytest

import unxt as u

import galax.dynamics as gd
import galax.potential as gp

MW_POT = gp.MilkyWayPotential(units="galactic")
LMC_POT = gp.NFWPotential(m=1e11, r_s=5, units="galactic")
LMC_X = jnp.array([-0.8, -41.5, -26.8])  # kpc
LMC_V = jnp.array([-0.057, -0.224, 0.190])  # kpc/Myr
DIST = jnp.linalg.vector_norm(LMC_X)


def lmc_acc(b_min, lmc_x=LMC_X):
    field = gd.examples.RigidMWandLMCField(
        mw_pot=MW_POT,
        lmc_pot=LMC_POT,
        sigma_func=lambda _: u.Q(130, "km/s").ustrip("kpc/Myr"),
        b_coulomb_min=u.Q(b_min, "kpc"),
    )
    coords = ((jnp.zeros(3), jnp.zeros(3)), (lmc_x, LMC_V))
    return field(0.0, coords)[1][1]


@pytest.mark.parametrize("b_min", [DIST, 2 * DIST, 1e9])
def test_no_friction_inside_b_coulomb_min(b_min):
    """At or inside b_coulomb_min the Coulomb log is 0, not NaN."""
    acc = lmc_acc(b_min)
    assert jnp.allclose(acc, MW_POT.acceleration(LMC_X, 0.0))

    jac = jax.jacobian(lmc_acc, argnums=1)(b_min, LMC_X)
    assert jnp.all(jnp.isfinite(jac))


def test_friction_outside_b_coulomb_min():
    """Outside b_coulomb_min the drag scales as sqrt(log(dist / b_min))."""
    gravity = MW_POT.acceleration(LMC_X, 0.0)
    drag1 = lmc_acc(1.0) - gravity
    drag2 = lmc_acc(5.0) - gravity
    expect = jnp.sqrt(jnp.log(DIST / 1.0) / jnp.log(DIST / 5.0))
    assert jnp.all(drag1 != 0)
    assert jnp.allclose(drag1 / drag2, expect)

    jac = jax.jacobian(lmc_acc, argnums=1)(1.0, LMC_X)
    assert jnp.all(jnp.isfinite(jac))


def test_quantity_inputs_match_arrays():
    """A Quantity time and coordinates give the same result as plain arrays."""
    field = gd.examples.RigidMWandLMCField(
        mw_pot=MW_POT,
        lmc_pot=LMC_POT,
        sigma_func=lambda _: u.Q(130, "km/s").ustrip("kpc/Myr"),
        b_coulomb_min=u.Q(1, "kpc"),
    )
    mw_x, mw_v = jnp.array([1.0, 2, 3]), jnp.array([0.01, 0.02, 0.03])
    coords = ((mw_x, mw_v), (LMC_X, LMC_V))
    qcoords = (
        (u.Q(mw_x, "kpc"), u.Q(mw_v, "kpc/Myr").uconvert("km/s")),
        (u.Q(LMC_X, "kpc"), u.Q(LMC_V, "kpc/Myr").uconvert("km/s")),
    )
    expect = field(100.0, coords)
    got = field(u.Q(0.1, "Gyr"), qcoords)
    for e, g in zip(jax.tree.leaves(expect), jax.tree.leaves(got), strict=True):
        assert jnp.allclose(e, g)
