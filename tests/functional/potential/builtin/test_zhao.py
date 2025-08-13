import jax
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
from galax.potential._src.builtin.zhao import ZhaoPotential

check_funcs = ["potential", "gradient", "density", "hessian"]

# settings of (alpha, beta, gamma) and correspondence to known models:
abg_pot = {
    (1, 4, 1): gp.HernquistPotential,
    (1, 4, 2): gp.JaffePotential,
    (1 / 2, 5, 0): gp.PlummerPotential,
    # (1, 3, 1): gp.NFWPotential,
}


@pytest.fixture
def xyz():
    test_r = jnp.geomspace(1e-3, 1e2, 128)
    rand_uvecs = jax.random.normal(jax.random.key(42), shape=(test_r.size, 3))
    rand_uvecs = rand_uvecs / jnp.linalg.norm(rand_uvecs, axis=-1, keepdims=True)
    return u.Quantity(test_r[:, None] * rand_uvecs, "kpc")


@pytest.mark.parametrize("func_name", check_funcs)
@pytest.mark.parametrize(("abg", "OtherPotential"), list(abg_pot.items()))
def test_zhao_against_correspondences(func_name, abg, OtherPotential, xyz):
    zhao = ZhaoPotential(
        m=u.Quantity(1.3e11, "Msun"),
        r_s=u.Quantity(8.1, "kpc"),
        alpha=abg[0],
        beta=abg[1],
        gamma=abg[2],
        units="galactic",
    )
    other = OtherPotential(
        m_tot=zhao.parameters["m"], r_s=zhao.parameters["r_s"], units="galactic"
    )

    zhao_result = getattr(zhao, func_name)(xyz, u.Quantity(0.0, "Myr"))
    other_result = getattr(other, func_name)(xyz, u.Quantity(0.0, "Myr"))

    assert jnp.allclose(
        zhao_result, other_result, rtol=1e-8, atol=u.Quantity(1e-6, zhao_result.unit)
    )
