import jax
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
from galax.potential._src.builtin.nfw.hyp2f1 import Bz_from_hyp2f1

check_funcs = ["potential", "gradient", "density", "hessian"]

# Settings of (alpha, beta, gamma) and correspondence to known models
abg_pot_finite = [
    ((1, 4, 1), gp.HernquistPotential),
    ((1, 4, 2), gp.JaffePotential),
    ((1 / 2, 5, 0), gp.PlummerPotential),
]
# beta <= 3 => infinite total mass, so these are compared via `gp.ZhaoPotential`
# directly (not `from_m_tot`) against the NFW family, which parametrizes its
# analogous "characteristic mass" m = 4 pi rho0 r_s^3 rather than Zhao's mass
# enclosed within `r_s`. `other_kw` supplies any extra constructor kwargs (e.g.
# gNFW's `gamma`) needed to match the `(alpha, beta, gamma)` setting.
abg_pot_infinite = [
    ((1, 3, 1), gp.NFWPotential, {}),
    ((1, 3, 0.5), gp.gNFWPotential, {"gamma": 0.5}),
]


@pytest.fixture
def xyz():
    test_r = jnp.geomspace(1e-3, 1e2, 128)
    rand_uvecs = jax.random.normal(jax.random.key(42), shape=(test_r.size, 3))
    rand_uvecs = rand_uvecs / jnp.linalg.norm(rand_uvecs, axis=-1, keepdims=True)
    return u.Quantity(test_r[:, None] * rand_uvecs, "kpc")


@pytest.mark.parametrize("func_name", check_funcs)
@pytest.mark.parametrize(("abg", "OtherPotential"), abg_pot_finite)
def test_zhao_against_finite_mass_correspondences(func_name, abg, OtherPotential, xyz):
    m_tot = u.Quantity(1.3e11, "Msun")
    zhao = gp.ZhaoPotential.from_m_tot(
        m_tot=m_tot,
        r_s=u.Quantity(8.1, "kpc"),
        alpha=abg[0],
        beta=abg[1],
        gamma=abg[2],
        units="galactic",
    )
    other = OtherPotential(m_tot=m_tot, r_s=zhao.parameters["r_s"], units="galactic")

    zhao_result = getattr(zhao, func_name)(xyz, u.Quantity(0.0, "Myr"))
    other_result = getattr(other, func_name)(xyz, u.Quantity(0.0, "Myr"))

    assert jnp.allclose(
        zhao_result, other_result, rtol=1e-8, atol=u.Quantity(1e-6, zhao_result.unit)
    )


@pytest.mark.parametrize("func_name", check_funcs)
@pytest.mark.parametrize(("abg", "OtherPotential", "other_kw"), abg_pot_infinite)
def test_zhao_against_infinite_mass_correspondences(
    func_name, abg, OtherPotential, other_kw, xyz
):
    """Compare against the NFW family for beta <= 3 (infinite total mass).

    There is no `from_m_tot` constructor here (it requires beta > 3), so instead
    convert between Zhao's "mass enclosed within r_s" convention and the NFW
    family's "characteristic mass" convention, m = 4 pi rho0 r_s^3.
    """
    alpha, beta, gamma = abg
    r_s = u.Quantity(8.1, "kpc")
    m_char = u.Quantity(5e11, "Msun")

    m_enclosed = m_char * Bz_from_hyp2f1(3.0 - gamma, 0.0, 0.5)
    zhao = gp.ZhaoPotential(
        m=m_enclosed, r_s=r_s, alpha=alpha, beta=beta, gamma=gamma, units="galactic"
    )
    other = OtherPotential(m=m_char, r_s=r_s, units="galactic", **other_kw)

    zhao_result = getattr(zhao, func_name)(xyz, u.Quantity(0.0, "Myr"))
    other_result = getattr(other, func_name)(xyz, u.Quantity(0.0, "Myr"))

    assert jnp.allclose(
        zhao_result, other_result, rtol=1e-8, atol=u.Quantity(1e-6, zhao_result.unit)
    )
