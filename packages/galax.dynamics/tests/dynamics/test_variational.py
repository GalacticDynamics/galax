"""Tests for the variational equations of orbit fields."""

import diffrax as dfx
import jax
import jax.numpy as jnp
import pytest

import galax.dynamics as gd
import galax.potential as gp
from galax.dynamics._src.orbit.variational import tangent_terms

POT = gp.HernquistPotential(m_tot=1e12, r_s=5, units="galactic")
FIELD = gd.fields.HamiltonianField(POT)
X = jnp.array([8.0, 0, 1])
V = jnp.array([0.0, 0.2, 0.02])
EYE = jnp.eye(6)


# ===================================================================
# tangent_terms


def test_tangent_odeterm_matches_jacobian():
    """The augmented ODE term returns the field and its Jacobian times dW."""
    term = FIELD.terms(dfx.Dopri8())
    aug = tangent_terms(term)
    (xd, dxd), (vd, dvd) = aug.vf(0.0, ((X, EYE[:3]), (V, EYE[3:])), None)

    def rhs(w):
        return jnp.concat(term.vf(0.0, (w[:3], w[3:]), None))

    w = jnp.concat([X, V])
    jac = jax.jacfwd(rhs)(w)
    assert jnp.allclose(jnp.concat([xd, vd]), rhs(w))
    assert jnp.allclose(jnp.concat([dxd, dvd]), jac, rtol=1e-12)


@pytest.mark.parametrize(
    "solver", [gd.experimental.Leapfrog(), dfx.SemiImplicitEuler()]
)
def test_tangent_split_terms_match_jacobian(solver):
    """The augmented drift and kick terms return their Jacobians times dW."""
    f, g = FIELD.terms(solver)
    f_aug, g_aug = tangent_terms((f, g))

    vd, dvd = f_aug.vf(0.0, (V, EYE[3:]), None)
    assert jnp.array_equal(vd, f.vf(0.0, V, None))
    assert jnp.array_equal(dvd, EYE[3:])

    ad, dad = g_aug.vf(0.0, (X, EYE[:3]), None)
    jac = jax.jacfwd(lambda q: g.vf(0.0, q, None))(X)
    assert jnp.allclose(ad, g.vf(0.0, X, None))
    assert jnp.allclose(dad, jac @ EYE[:3], rtol=1e-12)
