from typing import Any, ClassVar

import jax
import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.potential.custom_types as gt
from ..param.test_field import ParameterFieldMixin
from ..test_core import AbstractSinglePotential_Test
from .test_common import ParameterMMixin, ParameterRSMixin


class AlphaParameterMixin(ParameterFieldMixin):
    """Test the alpha parameter."""

    @pytest.fixture(scope="class")
    def field_alpha(self) -> u.Quantity["dimensionless"]:
        return u.Quantity(0.9, "")

    # =====================================================

    def test_alpha_constant(self, pot_cls, fields):
        """Test the `alpha` parameter."""
        fields["alpha"] = u.Quantity(1.0, "")
        pot = pot_cls(**fields)
        assert pot.alpha(t=u.Quantity(0, "Myr")) == u.Quantity(1.0, "")

    def test_alpha_userfunc(self, pot_cls, fields):
        """Test the `alpha` parameter."""

        def cos_alpha(t: u.Quantity["time"]) -> u.Quantity[""]:
            return u.Quantity(0.5 * jnp.cos(t.ustrip("Myr")) ** 2 + 0.5, "")

        fields["alpha"] = cos_alpha
        pot = pot_cls(**fields)
        assert pot.alpha(t=u.Quantity(0, "Myr")) == u.Quantity(1.0, "")


class BetaParameterMixin(ParameterFieldMixin):
    """Test the beta parameter."""

    @pytest.fixture(scope="class")
    def field_beta(self) -> u.Quantity["dimensionless"]:
        return u.Quantity(4.31, "")

    # =====================================================

    def test_beta_constant(self, pot_cls, fields):
        """Test the `beta` parameter."""
        fields["beta"] = u.Quantity(3.5, "")
        pot = pot_cls(**fields)
        assert pot.beta(t=u.Quantity(0, "Myr")) == u.Quantity(3.5, "")

    def test_beta_userfunc(self, pot_cls, fields):
        """Test the `beta` parameter."""

        def cos_beta(t: u.Quantity["time"]) -> u.Quantity[""]:
            return u.Quantity(jnp.cos(t.ustrip("Myr")) + 4.2, "")

        fields["beta"] = cos_beta
        pot = pot_cls(**fields)
        assert pot.beta(t=u.Quantity(0, "Myr")) == u.Quantity(5.2, "")


class GammaParameterMixin(ParameterFieldMixin):
    """Test the gamma parameter."""

    @pytest.fixture(scope="class")
    def field_gamma(self) -> u.Quantity["dimensionless"]:
        return u.Quantity(1.2, "")

    # =====================================================

    def test_gamma_constant(self, pot_cls, fields):
        """Test the `gamma` parameter."""
        fields["gamma"] = u.Quantity(1.5, "")
        pot = pot_cls(**fields)
        assert pot.gamma(t=u.Quantity(0, "Myr")) == u.Quantity(1.5, "")

    def test_gamma_userfunc(self, pot_cls, fields):
        """Test the `gamma` parameter."""

        def cos_gamma(t: u.Quantity["time"]) -> u.Quantity[""]:
            return u.Quantity(0.5 * jnp.cos(t.ustrip("Myr")) + 1.5, "")

        fields["gamma"] = cos_gamma
        pot = pot_cls(**fields)
        assert pot.gamma(t=u.Quantity(0, "Myr")) == u.Quantity(2.0, "")


class TestZhaoPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMMixin,
    ParameterRSMixin,
    AlphaParameterMixin,
    BetaParameterMixin,
    GammaParameterMixin,
):
    """Test the `galax.potential.ZhaoPotential` class."""

    HAS_GALA_COUNTERPART: ClassVar[bool] = False

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.ZhaoPotential]:
        return gp.ZhaoPotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m: u.Quantity,
        field_r_s: u.Quantity,
        field_alpha: u.Quantity,
        field_beta: u.Quantity,
        field_gamma: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m": field_m,
            "r_s": field_r_s,
            "alpha": field_alpha,
            "beta": field_beta,
            "gamma": field_gamma,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(self, pot: gp.ZhaoPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(-2.83144346, unit="kpc2 / Myr2")
        assert jnp.isclose(
            pot.potential(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: gp.ZhaoPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity([0.1758548, 0.35170961, 0.52756441], "kpc / Myr2")
        got = pot.gradient(x, t=0)
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: gp.ZhaoPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(8.93719599e08, "solMass / kpc3")
        assert jnp.isclose(
            pot.density(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: gp.ZhaoPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(
            [
                [0.14178033, -0.06814894, -0.10222341],
                [-0.06814894, 0.03955692, -0.20444682],
                [-0.10222341, -0.20444682, -0.13081543],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.hessian(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        expect = u.Quantity(
            [
                [0.12493972, -0.06814894, -0.10222341],
                [-0.06814894, 0.02271631, -0.20444682],
                [-0.10222341, -0.20444682, -0.14765604],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )


# ===================================================================
# Analytic derivatives
#
# `gradient`, `hessian` and `laplacian` are written analytically (Zhao Eqs. 15
# and 1) rather than by autodiff of `potential`, so check them against autodiff
# of the potential, over the special cases of the model's Table 1 list.

ABG = [
    (1.0, 4.0, 1.0),  # Hernquist
    (1.0, 4.0, 2.0),  # Jaffe (p0 = 0)
    (0.5, 5.0, 0.0),  # Plummer
    (1.0, 3.0, 1.0),  # NFW (q0 = 0, infinite mass)
    (0.9, 4.31, 1.2),  # generic
]
RADII = [0.05, 0.5, 1.0, 5.0, 50.0]


def _pot(abg: tuple[float, float, float]) -> gp.ZhaoPotential:
    alpha, beta, gamma = abg
    return gp.ZhaoPotential(
        m=u.Quantity(1e12, "Msun"),
        r_s=u.Quantity(8.0, "kpc"),
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        units="galactic",
    )


def _xyz_at(r: float) -> jnp.ndarray:
    xyz = jnp.asarray([0.3, -0.5, 0.81])
    return xyz / jnp.linalg.norm(xyz) * r


@pytest.mark.parametrize("abg", ABG)
@pytest.mark.parametrize("r", RADII)
def test_analytic_gradient_matches_autodiff(abg, r: float) -> None:
    """The shell-theorem gradient must match `jax.grad` of the potential."""
    pot, xyz = _pot(abg), _xyz_at(r)
    got = pot.gradient(xyz, t=0)
    expect = jax.grad(lambda q: pot._potential(q, 0.0))(xyz)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expect), rtol=1e-8)


@pytest.mark.parametrize("abg", ABG)
@pytest.mark.parametrize("r", RADII)
def test_analytic_hessian_matches_autodiff(abg, r: float) -> None:
    """The analytic hessian must match `jax.hessian` of the potential."""
    pot, xyz = _pot(abg), _xyz_at(r)
    got = pot.hessian(xyz, t=0)
    expect = jax.hessian(lambda q: pot._potential(q, 0.0))(xyz)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expect), rtol=1e-6)


@pytest.mark.parametrize("abg", ABG)
@pytest.mark.parametrize("r", RADII)
def test_laplacian_is_poisson(abg, r: float) -> None:
    """Poisson's equation: the laplacian must be 4 pi G rho (Eq. 1)."""
    pot, xyz = _pot(abg), _xyz_at(r)
    lap = pot.laplacian(xyz, t=0)
    expect = 4 * jnp.pi * pot.constants["G"].value * pot.density(xyz, t=0)
    np.testing.assert_allclose(np.asarray(lap), np.asarray(expect), rtol=1e-10)
