from typing import Any, ClassVar

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
import galax.potential as gp
from ..param.test_field import ParameterFieldMixin
from ..test_core import AbstractSinglePotential_Test
from .test_common import ParameterMMixin, ParameterRSMixin
from galax.potential._src.builtin.zhao import ZhaoPotential


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
    def pot_cls(self) -> type[ZhaoPotential]:
        return ZhaoPotential

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

    def test_potential(self, pot: ZhaoPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(-1.07419698, unit="kpc2 / Myr2")
        print(pot.parameters)
        assert jnp.isclose(
            pot.potential(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: ZhaoPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity([0.06671604, 0.13343208, 0.20014812], "kpc / Myr2")
        got = pot.gradient(x, t=0)
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: ZhaoPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(3.39060592e08, "solMass / kpc3")
        assert jnp.isclose(
            pot.density(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: ZhaoPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(
            [
                [0.05378882, -0.02585444, -0.03878166],
                [-0.02585444, 0.01500716, -0.07756332],
                [-0.03878166, -0.07756332, -0.04962894],
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
                [0.04739981, -0.02585444, -0.03878166],
                [-0.02585444, 0.00861815, -0.07756332],
                [-0.03878166, -0.07756332, -0.05601795],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )
