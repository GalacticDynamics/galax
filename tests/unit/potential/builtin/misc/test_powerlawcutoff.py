from typing import Any

import pytest
from plum import convert

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.typing as gt
from ...io.test_gala import parametrize_test_method_gala
from ...param.test_field import ParameterFieldMixin
from ...test_core import AbstractPotential_Test
from ..test_common import ParameterMTotMixin
from galax._interop.optional_deps import GSL_ENABLED, OptDeps
from galax.potential import AbstractBasePotential, PowerLawCutoffPotential


class AlphaParameterMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_alpha(self) -> u.Quantity["dimensionless"]:
        return u.Quantity(0.9, "")

    # =====================================================

    def test_alpha_constant(self, pot_cls, fields):
        """Test the `alpha` parameter."""
        fields["alpha"] = u.Quantity(1.0, "")
        pot = pot_cls(**fields)
        assert pot.alpha(t=u.Quantity(0, "Myr")) == u.Quantity(1.0, "")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_alpha_userfunc(self, pot_cls, fields):
        """Test the `alpha` parameter."""
        fields["alpha"] = lambda t: t * 1.2
        pot = pot_cls(**fields)
        assert pot.alpha(t=u.Quantity(0, "Myr")) == 2


class RCParameterMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_r_c(self) -> u.Quantity["length"]:
        return u.Quantity(1.0, "kpc")

    # =====================================================

    def test_r_c_constant(self, pot_cls, fields):
        """Test the `r_c` parameter."""
        fields["r_c"] = u.Quantity(1.0, "kpc")
        pot = pot_cls(**fields)
        assert pot.r_c(t=u.Quantity(0, "Myr")) == u.Quantity(1.0, "kpc")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_r_c_userfunc(self, pot_cls, fields):
        """Test the `r_c` parameter."""
        fields["r_c"] = lambda t: t * 1.2
        pot = pot_cls(**fields)
        assert pot.r_c(t=u.Quantity(0, "Myr")) == 2


class TestPowerLawCutoffPotential(
    AbstractPotential_Test,
    # Parameters
    ParameterMTotMixin,
    AlphaParameterMixin,
    RCParameterMixin,
):
    """Test the `galax.potential.PowerLawCutoffPotential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.PowerLawCutoffPotential]:
        return gp.PowerLawCutoffPotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m_tot: u.Quantity,
        field_alpha: u.Quantity,
        field_r_c: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m_tot": field_m_tot,
            "alpha": field_alpha,
            "r_c": field_r_c,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(self, pot: PowerLawCutoffPotential, x: gt.QSz3) -> None:
        expect = u.Quantity(6.26573365, unit="kpc2 / Myr2")
        assert jnp.isclose(
            pot.potential(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: PowerLawCutoffPotential, x: gt.QSz3) -> None:
        expect = u.Quantity([0.08587672, 0.17175344, 0.25763016], "kpc / Myr2")
        got = convert(pot.gradient(x, t=0), u.Quantity)
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: PowerLawCutoffPotential, x: gt.QSz3) -> None:
        expect = u.Quantity(41457.38551946, "solMass / kpc3")
        assert jnp.isclose(
            pot.density(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: PowerLawCutoffPotential, x: gt.QSz3) -> None:
        expect = u.Quantity(
            [
                [0.06747473, -0.03680397, -0.05520596],
                [-0.03680397, 0.01226877, -0.11041192],
                [-0.05520596, -0.11041192, -0.07974116],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.hessian(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: AbstractBasePotential, x: gt.QSz3) -> None:
        """Test the `AbstractBasePotential.tidal_tensor` method."""
        expect = u.Quantity(
            [
                [0.06747395, -0.03680397, -0.05520596],
                [-0.03680397, 0.01226799, -0.11041192],
                [-0.05520596, -0.11041192, -0.07974194],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Interoperability

    @pytest.mark.skipif(
        not OptDeps.GALA.installed or not GSL_ENABLED, reason="requires gala + GSL"
    )
    def test_galax_to_gala_to_galax_roundtrip(
        self, pot: gp.AbstractBasePotential, x: gt.QSz3
    ) -> None:
        super().test_galax_to_gala_to_galax_roundtrip(pot, x)

    @pytest.mark.skipif(
        not OptDeps.GALA.installed or not GSL_ENABLED, reason="requires gala + GSL"
    )
    @parametrize_test_method_gala
    def test_method_gala(
        self,
        pot: PowerLawCutoffPotential,
        method0: str,
        method1: str,
        x: gt.QSz3,
        atol: float,
    ) -> None:
        super().test_method_gala(pot, method0, method1, x, atol)
