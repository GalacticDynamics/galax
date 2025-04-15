from typing import Any, ClassVar
from typing_extensions import override

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
import galax.potential as gp
import galax.potential.params as gpp
from ..param.test_field import ParameterFieldMixin
from ..test_core import AbstractSinglePotential_Test
from .test_common import ParameterMMixin, ParameterRSMixin


class ParameterRTMixin(ParameterFieldMixin):
    """Test the mass parameter."""

    pot_cls: type[gp.AbstractSinglePotential]

    @pytest.fixture(scope="class")
    def field_r_t(self) -> u.Quantity["length"]:
        return u.Quantity(67.0, "kpc")

    # =====================================================

    def test_r_t_units(
        self, pot_cls: type[gp.AbstractSinglePotential], fields: dict[str, Any]
    ) -> None:
        """Test the mass parameter."""
        fields["r_t"] = 1.0 * u.unit(10 * u.unit("kpc"))
        fields["units"] = u.unitsystems.galactic
        pot = pot_cls(**fields)
        assert isinstance(pot.r_t, gpp.ConstantParameter)
        assert jnp.isclose(
            pot.r_t(0), u.Quantity(10, "kpc"), atol=u.Quantity(1e-8, "kpc")
        )

    def test_r_t_constant(
        self, pot_cls: type[gp.AbstractSinglePotential], fields: dict[str, Any]
    ):
        """Test the mass parameter."""
        fields["r_t"] = u.Quantity(1.0, "kpc")
        pot = pot_cls(**fields)
        assert pot.r_t(t=u.Quantity(0, "Myr")) == u.Quantity(1.0, "kpc")

    def test_r_t_userfunc(
        self, pot_cls: type[gp.AbstractSinglePotential], fields: dict[str, Any]
    ):
        """Test the scale radius parameter."""

        def cos_scalelength(t: u.Quantity["time"]) -> u.Quantity["length"]:
            return u.Quantity(10 * jnp.cos(t.ustrip("Myr")), "kpc")

        fields["r_t"] = cos_scalelength
        pot = pot_cls(**fields)
        assert pot.r_t(t=u.Quantity(0, "Myr")) == u.Quantity(10, "kpc")


###############################################################################


class TestHardCutoffNFWPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMMixin,
    ParameterRSMixin,
    ParameterRTMixin,
):
    HAS_GALA_COUNTERPART: ClassVar[bool] = False

    @pytest.fixture(scope="class")
    @override
    def pot_cls(self) -> type[gp.HardCutoffNFWPotential]:
        return gp.HardCutoffNFWPotential

    @pytest.fixture(scope="class")
    @override
    def fields_(
        self,
        field_m: u.Quantity,
        field_r_s: u.Quantity,
        field_r_t: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {"m": field_m, "r_s": field_r_s, "r_t": field_r_t, "units": field_units}

    # ==========================================================================

    def test_potential(self, pot: gp.HardCutoffNFWPotential, x: gt.QuSz3) -> None:
        exp = u.Quantity(-1.80505084, pot.units["specific energy"])
        got = pot.potential(x, t=0)
        assert jnp.isclose(got, exp, atol=u.Quantity(1e-8, exp.unit))

    def test_gradient(self, pot: gp.HardCutoffNFWPotential, x: gt.QuSz3) -> None:
        exp = u.Quantity([0.06589185, 0.1317837, 0.19767556], pot.units["acceleration"])
        got = pot.gradient(x, t=0)
        assert jnp.allclose(got, exp, atol=u.Quantity(1e-8, exp.unit))

    def test_density(self, pot: gp.HardCutoffNFWPotential, x: gt.QuSz3) -> None:
        got = pot.density(x, t=0)
        exp = u.Quantity(9.45944763e08, pot.units["mass density"])
        assert jnp.isclose(got, exp, atol=u.Quantity(1e-8, exp.unit))

    def test_hessian(self, pot: gp.HardCutoffNFWPotential, x: gt.QuSz3) -> None:
        exp = u.Quantity(
            [
                [0.05559175, -0.02060021, -0.03090031],
                [-0.02060021, 0.02469144, -0.06180062],
                [-0.03090031, -0.06180062, -0.02680908],
            ],
            "1/Myr2",
        )
        got = pot.hessian(x, t=0)
        assert jnp.allclose(got, exp, atol=u.Quantity(1e-8, exp.unit))

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        exp = u.Quantity(
            [
                [0.03776704, -0.02060021, -0.03090031],
                [-0.02060021, 0.00686674, -0.06180062],
                [-0.03090031, -0.06180062, -0.04463378],
            ],
            "1/Myr2",
        )
        got = pot.tidal_tensor(x, t=0)
        assert jnp.allclose(got, exp, atol=u.Quantity(1e-8, exp.unit))
