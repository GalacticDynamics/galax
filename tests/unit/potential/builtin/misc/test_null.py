from typing import Any

import pytest
from jaxtyping import Array
from typing_extensions import override

import quaxed.numpy as qnp
import unxt.unitsystems as usx
from unxt import AbstractUnitSystem, Quantity

import galax.potential as gp
import galax.typing as gt
from ...test_core import AbstractPotential_Test


class TestNullPotential(AbstractPotential_Test):
    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.NullPotential]:
        return gp.NullPotential

    @pytest.fixture(scope="class")
    def fields_(self, field_units: AbstractUnitSystem) -> dict[str, Any]:
        return {"units": field_units}

    # ==========================================================================

    def test_init_units_from_args(
        self, pot_cls: type[gp.AbstractPotentialBase], fields_unitless: dict[str, Array]
    ) -> None:
        """Test unit system from None."""
        # strip the units from the fields otherwise the test will fail
        # because the units are not equal and we just want to check that
        # when the units aren't specified, the default is dimensionless
        # and a numeric value works.
        fields_unitless.pop("units", None)
        pot = pot_cls(**fields_unitless, units=None)
        assert pot.units == usx.dimensionless

    @override
    def test_init_units_from_name(
        self, pot_cls: type[gp.AbstractPotentialBase], fields_unitless: dict[str, Array]
    ) -> None:
        """Test unit system from named string."""
        fields_unitless.pop("units")

        pot = pot_cls(**fields_unitless, units="dimensionless")
        assert pot.units == usx.dimensionless

        pot = pot_cls(**fields_unitless, units="solarsystem")
        assert pot.units == usx.solarsystem

        pot = pot_cls(**fields_unitless, units="galactic")
        assert pot.units == usx.galactic

        with pytest.raises(KeyError, match="invalid_value"):
            pot_cls(**fields_unitless, units="invalid_value")

    # ==========================================================================

    def test_potential(self, pot: gp.NullPotential, x: gt.QVec3) -> None:
        """Test :meth:`NullPotential.potential`."""
        expect = Quantity(0.0, pot.units["specific energy"])
        assert qnp.isclose(
            pot.potential(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: gp.NullPotential, x: gt.QVec3) -> None:
        """Test :meth:`NullPotential.gradient`."""
        expect = Quantity([0.0, 0.0, 0.0], pot.units["acceleration"])
        assert qnp.allclose(
            pot.gradient(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_density(self, pot: gp.NullPotential, x: gt.QVec3) -> None:
        """Test :meth:`NullPotential.density`."""
        expect = Quantity(0.0, pot.units["mass density"])
        assert qnp.isclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: gp.NullPotential, x: gt.QVec3) -> None:
        """Test :meth:`NullPotential.hessian`."""
        expect = Quantity([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], "1/Myr2")
        assert qnp.allclose(
            pot.hessian(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: gp.AbstractPotentialBase, x: gt.QVec3) -> None:
        """Test the `AbstractPotentialBase.tidal_tensor` method."""
        expect = Quantity([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], "1/Myr2")
        assert qnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )
