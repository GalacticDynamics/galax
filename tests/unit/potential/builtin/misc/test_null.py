from typing import Any
from typing_extensions import override

import pytest
from jaxtyping import Array
from plum import convert

import quaxed.numpy as jnp
import unxt as u
import unxt.unitsystems as usx

import galax.potential as gp
import galax.typing as gt
from ...test_core import AbstractPotential_Test


class TestNullPotential(AbstractPotential_Test):
    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.NullPotential]:
        return gp.NullPotential

    @pytest.fixture(scope="class")
    def fields_(self, field_units: usx.AbstractUnitSystem) -> dict[str, Any]:
        return {"units": field_units}

    # ==========================================================================

    @override
    def test_init_units_from_name(
        self, pot_cls: type[gp.AbstractBasePotential], fields_unitless: dict[str, Array]
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

    def test_potential(self, pot: gp.NullPotential, x: gt.QuSz3) -> None:
        """Test :meth:`NullPotential.potential`."""
        expect = u.Quantity(0.0, pot.units["specific energy"])
        assert jnp.isclose(
            pot.potential(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: gp.NullPotential, x: gt.QuSz3) -> None:
        """Test :meth:`NullPotential.gradient`."""
        expect = u.Quantity([0.0, 0.0, 0.0], pot.units["acceleration"])
        got = convert(pot.gradient(x, t=0), u.Quantity)
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: gp.NullPotential, x: gt.QuSz3) -> None:
        """Test :meth:`NullPotential.density`."""
        expect = u.Quantity(0.0, pot.units["mass density"])
        assert jnp.isclose(
            pot.density(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: gp.NullPotential, x: gt.QuSz3) -> None:
        """Test :meth:`NullPotential.hessian`."""
        expect = u.Quantity(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], "1/Myr2"
        )
        assert jnp.allclose(
            pot.hessian(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: gp.AbstractBasePotential, x: gt.QuSz3) -> None:
        """Test the `AbstractBasePotential.tidal_tensor` method."""
        expect = u.Quantity(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], "1/Myr2"
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )
