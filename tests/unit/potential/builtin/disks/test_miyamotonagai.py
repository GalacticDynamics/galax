from typing import Any

import pytest
from plum import convert

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
from ...test_core import AbstractSinglePotential_Test
from ..test_common import ParameterMTotMixin, ParameterShapeAMixin, ParameterShapeBMixin
from galax._custom_types import Sz3
from galax.potential import AbstractPotential, MiyamotoNagaiPotential


class TestMiyamotoNagaiPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMTotMixin,
    ParameterShapeAMixin,
    ParameterShapeBMixin,
):
    """Test the `galax.potential.MiyamotoNagaiPotential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.MiyamotoNagaiPotential]:
        return gp.MiyamotoNagaiPotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m_tot: u.Quantity,
        field_a: u.Quantity,
        field_b: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {"m_tot": field_m_tot, "a": field_a, "b": field_b, "units": field_units}

    # ==========================================================================

    def test_potential(self, pot: MiyamotoNagaiPotential, x: Sz3) -> None:
        expect = u.Quantity(-0.95208676, pot.units["specific energy"])
        assert jnp.isclose(
            pot.potential(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: MiyamotoNagaiPotential, x: Sz3) -> None:
        expect = u.Quantity(
            [0.04264751, 0.08529503, 0.16840152], pot.units["acceleration"]
        )
        got = convert(pot.gradient(x, t=0), u.Quantity)
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: MiyamotoNagaiPotential, x: Sz3) -> None:
        expect = u.Quantity(1.9949418e08, pot.units["mass density"])
        assert jnp.isclose(
            pot.density(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: MiyamotoNagaiPotential, x: Sz3) -> None:
        expect = u.Quantity(
            [
                [0.03691649, -0.01146205, -0.02262999],
                [-0.01146205, 0.01972342, -0.04525999],
                [-0.02262999, -0.04525999, -0.04536254],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.hessian(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: AbstractPotential, x: Sz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        expect = u.Quantity(
            [
                [0.03315736, -0.01146205, -0.02262999],
                [-0.01146205, 0.0159643, -0.04525999],
                [-0.02262999, -0.04525999, -0.04912166],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )
