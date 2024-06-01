from typing import Any

import astropy.units as u
import pytest

import quaxed.numpy as qnp
from unxt import AbstractUnitSystem, Quantity

import galax.potential as gp
from ...test_core import AbstractPotential_Test
from ..test_common import ParameterMTotMixin, ParameterShapeAMixin, ParameterShapeBMixin
from galax.potential import AbstractPotentialBase, MiyamotoNagaiPotential
from galax.typing import Vec3


class TestMiyamotoNagaiPotential(
    AbstractPotential_Test,
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
        field_units: AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {"m_tot": field_m_tot, "a": field_a, "b": field_b, "units": field_units}

    # ==========================================================================

    def test_potential(self, pot: MiyamotoNagaiPotential, x: Vec3) -> None:
        expect = Quantity(-0.95208676, pot.units["specific energy"])
        assert qnp.isclose(
            pot.potential(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: MiyamotoNagaiPotential, x: Vec3) -> None:
        expect = Quantity(
            [0.04264751, 0.08529503, 0.16840152], pot.units["acceleration"]
        )
        assert qnp.allclose(
            pot.gradient(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_density(self, pot: MiyamotoNagaiPotential, x: Vec3) -> None:
        expect = Quantity(1.9949418e08, pot.units["mass density"])
        assert qnp.isclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: MiyamotoNagaiPotential, x: Vec3) -> None:
        expect = Quantity(
            [
                [0.03691649, -0.01146205, -0.02262999],
                [-0.01146205, 0.01972342, -0.04525999],
                [-0.02262999, -0.04525999, -0.04536254],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.hessian(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: AbstractPotentialBase, x: Vec3) -> None:
        """Test the `AbstractPotentialBase.tidal_tensor` method."""
        expect = Quantity(
            [
                [0.03315736, -0.01146205, -0.02262999],
                [-0.01146205, 0.0159643, -0.04525999],
                [-0.02262999, -0.04525999, -0.04912166],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )
