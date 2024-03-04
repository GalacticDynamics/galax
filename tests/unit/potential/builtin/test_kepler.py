from typing import Any

import pytest

import quaxed.numpy as qnp
from unxt import Quantity

import galax.potential as gp
from ..test_core import TestAbstractPotential as AbstractPotential_Test
from .test_common import MassParameterMixin
from galax.potential import AbstractPotentialBase, KeplerPotential
from galax.typing import QVec3


class TestKeplerPotential(
    AbstractPotential_Test,
    # Parameters
    MassParameterMixin,
):
    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.KeplerPotential]:
        return gp.KeplerPotential

    @pytest.fixture(scope="class")
    def fields_(self, field_m, field_units) -> dict[str, Any]:
        return {"m": field_m, "units": field_units}

    # ==========================================================================

    def test_potential_energy(self, pot: KeplerPotential, x: QVec3) -> None:
        expect = Quantity(-1.20227527, pot.units["specific energy"])
        assert qnp.isclose(  # TODO: .value & use pytest-arraydiff
            pot.potential_energy(x, t=0).decompose(pot.units).value,
            expect.value,
        )

    def test_gradient(self, pot: KeplerPotential, x: QVec3) -> None:
        expected = Quantity(
            [0.08587681, 0.17175361, 0.25763042], pot.units["acceleration"]
        )
        assert qnp.allclose(  # TODO: .value & use pytest-arraydiff
            pot.gradient(x, t=0).decompose(pot.units).value, expected.value
        )

    def test_density(self, pot: KeplerPotential, x: QVec3) -> None:
        expect = Quantity(2.64743093e-07, pot.units["mass density"])
        assert qnp.isclose(  # TODO: .value & use pytest-arraydiff
            pot.density(x, t=0).decompose(pot.units).value, expect.value
        )

    def test_hessian(self, pot: KeplerPotential, x: QVec3) -> None:
        expect = Quantity(
            [
                [0.06747463, -0.03680435, -0.05520652],
                [-0.03680435, 0.01226812, -0.11041304],
                [-0.05520652, -0.11041304, -0.07974275],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(  # TODO: .value & use pytest-arraydiff
            pot.hessian(x, t=0).decompose(pot.units).value, expect.value
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: AbstractPotentialBase, x: QVec3) -> None:
        """Test the `AbstractPotentialBase.tidal_tensor` method."""
        expect = Quantity(
            [
                [0.06747463, -0.03680435, -0.05520652],
                [-0.03680435, 0.01226812, -0.11041304],
                [-0.05520652, -0.11041304, -0.07974275],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(  # TODO: .value & use pytest-arraydiff
            pot.tidal_tensor(x, t=0).decompose(pot.units).value, expect.value
        )
