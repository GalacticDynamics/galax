from typing import Any

import astropy.units as u
import pytest
from typing_extensions import override

import quaxed.numpy as qnp
from unxt import Quantity
from unxt.unitsystems import AbstractUnitSystem

import galax.potential as gp
import galax.typing as gt
from ...test_core import AbstractPotential_Test
from ..test_common import ParameterMMixin, ParameterScaleRadiusMixin

###############################################################################


class TestNFWPotential(
    AbstractPotential_Test,
    # Parameters
    ParameterMMixin,
    ParameterScaleRadiusMixin,
):
    @pytest.fixture(scope="class")
    @override
    def pot_cls(self) -> type[gp.NFWPotential]:
        return gp.NFWPotential

    @pytest.fixture(scope="class")
    @override
    def fields_(
        self,
        field_m: u.Quantity,
        field_r_s: u.Quantity,
        field_units: AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {"m": field_m, "r_s": field_r_s, "units": field_units}

    # ==========================================================================

    def test_potential(self, pot: gp.NFWPotential, x: gt.QVec3) -> None:
        expect = Quantity(-1.87120528, pot.units["specific energy"])
        assert qnp.isclose(
            pot.potential(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: gp.NFWPotential, x: gt.QVec3) -> None:
        expect = Quantity(
            [0.06589185, 0.1317837, 0.19767556], pot.units["acceleration"]
        )
        assert qnp.allclose(
            pot.gradient(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_density(self, pot: gp.NFWPotential, x: gt.QVec3) -> None:
        expect = Quantity(9.45944763e08, pot.units["mass density"])
        assert qnp.isclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: gp.NFWPotential, x: gt.QVec3) -> None:
        expect = Quantity(
            [
                [0.05559175, -0.02060021, -0.03090031],
                [-0.02060021, 0.02469144, -0.06180062],
                [-0.03090031, -0.06180062, -0.02680908],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.hessian(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: gp.AbstractPotentialBase, x: gt.QVec3) -> None:
        """Test the `AbstractPotentialBase.tidal_tensor` method."""
        expect = Quantity(
            [
                [0.03776704, -0.02060021, -0.03090031],
                [-0.02060021, 0.00686674, -0.06180062],
                [-0.03090031, -0.06180062, -0.04463378],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )
