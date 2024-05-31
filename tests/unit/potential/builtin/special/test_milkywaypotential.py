from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import pytest
from typing_extensions import override

import quaxed.numpy as qnp
from unxt import Quantity
from unxt.unitsystems import galactic

import galax.typing as gt
from ...test_composite import AbstractCompositePotential_Test
from galax.potential import MilkyWayPotential

if TYPE_CHECKING:
    from galax.potential import AbstractPotentialBase


##############################################################################


class TestMilkyWayPotential(AbstractCompositePotential_Test):
    """Test the `galax.potential.MilkyWayPotential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[MilkyWayPotential]:
        return MilkyWayPotential

    @pytest.fixture(scope="class")
    def pot_map(
        self, pot_cls: type[MilkyWayPotential]
    ) -> dict[str, dict[str, Quantity]]:
        """Composite potential."""
        return {
            "disk": pot_cls._default_disk,
            "halo": pot_cls._default_halo,
            "bulge": pot_cls._default_bulge,
            "nucleus": pot_cls._default_nucleus,
        }

    @pytest.fixture(scope="class")
    def pot_map_unitless(self, pot_map) -> Mapping[str, AbstractPotentialBase]:
        """Composite potential."""
        return {k: {kk: vv.value for kk, vv in v.items()} for k, v in pot_map.items()}

    # ==========================================================================

    @override
    def test_init_units_from_args(
        self,
        pot_cls: type[MilkyWayPotential],
        pot_map: Mapping[str, AbstractPotentialBase],
    ) -> None:
        """Test unit system from None."""
        pot = pot_cls(**pot_map, units=None)
        assert pot.units == galactic

    # ==========================================================================

    def test_potential(self, pot: MilkyWayPotential, x: gt.QVec3) -> None:
        """Test the :meth:`MilkyWayPotential.potential` method."""
        expect = Quantity(-0.19386052, pot.units["specific energy"])
        assert qnp.isclose(
            pot.potential(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: MilkyWayPotential, x: gt.QVec3) -> None:
        """Test the :meth:`MilkyWayPotential.gradient` method."""
        expect = Quantity(
            [0.00256407, 0.00512815, 0.01115285], pot.units["acceleration"]
        )
        assert qnp.allclose(
            pot.gradient(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_density(self, pot: MilkyWayPotential, x: gt.QVec3) -> None:
        """Test the :meth:`MilkyWayPotential.density` method."""
        expect = Quantity(33_365_858.46361218, pot.units["mass density"])
        assert qnp.isclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: MilkyWayPotential, x: gt.QVec3) -> None:
        """Test the :meth:`MilkyWayPotential.hessian` method."""
        expect = Quantity(
            [
                [0.00231057, -0.000507, -0.00101276],
                [-0.000507, 0.00155007, -0.00202552],
                [-0.00101276, -0.00202552, -0.00197448],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.hessian(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: AbstractPotentialBase, x: gt.QVec3) -> None:
        """Test the `AbstractPotentialBase.tidal_tensor` method."""
        expect = Quantity(
            [
                [0.00168185, -0.000507, -0.00101276],
                [-0.000507, 0.00092135, -0.00202552],
                [-0.00101276, -0.00202552, -0.0026032],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )
