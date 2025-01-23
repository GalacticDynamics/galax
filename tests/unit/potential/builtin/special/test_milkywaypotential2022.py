from __future__ import annotations

from collections.abc import Mapping

import pytest
from plum import convert

import quaxed.numpy as qnp
import unxt as u

import galax.typing as gt
from .test_composite import AbstractSpecialCompositePotential_Test
from galax.potential import MilkyWayPotential2022


class TestMilkyWayPotential2022(AbstractSpecialCompositePotential_Test):
    """Test the `galax.potential.MilkyWayPotential2022` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[MilkyWayPotential2022]:
        return MilkyWayPotential2022

    @pytest.fixture(scope="class")
    def pot_map(
        self, pot_cls: type[MilkyWayPotential2022]
    ) -> dict[str, dict[str, u.Quantity]]:
        """Composite potential."""
        return {
            "disk": pot_cls.disk,
            "halo": pot_cls.halo,
            "bulge": pot_cls.bulge,
            "nucleus": pot_cls.nucleus,
        }

    @pytest.fixture(scope="class")
    def pot_map_unitless(self, pot_map) -> Mapping[str, AbstractPotential]:
        """Composite potential."""
        return {k: {kk: vv.value for kk, vv in v.items()} for k, v in pot_map.items()}

    # ==========================================================================

    def test_potential(self, pot: MilkyWayPotential2022, x: gt.QuSz3) -> None:
        """Test the :meth:`MilkyWayPotential2022.potential` method."""
        expect = u.Quantity(-0.1906119, pot.units["specific energy"])
        assert qnp.isclose(
            pot.potential(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: MilkyWayPotential2022, x: gt.QuSz3) -> None:
        """Test the :meth:`MilkyWayPotential2022.gradient` method."""
        expect = u.Quantity(
            [0.00235500422114, 0.00471000844229, 0.0101667940117],
            pot.units["acceleration"],
        )
        got = convert(pot.gradient(x, t=0), u.Quantity)
        assert qnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: MilkyWayPotential2022, x: gt.QuSz3) -> None:
        """Test the :meth:`MilkyWayPotential2022.density` method."""
        expect = u.Quantity(33_807_052.01837142, pot.units["mass density"])
        assert qnp.isclose(
            pot.density(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: MilkyWayPotential2022, x: gt.QuSz3) -> None:
        """Test the :meth:`MilkyWayPotential2022.hessian` method."""
        expect = u.Quantity(
            [
                [0.0021196, -0.00047082, -0.0008994],
                [-0.00047082, 0.00141337, -0.0017988],
                [-0.0008994, -0.0017988, -0.00162186],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.hessian(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        expect = u.Quantity(
            [
                [0.00148256, -0.00047082, -0.0008994],
                [-0.00047082, 0.00077633, -0.0017988],
                [-0.0008994, -0.0017988, -0.00225889],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )
