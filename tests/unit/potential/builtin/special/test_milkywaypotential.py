"""Test the `galax.potential.MilkyWayPotential` class."""

from collections.abc import Mapping

import pytest
from plum import convert

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.typing as gt
from .test_composite import AbstractSpecialCompositePotential_Test


class TestMilkyWayPotential(AbstractSpecialCompositePotential_Test):
    """Test the `galax.potential.MilkyWayPotential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.MilkyWayPotential]:
        return gp.MilkyWayPotential

    @pytest.fixture(scope="class")
    def pot_map(
        self, pot_cls: type[gp.MilkyWayPotential]
    ) -> dict[str, dict[str, u.Quantity]]:
        """Composite potential."""
        return {
            "disk": pot_cls.disk,
            "halo": pot_cls.halo,
            "bulge": pot_cls.bulge,
            "nucleus": pot_cls.nucleus,
        }

    @pytest.fixture(scope="class")
    def pot_map_unitless(self, pot_map) -> Mapping[str, gp.AbstractPotential]:
        """Composite potential."""
        return {k: {kk: vv.value for kk, vv in v.items()} for k, v in pot_map.items()}

    # ==========================================================================

    def test_potential(self, pot: gp.MilkyWayPotential, x: gt.QuSz3) -> None:
        """Test the :meth:`MilkyWayPotential.potential` method."""
        expect = u.Quantity(-0.19386052, pot.units["specific energy"])
        assert jnp.isclose(
            pot.potential(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: gp.MilkyWayPotential, x: gt.QuSz3) -> None:
        """Test the :meth:`MilkyWayPotential.gradient` method."""
        expect = u.Quantity(
            [0.00256407, 0.00512815, 0.01115285], pot.units["acceleration"]
        )
        got = convert(pot.gradient(x, t=0), u.Quantity)
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: gp.MilkyWayPotential, x: gt.QuSz3) -> None:
        """Test the :meth:`MilkyWayPotential.density` method."""
        expect = u.Quantity(33_365_858.46361218, pot.units["mass density"])
        assert jnp.isclose(
            pot.density(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: gp.MilkyWayPotential, x: gt.QuSz3) -> None:
        """Test the :meth:`MilkyWayPotential.hessian` method."""
        expect = u.Quantity(
            [
                [0.00231057, -0.000507, -0.00101276],
                [-0.000507, 0.00155007, -0.00202552],
                [-0.00101276, -0.00202552, -0.00197448],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.hessian(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        expect = u.Quantity(
            [
                [0.00168185, -0.000507, -0.00101276],
                [-0.000507, 0.00092135, -0.00202552],
                [-0.00101276, -0.00202552, -0.0026032],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )
