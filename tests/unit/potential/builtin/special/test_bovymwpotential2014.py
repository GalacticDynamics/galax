from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typing_extensions import override

import quaxed.numpy as qnp
from unxt import Quantity
from unxt.unitsystems import galactic

import galax.potential as gp
import galax.typing as gt
from ...io.test_gala import parametrize_test_method_gala
from ...test_composite import AbstractCompositePotential_Test
from galax.potential import BovyMWPotential2014
from galax.utils._optional_deps import GSL_ENABLED, HAS_GALA

if TYPE_CHECKING:
    from galax.potential import AbstractPotentialBase


class TestBovyMWPotential2014(AbstractCompositePotential_Test):
    """Test the `galax.potential.BovyMWPotential2014` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.BovyMWPotential2014]:
        return gp.BovyMWPotential2014

    @pytest.fixture(scope="class")
    def pot_map(
        self, pot_cls: type[BovyMWPotential2014]
    ) -> dict[str, dict[str, Quantity]]:
        """Composite potential."""
        return {
            "disk": pot_cls._default_disk,
            "bulge": pot_cls._default_bulge,
            "halo": pot_cls._default_halo,
        }

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

    def test_potential(self, pot: BovyMWPotential2014, x: gt.QVec3) -> None:
        expect = Quantity(-0.09550731, unit="kpc2 / Myr2")
        assert qnp.isclose(
            pot.potential(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: BovyMWPotential2014, x: gt.QVec3) -> None:
        expect = Quantity([0.00231875, 0.0046375, 0.01042675], "kpc / Myr2")
        assert qnp.allclose(
            pot.gradient(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_density(self, pot: BovyMWPotential2014, x: gt.QVec3) -> None:
        expect = Quantity(24_911_277.33877818, "solMass / kpc3")
        assert qnp.isclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: BovyMWPotential2014, x: gt.QVec3) -> None:
        expect = Quantity(
            [
                [0.00208414, -0.00046922, -0.0009568],
                [-0.00046922, 0.00138031, -0.00191361],
                [-0.0009568, -0.00191361, -0.00205622],
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
                [0.00161473, -0.00046922, -0.0009568],
                [-0.00046922, 0.0009109, -0.00191361],
                [-0.0009568, -0.00191361, -0.00252563],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Interoperability

    @pytest.mark.skipif(not HAS_GALA or not GSL_ENABLED, reason="requires gala + GSL")
    def test_galax_to_gala_to_galax_roundtrip(
        self, pot: gp.AbstractPotentialBase, x: gt.QVec3
    ) -> None:
        super().test_galax_to_gala_to_galax_roundtrip(pot, x)

    @pytest.mark.skipif(not HAS_GALA or not GSL_ENABLED, reason="requires gala + GSL")
    @parametrize_test_method_gala
    def test_method_gala(
        self,
        pot: BovyMWPotential2014,
        method0: str,
        method1: str,
        x: gt.QVec3,
        atol: float,
    ) -> None:
        super().test_method_gala(pot, method0, method1, x, atol)
