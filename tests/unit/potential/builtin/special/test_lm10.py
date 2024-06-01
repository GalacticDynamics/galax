from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import pytest
from typing_extensions import override

import quaxed.numpy as qnp
from unxt import Quantity
from unxt.unitsystems import galactic

import galax.potential as gp
import galax.typing as gt
from ...test_composite import AbstractCompositePotential_Test
from galax.potential import AbstractCompositePotential, LM10Potential
from galax.utils._optional_deps import HAS_GALA

if TYPE_CHECKING:
    from galax.potential import AbstractPotentialBase


class TestLM10Potential(AbstractCompositePotential_Test):
    """Test the `galax.potential.LM10Potential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.LM10Potential]:
        return gp.LM10Potential

    @pytest.fixture(scope="class")
    def pot_map(self, pot_cls: type[LM10Potential]) -> dict[str, dict[str, Quantity]]:
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
        pot_cls: type[AbstractCompositePotential],
        pot_map: Mapping[str, AbstractPotentialBase],
    ) -> None:
        """Test unit system from None."""
        pot = pot_cls(**pot_map, units=None)
        assert pot.units == galactic

    # ==========================================================================

    def test_potential(self, pot: LM10Potential, x: gt.QVec3) -> None:
        expect = Quantity(-0.00242568, unit="kpc2 / Myr2")
        assert qnp.isclose(
            pot.potential(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: LM10Potential, x: gt.QVec3) -> None:
        expect = Quantity([0.00278038, 0.00533753, 0.0111171], "kpc / Myr2")
        assert qnp.allclose(
            pot.gradient(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_density(self, pot: LM10Potential, x: gt.QVec3) -> None:
        expect = Quantity(19085831.78310305, "solMass / kpc3")
        assert qnp.isclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: LM10Potential, x: gt.QVec3) -> None:
        expect = Quantity(
            [
                [0.00234114, -0.00081663, -0.0013405],
                [-0.00081663, 0.00100949, -0.00267623],
                [-0.0013405, -0.00267623, -0.00227171],
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
                [0.0019815, -0.00081663, -0.0013405],
                [-0.00081663, 0.00064985, -0.00267623],
                [-0.0013405, -0.00267623, -0.00263135],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    # ==========================================================================
    # Interoperability

    @pytest.mark.skipif(not HAS_GALA, reason="requires gala")
    @pytest.mark.parametrize(
        ("method0", "method1", "atol"),
        [
            ("potential", "energy", 1e-8),
            ("gradient", "gradient", 1e-8),
            # ("density", "density", 1e-8),  # TODO: get gala and galax to agree
            # ("hessian", "hessian", 1e-8),  # TODO: get gala and galax to agree
        ],
    )
    def test_method_gala(
        self,
        pot: gp.AbstractPotentialBase,
        method0: str,
        method1: str,
        x: gt.QVec3,
        atol: float,
    ) -> None:
        """Test the equivalence of methods between gala and galax.

        This test only runs if the potential can be mapped to gala.
        """
        super().test_method_gala(pot, method0, method1, x, atol)
