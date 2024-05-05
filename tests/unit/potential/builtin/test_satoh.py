from typing import Any

import astropy.units as u
import pytest
from plum import convert

import quaxed.numpy as qnp
from unxt import AbstractUnitSystem, Quantity

import galax.potential as gp
import galax.typing as gt
from ..test_core import TestAbstractPotential as AbstractPotential_Test
from .test_common import ParameterMTotMixin, ParameterShapeAMixin, ParameterShapeBMixin
from galax.potential import AbstractPotentialBase, SatohPotential
from galax.utils._optional_deps import GSL_ENABLED, HAS_GALA


class TestSatohPotential(
    AbstractPotential_Test,
    # Parameters
    ParameterMTotMixin,
    ParameterShapeAMixin,
    ParameterShapeBMixin,
):
    """Test the `galax.potential.SatohPotential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.SatohPotential]:
        return gp.SatohPotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m_tot: u.Quantity,
        field_a: u.Quantity,
        field_b: u.Quantity,
        field_units: AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m_tot": field_m_tot,
            "a": field_a,
            "b": field_b,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential_energy(self, pot: SatohPotential, x: gt.QVec3) -> None:
        expect = Quantity(-0.97415472, unit="kpc2 / Myr2")
        assert qnp.isclose(
            pot.potential_energy(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: SatohPotential, x: gt.QVec3) -> None:
        expect = Quantity([0.0456823, 0.0913646, 0.18038493], "kpc / Myr2")
        assert qnp.allclose(
            pot.gradient(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_density(self, pot: SatohPotential, x: gt.QVec3) -> None:
        expect = Quantity(1.08825455e08, "solMass / kpc3")
        assert qnp.isclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: SatohPotential, x: gt.QVec3) -> None:
        expect = Quantity(
            [
                [0.03925558, -0.01285344, -0.02537707],
                [-0.01285344, 0.01997543, -0.05075415],
                [-0.02537707, -0.05075415, -0.05307912],
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
                [0.03720495, -0.01285344, -0.02537707],
                [-0.01285344, 0.0179248, -0.05075415],
                [-0.02537707, -0.05075415, -0.05512975],
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
    @pytest.mark.parametrize(
        ("method0", "method1", "atol"),
        [
            ("potential_energy", "energy", 1e-8),
            ("gradient", "gradient", 1e-8),
            ("density", "density", 1e-8),
            ("hessian", "hessian", 1e-8),
        ],
    )
    def test_method_gala(
        self,
        pot: SatohPotential,
        method0: str,
        method1: str,
        x: gt.QVec3,
        atol: float,
    ) -> None:
        from ..io.gala_helper import galax_to_gala

        galax = getattr(pot, method0)(x, t=0)
        gala = getattr(galax_to_gala(pot), method1)(convert(x, u.Quantity), t=0 * u.Myr)
        assert qnp.allclose(
            qnp.ravel(galax),
            qnp.ravel(convert(gala, Quantity)),
            atol=Quantity(atol, galax.unit),
        )
