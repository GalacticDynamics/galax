from typing import Any

import astropy.units as u
import pytest
from plum import convert

import quaxed.numpy as qnp
from unxt import AbstractUnitSystem, Quantity

import galax.potential as gp
import galax.typing as gt
from ..test_core import TestAbstractPotential as AbstractPotential_Test
from .test_common import ParameterMTotMixin, ShapeBParameterMixin
from galax.potential import AbstractPotentialBase, PlummerPotential
from galax.utils._optional_deps import HAS_GALA


class TestPlummerPotential(
    AbstractPotential_Test,
    # Parameters
    ParameterMTotMixin,
    ShapeBParameterMixin,
):
    """Test the `galax.potential.PlummerPotential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.PlummerPotential]:
        return gp.PlummerPotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m_tot: u.Quantity,
        field_b: u.Quantity,
        field_units: AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {"m_tot": field_m_tot, "b": field_b, "units": field_units}

    # ==========================================================================

    def test_potential_energy(self, pot: PlummerPotential, x: gt.QVec3) -> None:
        expect = Quantity(-1.16150826, unit="kpc2 / Myr2")
        assert qnp.isclose(
            pot.potential_energy(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: PlummerPotential, x: gt.QVec3) -> None:
        expect = Quantity([0.07743388, 0.15486777, 0.23230165], "kpc / Myr2")
        assert qnp.allclose(
            pot.gradient(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_density(self, pot: PlummerPotential, x: gt.QVec3) -> None:
        expect = Quantity(2.73957531e08, "solMass / kpc3")
        assert qnp.isclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: PlummerPotential, x: gt.QVec3) -> None:
        expect = Quantity(
            [
                [0.06194711, -0.03097355, -0.04646033],
                [-0.03097355, 0.01548678, -0.09292066],
                [-0.04646033, -0.09292066, -0.06194711],
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
                [0.05678485, -0.03097355, -0.04646033],
                [-0.03097355, 0.01032452, -0.09292066],
                [-0.04646033, -0.09292066, -0.06710937],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Interoperability

    @pytest.mark.skipif(not HAS_GALA, reason="requires gala")
    @pytest.mark.parametrize(
        ("method0", "method1", "atol"),
        [
            ("potential_energy", "energy", 1e-8),
            ("gradient", "gradient", 1e-8),
            ("density", "density", 1e-8),  # TODO: why is this different?
            ("hessian", "hessian", 1e-8),  # TODO: why is gala's 0?
        ],
    )
    def test_method_gala(
        self,
        pot: PlummerPotential,
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
