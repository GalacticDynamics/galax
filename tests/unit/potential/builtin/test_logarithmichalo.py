from typing import Any

import astropy.units as u
import pytest
from plum import convert

import quaxed.numpy as qnp
from unxt import AbstractUnitSystem, Quantity
from unxt.unitsystems import galactic

import galax.potential as gp
import galax.typing as gt
from ..param.test_field import ParameterFieldMixin
from ..test_core import TestAbstractPotential as AbstractPotential_Test
from galax.potential import (
    AbstractPotentialBase,
    ConstantParameter,
    LogarithmicPotential,
)
from galax.utils._optional_deps import HAS_GALA


class ParameterVCMixin(ParameterFieldMixin):
    """Test the circular velocity parameter."""

    pot_cls: type[gp.AbstractPotential]

    @pytest.fixture(scope="class")
    def field_v_c(self) -> Quantity["speed"]:
        return Quantity(220, "km/s")

    # =====================================================

    def test_v_c_units(self, pot_cls, fields):
        """Test the speed parameter."""
        fields["v_c"] = Quantity(1.0, u.Unit(220 * u.km / u.s))
        fields["units"] = galactic
        pot = pot_cls(**fields)
        assert isinstance(pot.v_c, ConstantParameter)
        assert pot.v_c.value == Quantity(220, "km/s")

    def test_v_c_constant(self, pot_cls, fields):
        """Test the speed parameter."""
        fields["v_c"] = Quantity(1.0, "km/s")
        pot = pot_cls(**fields)
        assert pot.v_c(t=0) == Quantity(1.0, "km/s")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_v_c_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["v_c"] = lambda t: t + 2
        pot = pot_cls(**fields)
        assert pot.v_c(t=0) == 2


class ParameterRHMixin(ParameterFieldMixin):
    """Test the scale radius parameter."""

    pot_cls: type[gp.AbstractPotential]

    @pytest.fixture(scope="class")
    def field_r_h(self) -> Quantity["length"]:
        return Quantity(8, "kpc")

    # =====================================================

    def test_r_h_units(self, pot_cls, fields):
        """Test the speed parameter."""
        fields["r_h"] = Quantity(1, u.Unit(10 * u.kpc))
        fields["units"] = galactic
        pot = pot_cls(**fields)
        assert isinstance(pot.r_h, ConstantParameter)
        assert qnp.isclose(
            pot.r_h.value, Quantity(10, "kpc"), atol=Quantity(1e-15, "kpc")
        )

    def test_r_h_constant(self, pot_cls, fields):
        """Test the speed parameter."""
        fields["r_h"] = Quantity(11.0, "kpc")
        pot = pot_cls(**fields)
        assert pot.r_h(t=0) == Quantity(11.0, "kpc")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_r_h_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        fields["r_h"] = lambda t: t + 2
        pot = pot_cls(**fields)
        assert pot.r_h(t=0) == 2


class TestLogarithmicPotential(
    AbstractPotential_Test,
    # Parameters
    ParameterVCMixin,
    ParameterRHMixin,
):
    """Test the `galax.potential.LogarithmicPotential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.LogarithmicPotential]:
        return gp.LogarithmicPotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_v_c: u.Quantity,
        field_r_h: u.Quantity,
        field_units: AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {"v_c": field_v_c, "r_h": field_r_h, "units": field_units}

    # ==========================================================================

    def test_potential_energy(self, pot: LogarithmicPotential, x: gt.QVec3) -> None:
        expect = Quantity(0.11027593, unit="kpc2 / Myr2")
        assert qnp.isclose(
            pot.potential_energy(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: LogarithmicPotential, x: gt.QVec3) -> None:
        expect = Quantity([0.00064902, 0.00129804, 0.00194706], "kpc / Myr2")
        assert qnp.allclose(
            pot.gradient(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_density(self, pot: LogarithmicPotential, x: gt.QVec3) -> None:
        expect = Quantity(30321621.61178864, "solMass / kpc3")
        assert qnp.isclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: LogarithmicPotential, x: gt.QVec3) -> None:
        expect = Quantity(
            [
                [6.32377766e-04, -3.32830403e-05, -4.99245605e-05],
                [-3.32830403e-05, 5.82453206e-04, -9.98491210e-05],
                [-4.99245605e-05, -9.98491210e-05, 4.99245605e-04],
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
                [6.10189073e-05, -3.32830403e-05, -4.99245605e-05],
                [-3.32830403e-05, 1.10943468e-05, -9.98491210e-05],
                [-4.99245605e-05, -9.98491210e-05, -7.21132541e-05],
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
        pot: LogarithmicPotential,
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
