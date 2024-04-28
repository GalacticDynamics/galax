from typing import Any

import astropy.units as u
import pytest
from plum import convert

import quaxed.numpy as qnp
from unxt import Quantity

import galax.typing as gt
from ..param.test_field import ParameterFieldMixin
from ..test_core import TestAbstractPotential as AbstractPotential_Test
from galax.potential import HarmonicOscillatorPotential
from galax.potential._potential.base import AbstractPotentialBase
from galax.utils._optional_deps import HAS_GALA


class ParameterOmegaMixin(ParameterFieldMixin):
    """Test the omega parameter."""

    @pytest.fixture(scope="class")
    def field_omega(self) -> Quantity["frequency"]:
        return Quantity(1.0, "Hz")

    # =====================================================

    def test_omega_constant(self, pot_cls, fields):
        """Test the `omega` parameter."""
        fields["omega"] = Quantity(1.0, "Hz")
        pot = pot_cls(**fields)
        assert pot.omega(t=0) == Quantity(1.0, "Hz")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_omega_userfunc(self, pot_cls, fields):
        """Test the `omega` parameter."""
        fields["omega"] = lambda t: t * 1.2
        pot = pot_cls(**fields)
        assert pot.omega(t=0) == 2


class TestHarmonicOscillatorPotential(
    AbstractPotential_Test,
    # Parameters
    ParameterOmegaMixin,
):
    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[HarmonicOscillatorPotential]:
        return HarmonicOscillatorPotential

    @pytest.fixture(scope="class")
    def fields_(self, field_omega, field_units) -> dict[str, Any]:
        return {"omega": field_omega, "units": field_units}

    # ==========================================================================

    def test_potential_energy(
        self, pot: HarmonicOscillatorPotential, x: gt.Vec3
    ) -> None:
        expect = Quantity(-0.94871936, pot.units["specific energy"])
        assert qnp.isclose(
            pot.potential_energy(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: HarmonicOscillatorPotential, x: gt.Vec3) -> None:
        expect = Quantity(
            [0.05347411, 0.10694822, 0.16042233], pot.units["acceleration"]
        )
        assert qnp.allclose(
            pot.gradient(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_density(self, pot: HarmonicOscillatorPotential, x: gt.Vec3) -> None:
        expect = Quantity(3.989933e08, pot.units["mass density"])
        assert qnp.isclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: HarmonicOscillatorPotential, x: gt.Vec3) -> None:
        expect = Quantity(
            [
                [0.04362645, -0.01969533, -0.02954299],
                [-0.01969533, 0.01408345, -0.05908599],
                [-0.02954299, -0.05908599, -0.03515487],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.hessian(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: AbstractPotentialBase, x: gt.Vec3) -> None:
        """Test the `AbstractPotentialBase.tidal_tensor` method."""
        expect = Quantity(
            [
                [0.0361081, -0.01969533, -0.02954299],
                [-0.01969533, 0.00656511, -0.05908599],
                [-0.02954299, -0.05908599, -0.04267321],
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
            ("density", "density", 5e-7),  # TODO: why is this different?
            # ("hessian", "hessian", 1e-8),  # TODO: why is gala's 0?
        ],
    )
    def test_potential_energy_gala(
        self,
        pot: HarmonicOscillatorPotential,
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
