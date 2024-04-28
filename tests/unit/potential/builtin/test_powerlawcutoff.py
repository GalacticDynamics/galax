from typing import Any

import astropy.units as u
import pytest
from plum import convert

import quaxed.numpy as qnp
from unxt import AbstractUnitSystem, Quantity

import galax.potential as gp
import galax.typing as gt
from ..param.test_field import ParameterFieldMixin
from ..test_core import TestAbstractPotential as AbstractPotential_Test
from .test_common import ParameterMTotMixin
from galax.potential import AbstractPotentialBase, PowerLawCutoffPotential
from galax.utils._optional_deps import HAS_GALA


class AlphaParameterMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_alpha(self) -> Quantity["dimensionless"]:
        return Quantity(0.9, "")

    # =====================================================

    def test_alpha_constant(self, pot_cls, fields):
        """Test the `alpha` parameter."""
        fields["alpha"] = Quantity(1.0, "")
        pot = pot_cls(**fields)
        assert pot.alpha(t=0) == Quantity(1.0, "")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_alpha_userfunc(self, pot_cls, fields):
        """Test the `alpha` parameter."""
        fields["alpha"] = lambda t: t * 1.2
        pot = pot_cls(**fields)
        assert pot.alpha(t=0) == 2


class RCParameterMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_r_c(self) -> Quantity["length"]:
        return Quantity(1.0, "kpc")

    # =====================================================

    def test_r_c_constant(self, pot_cls, fields):
        """Test the `r_c` parameter."""
        fields["r_c"] = Quantity(1.0, "kpc")
        pot = pot_cls(**fields)
        assert pot.r_c(t=0) == Quantity(1.0, "kpc")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_r_c_userfunc(self, pot_cls, fields):
        """Test the `r_c` parameter."""
        fields["r_c"] = lambda t: t * 1.2
        pot = pot_cls(**fields)
        assert pot.r_c(t=0) == 2


class TestPowerLawCutoffPotential(
    AbstractPotential_Test,
    # Parameters
    ParameterMTotMixin,
    AlphaParameterMixin,
    RCParameterMixin,
):
    """Test the `galax.potential.PowerLawCutoffPotential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.PowerLawCutoffPotential]:
        return gp.PowerLawCutoffPotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m_tot: u.Quantity,
        field_alpha: u.Quantity,
        field_r_c: u.Quantity,
        field_units: AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m_tot": field_m_tot,
            "alpha": field_alpha,
            "r_c": field_r_c,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential_energy(self, pot: PowerLawCutoffPotential, x: gt.QVec3) -> None:
        expect = Quantity(6.26573365, unit="kpc2 / Myr2")
        assert qnp.isclose(
            pot.potential_energy(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: PowerLawCutoffPotential, x: gt.QVec3) -> None:
        expect = Quantity([0.08587672, 0.17175344, 0.25763016], "kpc / Myr2")
        assert qnp.allclose(
            pot.gradient(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_density(self, pot: PowerLawCutoffPotential, x: gt.QVec3) -> None:
        expect = Quantity(41457.38551946, "solMass / kpc3")
        assert qnp.isclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: PowerLawCutoffPotential, x: gt.QVec3) -> None:
        expect = Quantity(
            [
                [0.06747473, -0.03680397, -0.05520596],
                [-0.03680397, 0.01226877, -0.11041192],
                [-0.05520596, -0.11041192, -0.07974116],
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
                [0.06747395, -0.03680397, -0.05520596],
                [-0.03680397, 0.01226799, -0.11041192],
                [-0.05520596, -0.11041192, -0.07974194],
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
            ("density", "density", 1e-8),
            ("hessian", "hessian", 1e-8),
        ],
    )
    def test_potential_energy_gala(
        self,
        pot: PowerLawCutoffPotential,
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
