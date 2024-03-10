from dataclasses import replace
from typing import Any

import astropy.units as u
import jax.numpy as jnp
import pytest
from quax import quaxify
from typing_extensions import override

import quaxed.array_api as xp
from jax_quantity import Quantity

import galax.potential as gp
from ..param.test_field import ParameterFieldMixin
from ..test_core import TestAbstractPotential as AbstractPotential_Test
from .test_common import MassParameterMixin
from galax.potential import (
    AbstractPotential,
    AbstractPotentialBase,
    ConstantParameter,
    NFWPotential,
)
from galax.typing import Vec3
from galax.units import UnitSystem, galactic
from galax.utils._optional_deps import HAS_GALA

allclose = quaxify(jnp.allclose)


class ScaleRadiusParameterMixin(ParameterFieldMixin):
    """Test the mass parameter."""

    pot_cls: type[AbstractPotential]

    @pytest.fixture(scope="class")
    def field_r_s(self) -> float:
        return 1.0 * u.kpc

    # =====================================================

    def test_r_s_units(
        self, pot_cls: type[AbstractPotential], fields: dict[str, Any]
    ) -> None:
        """Test the mass parameter."""
        fields["r_s"] = 1.0 * u.Unit(10 * u.kpc)
        fields["units"] = galactic
        pot = pot_cls(**fields)
        assert isinstance(pot.r_s, ConstantParameter)
        assert jnp.isclose(pot.r_s.value, 10)

    def test_r_s_constant(
        self, pot_cls: type[AbstractPotential], fields: dict[str, Any]
    ):
        """Test the mass parameter."""
        fields["r_s"] = 1.0
        pot = pot_cls(**fields)
        assert pot.r_s(t=0) == 1.0

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_r_s_userfunc(
        self, pot_cls: type[AbstractPotential], fields: dict[str, Any]
    ):
        """Test the mass parameter."""
        fields["r_s"] = lambda t: t + 2
        pot = pot_cls(**fields)
        assert pot.r_s(t=0) == 2


###############################################################################


class TestNFWPotential(
    AbstractPotential_Test,
    # Parameters
    MassParameterMixin,
    ScaleRadiusParameterMixin,
):
    @pytest.fixture(scope="class")
    @override
    def pot_cls(self) -> type[NFWPotential]:
        return NFWPotential

    @pytest.fixture(scope="class")
    def field_softening_length(self) -> float:
        return 0.001

    @pytest.fixture(scope="class")
    @override
    def fields_(
        self,
        field_m: u.Quantity,
        field_r_s: u.Quantity,
        field_softening_length: float,
        field_units: UnitSystem,
    ) -> dict[str, Any]:
        return {
            "m": field_m,
            "r_s": field_r_s,
            "softening_length": field_softening_length,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential_energy(self, pot: NFWPotential, x: Vec3) -> None:
        assert jnp.isclose(pot.potential_energy(x, t=0).value, xp.asarray(-1.87117234))

    def test_gradient(self, pot: NFWPotential, x: Vec3) -> None:
        expected = Quantity(
            [0.0658867, 0.1317734, 0.19766011], pot.units["acceleration"]
        )
        assert allclose(pot.gradient(x, t=0).value, expected.value)  # TODO: not .value

    def test_density(self, pot: NFWPotential, x: Vec3) -> None:
        assert jnp.isclose(pot.density(x, t=0).value, 9.46039849e08)

    def test_hessian(self, pot: NFWPotential, x: Vec3) -> None:
        assert jnp.allclose(
            pot.hessian(x, t=0),
            xp.asarray(
                [
                    [0.05558809, -0.02059723, -0.03089585],
                    [-0.02059723, 0.02469224, -0.06179169],
                    [-0.03089585, -0.06179169, -0.02680084],
                ]
            ),
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: AbstractPotentialBase, x: Vec3) -> None:
        """Test the `AbstractPotentialBase.tidal_tensor` method."""
        expect = [
            [0.03776159, -0.02059723, -0.03089585],
            [-0.02059723, 0.00686574, -0.06179169],
            [-0.03089585, -0.06179169, -0.04462733],
        ]
        assert allclose(pot.tidal_tensor(x, t=0), xp.asarray(expect))

    # ==========================================================================
    # I/O

    @pytest.mark.skipif(not HAS_GALA, reason="requires gala")
    def test_galax_to_gala_to_galax_roundtrip(self, pot: NFWPotential, x: Vec3) -> None:
        """Test roundtripping ``gala_to_galax(galax_to_gala())``."""
        from ..io.gala_helper import galax_to_gala

        # Base is with non-zero softening
        assert pot.softening_length != 0
        with pytest.raises(TypeError, match="Gala does not support softening"):
            _ = galax_to_gala(pot)

        # Make a copy without softening
        pot = replace(pot, softening_length=0)

        rpot = gp.io.gala_to_galax(galax_to_gala(pot))

        # quick test that the potential energies are the same
        assert jnp.array_equal(pot(x, t=0).value, rpot(x, t=0).value)
