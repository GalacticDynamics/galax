from typing import Any

import pytest
from typing_extensions import override

import quaxed.numpy as qnp
from unxt import Quantity
from unxt.unitsystems import AbstractUnitSystem

import galax.potential as gp
import galax.typing as gt
from ..param.test_field import ParameterFieldMixin
from ..test_core import TestAbstractPotential as AbstractPotential_Test
from .test_common import ParameterMMixin
from .test_nfw import ParameterScaleRadiusMixin
from galax.utils._optional_deps import HAS_GALA


class ShapeA1ParameterMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_a1(self) -> Quantity["length"]:
        return Quantity(1.0, "")

    # =====================================================

    def test_a1_constant(self, pot_cls, fields):
        """Test the `a1` parameter."""
        fields["a1"] = Quantity(1.0, "")
        pot = pot_cls(**fields)
        assert pot.a1(t=0) == Quantity(1.0, "")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_a1_userfunc(self, pot_cls, fields):
        """Test the `a1` parameter."""
        fields["a1"] = lambda t: t * 1.2
        pot = pot_cls(**fields)
        assert pot.a1(t=0) == 2


class ShapeA2ParameterMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_a2(self) -> Quantity["length"]:
        return Quantity(1.0, "")

    # =====================================================

    def test_a2_constant(self, pot_cls, fields):
        """Test the `a2` parameter."""
        fields["a2"] = Quantity(1.0, "")
        pot = pot_cls(**fields)
        assert pot.a2(t=0) == Quantity(1.0, "")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_a2_userfunc(self, pot_cls, fields):
        """Test the `a2` parameter."""
        fields["a3"] = lambda t: t * 1.2
        pot = pot_cls(**fields)
        assert pot.a2(t=0) == 2


class ShapeA3ParameterMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_a3(self) -> Quantity["length"]:
        return Quantity(1.0, "")

    # =====================================================

    def test_a3_constant(self, pot_cls, fields):
        """Test the `a3` parameter."""
        fields["a3"] = Quantity(1.0, "")
        pot = pot_cls(**fields)
        assert pot.a3(t=0) == Quantity(1.0, "")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_a3_userfunc(self, pot_cls, fields):
        """Test the `a3` parameter."""
        fields["a3"] = lambda t: t * 1.2
        pot = pot_cls(**fields)
        assert pot.a3(t=0) == 2


################################################################################


class TestLeeSutoTriaxialNFWPotential(
    AbstractPotential_Test,
    # Parameters
    ParameterMMixin,
    ParameterScaleRadiusMixin,
    ShapeA1ParameterMixin,
    ShapeA2ParameterMixin,
    ShapeA3ParameterMixin,
):
    @pytest.fixture(scope="class")
    @override
    def pot_cls(self) -> type[gp.LeeSutoTriaxialNFWPotential]:
        return gp.LeeSutoTriaxialNFWPotential

    @pytest.fixture(scope="class")
    @override
    def fields_(
        self,
        field_m: Quantity,
        field_r_s: Quantity,
        field_a1: Quantity,
        field_a2: Quantity,
        field_a3: Quantity,
        field_units: AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m": field_m,
            "r_s": field_r_s,
            "a1": field_a1,
            "a2": field_a2,
            "a3": field_a3,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential_energy(
        self, pot: gp.LeeSutoTriaxialNFWPotential, x: gt.Vec3
    ) -> None:
        expect = Quantity(-9.68797618, pot.units["specific energy"])
        assert qnp.isclose(
            pot.potential_energy(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: gp.LeeSutoTriaxialNFWPotential, x: gt.Vec3) -> None:
        expect = Quantity([0.3411484, 0.6822968, 1.0234452], pot.units["acceleration"])
        assert qnp.allclose(
            pot.gradient(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_density(self, pot: gp.LeeSutoTriaxialNFWPotential, x: gt.Vec3) -> None:
        expect = Quantity(4.89753338e09, pot.units["mass density"])
        assert qnp.isclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: gp.LeeSutoTriaxialNFWPotential, x: gt.Vec3) -> None:
        expect = Quantity(
            [
                [0.28782066, -0.10665549, -0.15998323],
                [-0.10665549, 0.12783743, -0.31996646],
                [-0.15998323, -0.31996646, -0.13880128],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.hessian(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: gp.AbstractPotentialBase, x: gt.Vec3) -> None:
        """Test the `AbstractPotentialBase.tidal_tensor` method."""
        expect = Quantity(
            [
                [0.19553506, -0.10665549, -0.15998323],
                [-0.10665549, 0.03555183, -0.31996646],
                [-0.15998323, -0.31996646, -0.23108689],
            ],
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    # ==========================================================================
    # I/O

    @pytest.mark.skipif(not HAS_GALA, reason="requires gala")
    def test_galax_to_gala_to_galax_roundtrip(
        self, pot: gp.LeeSutoTriaxialNFWPotential, x: gt.Vec3
    ) -> None:
        """Test roundtripping ``gala_to_galax(galax_to_gala())``."""
        from ..io.gala_helper import galax_to_gala

        rpot = gp.io.gala_to_galax(galax_to_gala(pot))

        # quick test that the potential energies are the same
        assert qnp.allclose(
            pot(x, t=0),
            rpot(x, t=0),
            atol=Quantity(1e-14, rpot.units["specific energy"]),
        )
