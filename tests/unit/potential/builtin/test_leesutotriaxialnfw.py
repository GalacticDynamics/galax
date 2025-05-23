from typing import Any
from typing_extensions import override

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
import galax.potential as gp
from ..param.test_field import ParameterFieldMixin
from ..test_core import AbstractSinglePotential_Test
from .test_common import ParameterMMixin, ParameterRSMixin
from galax._interop.optional_deps import OptDeps


class ShapeA1ParameterMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_a1(self) -> u.Quantity[""]:
        return u.Quantity(1.0, "")

    # =====================================================

    def test_a1_constant(self, pot_cls, fields):
        """Test the `a1` parameter."""
        fields["a1"] = u.Quantity(1.0, "")
        pot = pot_cls(**fields)
        assert pot.a1(t=u.Quantity(0, "Myr")) == u.Quantity(1.0, "")

    def test_a1_userfunc(self, pot_cls, fields):
        """Test the `a1` parameter."""

        def cos_a1(t: u.Quantity["time"]) -> u.Quantity[""]:
            return u.Quantity(jnp.abs(jnp.cos(t.ustrip("Myr"))) + 1.2, "")

        fields["a1"] = cos_a1
        fields["a2"] = u.Quantity(1.1, "")
        fields["a3"] = u.Quantity(1.0, "")
        pot = pot_cls(**fields)
        assert pot.a1(t=u.Quantity(0, "Myr")) == u.Quantity(2.2, "")


class ShapeA2ParameterMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_a2(self) -> u.Quantity[""]:
        return u.Quantity(1.0, "")

    # =====================================================

    def test_a2_constant(self, pot_cls, fields):
        """Test the `a2` parameter."""
        fields["a2"] = u.Quantity(1.0, "")
        pot = pot_cls(**fields)
        assert pot.a2(t=u.Quantity(0, "Myr")) == u.Quantity(1.0, "")

    def test_a2_userfunc(self, pot_cls, fields):
        """Test the `a2` parameter."""

        def cos_a2(t: u.Quantity["time"]) -> u.Quantity[""]:
            return u.Quantity(jnp.abs(jnp.cos(t.ustrip("Myr"))) / 2 + 1.1, "")

        fields["a1"] = u.Quantity(2, "")
        fields["a2"] = cos_a2
        fields["a3"] = u.Quantity(1, "")
        pot = pot_cls(**fields)
        assert pot.a2(t=u.Quantity(0, "Myr")) == u.Quantity(1.6, "")


class ShapeA3ParameterMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_a3(self) -> u.Quantity[""]:
        return u.Quantity(1.0, "")

    # =====================================================

    def test_a3_constant(self, pot_cls, fields):
        """Test the `a3` parameter."""
        fields["a3"] = u.Quantity(1.0, "")
        pot = pot_cls(**fields)
        assert pot.a3(t=u.Quantity(0, "Myr")) == u.Quantity(1.0, "")

    def test_a3_userfunc(self, pot_cls, fields):
        """Test the `a3` parameter."""

        def cos_a3(t: u.Quantity["time"]) -> u.Quantity[""]:
            return u.Quantity(jnp.abs(jnp.cos(t.ustrip("Myr"))) + 0.1, "")

        fields["a1"] = u.Quantity(1.3, "")
        fields["a2"] = u.Quantity(1.2, "")
        fields["a3"] = cos_a3
        pot = pot_cls(**fields)
        assert pot.a3(t=u.Quantity(0, "Myr")) == u.Quantity(1.1, "")


################################################################################


class TestLeeSutoTriaxialNFWPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMMixin,
    ParameterRSMixin,
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
        field_m: u.Quantity,
        field_r_s: u.Quantity,
        field_a1: u.Quantity,
        field_a2: u.Quantity,
        field_a3: u.Quantity,
        field_units: u.AbstractUnitSystem,
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

    def test_potential(self, pot: gp.LeeSutoTriaxialNFWPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(-9.68797618, pot.units["specific energy"])
        assert jnp.isclose(
            pot.potential(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: gp.LeeSutoTriaxialNFWPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(
            [0.3411484, 0.6822968, 1.0234452], pot.units["acceleration"]
        )
        got = pot.gradient(x, t=0)
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: gp.LeeSutoTriaxialNFWPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(4.89753338e09, pot.units["mass density"])
        assert jnp.isclose(
            pot.density(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: gp.LeeSutoTriaxialNFWPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(
            [
                [0.28782066, -0.10665549, -0.15998323],
                [-0.10665549, 0.12783743, -0.31996646],
                [-0.15998323, -0.31996646, -0.13880128],
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
                [0.19553506, -0.10665549, -0.15998323],
                [-0.10665549, 0.03555183, -0.31996646],
                [-0.15998323, -0.31996646, -0.23108689],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    # ==========================================================================
    # I/O

    @pytest.mark.skipif(not OptDeps.GALA.installed, reason="requires gala")
    @pytest.mark.parametrize(
        ("method0", "method1", "atol"),
        [
            ("potential", "energy", 1e-8),
            ("gradient", "gradient", 1e-8),
            ("density", "density", 1e-8),
            # ("hessian", "hessian", 1e-8),  # No hessian method in gala!
        ],
    )
    def test_method_gala(
        self,
        pot: gp.AbstractPotential,
        method0: str,
        method1: str,
        x: gt.QuSz3,
        atol: float,
    ) -> None:
        """Test the equivalence of methods between gala and galax.

        This test only runs if the potential can be mapped to gala.
        """
        super().test_method_gala(pot, method0, method1, x, atol)
