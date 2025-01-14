from typing import Any, ClassVar

import pytest
from packaging.version import Version
from plum import convert

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.typing as gt
from ...test_core import AbstractPotential_Test
from ..test_common import ParameterMMixin, ParameterScaleRadiusMixin
from galax._interop.optional_deps import OptDeps
from galax.potential import AbstractBasePotential, BurkertPotential


class TestBurkertPotential(
    AbstractPotential_Test,
    # Parameters
    ParameterMMixin,
    ParameterScaleRadiusMixin,
):
    """Test the `galax.potential.BurkertPotential` class."""

    HAS_GALA_COUNTERPART: ClassVar[bool] = OptDeps.GALA.installed and (
        Version("1.8.2") <= OptDeps.GALA
    )

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.BurkertPotential]:
        return gp.BurkertPotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m: u.Quantity,
        field_r_s: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {"m": field_m, "r_s": field_r_s, "units": field_units}

    # ==========================================================================

    def test_potential(self, pot: BurkertPotential, x: gt.Sz3) -> None:
        expect = u.Quantity(-15.76623941, "kpc2 / Myr2")
        assert jnp.isclose(
            pot.potential(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: BurkertPotential, x: gt.Sz3) -> None:
        expect = u.Quantity([0.54053104, 1.08106208, 1.62159313], "kpc / Myr2")
        got = convert(pot.gradient(x, t=0), u.Quantity)
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: BurkertPotential, x: gt.Sz3) -> None:
        expect = u.Quantity(8.79860325e09, "solMass / kpc3")
        assert jnp.isclose(
            pot.density(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: BurkertPotential, x: gt.Sz3) -> None:
        expect = u.Quantity(
            [
                [0.46023037, -0.16060135, -0.24090202],
                [-0.16060135, 0.21932834, -0.48180405],
                [-0.24090202, -0.48180405, -0.18217503],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.hessian(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: AbstractBasePotential, x: gt.Sz3) -> None:
        """Test the `AbstractBasePotential.tidal_tensor` method."""
        expect = u.Quantity(
            [
                [0.29443581, -0.16060135, -0.24090202],
                [-0.16060135, 0.05353378, -0.48180405],
                [-0.24090202, -0.48180405, -0.34796959],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    # ==========================================================================
    # Interoperability

    @pytest.mark.skipif(not OptDeps.GALA.installed, reason="requires gala")
    @pytest.mark.parametrize(
        ("method0", "method1", "atol"),
        [
            ("potential", "energy", 1e-8),
            ("gradient", "gradient", 1e-8),
            ("density", "density", 1e-8),
            # ("hessian", "hessian", 1e-8),  # TODO: get gala and galax to agree
        ],
    )
    def test_method_gala(
        self,
        pot: gp.AbstractBasePotential,
        method0: str,
        method1: str,
        x: gt.QuSz3,
        atol: float,
    ) -> None:
        """Test the equivalence of methods between gala and galax.

        This test only runs if the potential can be mapped to gala.
        """
        super().test_method_gala(pot, method0, method1, x, atol)
