from typing import Any

import astropy.units as u
import pytest
from plum import convert

import quaxed.numpy as jnp
from unxt import AbstractUnitSystem, Quantity

import galax.potential as gp
import galax.typing as gt
from ...test_core import AbstractPotential_Test
from ..test_common import ParameterMTotMixin, ParameterShapeAMixin
from galax._interop.optional_deps import OptDeps
from galax.potential import AbstractPotentialBase, KuzminPotential


class TestKuzminPotential(
    AbstractPotential_Test,
    # Parameters
    ParameterMTotMixin,
    ParameterShapeAMixin,
):
    """Test the `galax.potential.KuzminPotential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.KuzminPotential]:
        return gp.KuzminPotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m_tot: u.Quantity,
        field_a: u.Quantity,
        field_units: AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {"m_tot": field_m_tot, "a": field_a, "units": field_units}

    # ==========================================================================

    def test_potential(self, pot: KuzminPotential, x: gt.QVec3) -> None:
        expect = Quantity(-0.98165365, unit="kpc2 / Myr2")
        assert jnp.isclose(
            pot.potential(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: KuzminPotential, x: gt.QVec3) -> None:
        expect = Quantity([0.04674541, 0.09349082, 0.18698165], "kpc / Myr2")
        got = convert(pot.gradient(x, t=0), Quantity)
        assert jnp.allclose(got, expect, atol=Quantity(1e-8, expect.unit))

    def test_density(self, pot: KuzminPotential, x: gt.QVec3) -> None:
        expect = Quantity(2.45494884e-07, "solMass / kpc3")
        assert jnp.isclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: KuzminPotential, x: gt.QVec3) -> None:
        expect = Quantity(
            [
                [0.0400675, -0.01335583, -0.02671166],
                [-0.01335583, 0.02003375, -0.05342333],
                [-0.02671166, -0.05342333, -0.06010124],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.hessian(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: AbstractPotentialBase, x: gt.QVec3) -> None:
        """Test the `AbstractPotentialBase.tidal_tensor` method."""
        expect = Quantity(
            [
                [0.0400675, -0.01335583, -0.02671166],
                [-0.01335583, 0.02003375, -0.05342333],
                [-0.02671166, -0.05342333, -0.06010124],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Interoperability

    @pytest.mark.skipif(not OptDeps.GALA.is_installed, reason="requires gala")
    @pytest.mark.parametrize(
        ("method0", "method1", "atol"),
        [
            ("potential", "energy", 1e-8),
            ("gradient", "gradient", 1e-8),
            ("density", "density", 5e-7),  # TODO: why is this different?
            # ("hessian", "hessian", 1e-8),  # TODO: why is gala's 0?
        ],
    )
    def test_method_gala(
        self, pot: KuzminPotential, method0: str, method1: str, x: gt.QVec3, atol: float
    ) -> None:
        super().test_method_gala(pot, method0, method1, x, atol)
