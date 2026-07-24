"""Test the `galax.potential.TriaxialGaussianPotential` class."""

from typing import Any, ClassVar

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
import galax.potential as gp
from ..test_core import AbstractSinglePotential_Test
from .test_common import (
    ParameterMMixin,
    ParameterRSMixin,
    ParameterShapeQ1Mixin,
    ParameterShapeQ2Mixin,
    assert_gaussian_matches_galpy,
)
from galax._interop.optional_deps import OptDeps


class TestTriaxialGaussianPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMMixin,
    ParameterRSMixin,
    ParameterShapeQ1Mixin,
    ParameterShapeQ2Mixin,
):
    """Test the `galax.potential.TriaxialGaussianPotential` class."""

    HAS_GALA_COUNTERPART: ClassVar[bool] = False

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.TriaxialGaussianPotential]:
        return gp.TriaxialGaussianPotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m: u.Quantity,
        field_r_s: u.Quantity,
        field_q1: u.Quantity,
        field_q2: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m": field_m,
            "r_s": field_r_s,
            "q1": field_q1,
            "q2": field_q2,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(self, pot: gp.TriaxialGaussianPotential, x: gt.QuSz3) -> None:
        expect = u.Q(-0.64255935, unit="kpc2 / Myr2")
        got = pot.potential(x, t=0)
        assert jnp.isclose(got, expect, atol=u.Q(1e-8, expect.unit))

    def test_gradient(self, pot: gp.TriaxialGaussianPotential, x: gt.QuSz3) -> None:
        expect = u.Q([0.03956623, 0.0765244, 0.13704284], "kpc / Myr2")
        got = pot.gradient(x, t=0)
        assert jnp.allclose(got, expect, atol=u.Q(1e-8, expect.unit))

    def test_density(self, pot: gp.TriaxialGaussianPotential, x: gt.QuSz3) -> None:
        expect = u.Q(112.31556702, "solMass / kpc3")
        assert jnp.isclose(pot.density(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    def test_hessian(self, pot: gp.TriaxialGaussianPotential, x: gt.QuSz3) -> None:
        expect = u.Q(
            [
                [0.03303733, -0.0124193, -0.02445886],
                [-0.0124193, 0.0146227, -0.04636713],
                [-0.02445886, -0.04636713, -0.04766002],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(pot.hessian(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        expect = u.Q(
            [
                [0.03303733, -0.0124193, -0.02445886],
                [-0.0124193, 0.01462269, -0.04636713],
                [-0.02445886, -0.04636713, -0.04766002],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Q(1e-8, expect.unit)
        )

    # ---------------------------------
    # Interoperability

    @pytest.mark.skipif(not OptDeps.GALPY.installed, reason="requires galpy")
    def test_method_galpy(self, pot: gp.TriaxialGaussianPotential, x: gt.QuSz3) -> None:
        """Test the equivalence of potential/density between galpy and galax."""
        assert_gaussian_matches_galpy(pot, x)
