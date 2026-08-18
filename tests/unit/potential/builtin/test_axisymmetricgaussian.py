"""Test the `galax.potential.AxisymmetricGaussianPotential` class."""

from typing import Any, ClassVar

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.potential.custom_types as gt
from ..test_core import AbstractSinglePotential_Test
from .test_common import (
    ParameterMTotMixin,
    ParameterRSMixin,
    ParameterShapeQ2Mixin,
    assert_gaussian_matches_galpy,
)
from galax.interop.optional_deps import OptDeps


class TestAxisymmetricGaussianPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMTotMixin,
    ParameterRSMixin,
    ParameterShapeQ2Mixin,
):
    """Test the `galax.potential.AxisymmetricGaussianPotential` class."""

    HAS_GALA_COUNTERPART: ClassVar[bool] = False

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.AxisymmetricGaussianPotential]:
        return gp.AxisymmetricGaussianPotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m_tot: u.Quantity,
        field_r_s: u.Quantity,
        field_q2: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m_tot": field_m_tot,
            "r_s": field_r_s,
            "q2": field_q2,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(
        self, pot: gp.AxisymmetricGaussianPotential, x: gt.QuSz3
    ) -> None:
        expect = u.Q(-1.17098953, unit="kpc2 / Myr2")
        got = pot.potential(x, t=0)
        assert jnp.isclose(got, expect, atol=u.Q(1e-8, expect.unit))

    def test_gradient(self, pot: gp.AxisymmetricGaussianPotential, x: gt.QuSz3) -> None:
        expect = u.Q([0.07199738, 0.14399475, 0.24909689], "kpc / Myr2")
        got = pot.gradient(x, t=0)
        assert jnp.allclose(got, expect, atol=u.Q(1e-8, expect.unit))

    def test_density(self, pot: gp.AxisymmetricGaussianPotential, x: gt.QuSz3) -> None:
        expect = u.Q(158.75350192, "solMass / kpc3")
        assert jnp.isclose(pot.density(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    def test_hessian(self, pot: gp.AxisymmetricGaussianPotential, x: gt.QuSz3) -> None:
        expect = u.Q(
            [
                [0.06018828, -0.02361818, -0.04413969],
                [-0.02361818, 0.02476101, -0.08827938],
                [-0.04413969, -0.08827938, -0.08494928],
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
                [0.06018828, -0.02361818, -0.04413969],
                [-0.02361818, 0.02476101, -0.08827938],
                [-0.04413969, -0.08827938, -0.08494929],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Q(1e-8, expect.unit)
        )

    # ---------------------------------
    # Interoperability

    @pytest.mark.skipif(not OptDeps.GALPY.installed, reason="requires galpy")
    def test_method_galpy(
        self, pot: gp.AxisymmetricGaussianPotential, x: gt.QuSz3
    ) -> None:
        """Test the equivalence of potential/density between galpy and galax."""
        assert_gaussian_matches_galpy(pot, x)
