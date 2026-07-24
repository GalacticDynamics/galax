from typing import Any, ClassVar
from typing_extensions import override

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
import galax.potential as gp
from ..test_core import AbstractSinglePotential_Test
from .test_common import (
    ParameterMMixin,
    ParameterRSMixin,
    assert_gaussian_matches_galpy,
)
from galax._interop.optional_deps import OptDeps

###############################################################################


class TestGaussianPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMMixin,
    ParameterRSMixin,
):
    HAS_GALA_COUNTERPART: ClassVar[bool] = False

    @pytest.fixture(scope="class")
    @override
    def pot_cls(self) -> type[gp.GaussianPotential]:
        return gp.GaussianPotential

    @pytest.fixture(scope="class")
    @override
    def fields_(
        self,
        field_m: u.Quantity,
        field_r_s: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {"m": field_m, "r_s": field_r_s, "units": field_units}

    # ==========================================================================

    def test_potential(self, pot: gp.GaussianPotential, x: gt.QuSz3) -> None:
        expect = u.Q(-1.20205545, pot.units["specific energy"])
        assert jnp.isclose(pot.potential(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    def test_gradient(self, pot: gp.GaussianPotential, x: gt.QuSz3) -> None:
        expect = u.Q([0.08562732, 0.17125463, 0.25688195], pot.units["acceleration"])
        got = pot.gradient(x, t=0)
        assert jnp.allclose(got, expect, atol=u.Q(1e-8, expect.unit))

    def test_density(self, pot: gp.GaussianPotential, x: gt.QuSz3) -> None:
        got = pot.density(x, t=0)
        exp = u.Q(5.78986720e07, pot.units["mass density"])
        assert jnp.isclose(got, exp, atol=u.Q(1e-8, exp.unit))

    def test_hessian(self, pot: gp.GaussianPotential, x: gt.QuSz3) -> None:
        expect = u.Q(
            [
                [0.06751239, -0.03622985, -0.05434476],
                [-0.03622985, 0.01316763, -0.10868952],
                [-0.05434477, -0.10868953, -0.07740697],
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
                [0.06642137, -0.03622985, -0.05434476],
                [-0.03622985, 0.01207661, -0.10868952],
                [-0.05434477, -0.10868953, -0.07849799],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Q(1e-8, expect.unit)
        )

    # ---------------------------------
    # Interoperability

    @pytest.mark.skipif(not OptDeps.GALPY.installed, reason="requires galpy")
    def test_method_galpy(self, pot: gp.GaussianPotential, x: gt.QuSz3) -> None:
        """Test the equivalence of potential/density between galpy and galax."""
        assert_gaussian_matches_galpy(pot, x)
