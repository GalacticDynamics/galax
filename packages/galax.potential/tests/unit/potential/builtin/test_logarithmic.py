from typing import Any

import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.potential.custom_types as gt
from ..test_core import AbstractSinglePotential_Test
from .test_common import ParameterRSMixin, ParameterVCMixin


class TestLogarithmicPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterVCMixin,
    ParameterRSMixin,
):
    """Test the `galax.potential.LogarithmicPotential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.LogarithmicPotential]:
        return gp.LogarithmicPotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_v_c: u.Quantity,
        field_r_s: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {"v_c": field_v_c, "r_s": field_r_s, "units": field_units}

    # ==========================================================================

    def test_potential(self, pot: gp.LogarithmicPotential, x: gt.QuSz3) -> None:
        exp = u.Q(0.0685455, unit="kpc2 / Myr2")
        got = pot.potential(x, t=0)
        assert jnp.isclose(got, exp, atol=u.Q(1e-8, exp.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/logarithmic", atol=1e-8
    )
    def test_gradient(self, pot: gp.LogarithmicPotential, x: gt.QuSz3) -> None:
        return pot.gradient(x, t=0).ustrip(pot.units["acceleration"])

    def test_density(self, pot: gp.LogarithmicPotential, x: gt.QuSz3) -> None:
        exp = u.Q(67661373.89566506, "solMass / kpc3")
        got = pot.density(x, t=0)
        assert jnp.isclose(got, exp, atol=u.Q(1e-8, exp.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/logarithmic", atol=1e-8
    )
    def test_hessian(self, pot: gp.LogarithmicPotential, x: gt.QuSz3) -> None:
        return pot.hessian(x, t=0).ustrip("1/Myr2")

    # ---------------------------------
    # Convenience methods

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/logarithmic", atol=1e-8
    )
    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        return pot.tidal_tensor(x, t=0).ustrip("1/Myr2")


# ==============================================================================
# The module functions


def test_module_closed_forms() -> None:
    """`galax.potential.logarithmic` equals the closed forms at a few radii."""
    pot = gp.LogarithmicPotential(
        v_c=u.Q(220.0, "km/s"), r_s=u.Q(5.0, "kpc"), units="galactic"
    )
    p = pot.params(0.0)
    v2, rs2, G = p["v_c"] ** 2, p["r_s"] ** 2, p["G"]

    for r in (0.5, 5.0, 30.0):
        np.testing.assert_allclose(
            gp.logarithmic.mass_enclosed(p, r), v2 * r**3 / (G * (rs2 + r**2))
        )
        np.testing.assert_allclose(
            gp.logarithmic.dpotential_dr(p, r), v2 * r / (rs2 + r**2)
        )

    # The circular speed tends to v_c far out.
    r = 1e4 * p["r_s"]
    vc = jnp.sqrt(r * gp.logarithmic.dpotential_dr(p, r))
    np.testing.assert_allclose(vc, p["v_c"], rtol=1e-7)
