"""Test the `TriaxialHernquistPotential` class."""

from typing import Any, ClassVar

import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.potential.custom_types as gt
from ..test_core import AbstractSinglePotential_Test
from .test_common import (
    ParameterMTotMixin,
    ParameterRSMixin,
    ParameterShapeQ1Mixin,
    ParameterShapeQ2Mixin,
)


class TestTriaxialHernquistPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMTotMixin,
    ParameterRSMixin,
    ParameterShapeQ1Mixin,
    ParameterShapeQ2Mixin,
):
    HAS_GALA_COUNTERPART: ClassVar[bool] = False

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.TriaxialHernquistPotential]:
        return gp.TriaxialHernquistPotential

    @pytest.fixture(scope="class")
    def fields_(
        self, field_m_tot, field_r_s, field_q1, field_q2, field_units
    ) -> dict[str, Any]:
        return {
            "m_tot": field_m_tot,
            "r_s": field_r_s,
            "q1": field_q1,
            "q2": field_q2,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(self, pot: gp.TriaxialHernquistPotential, x: gt.QuSz3) -> None:
        expect = u.Q(-0.61215074, pot.units["specific energy"])
        assert jnp.isclose(pot.potential(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/triaxialhernquist", atol=1e-8
    )
    def test_gradient(self, pot: gp.TriaxialHernquistPotential, x: gt.QuSz3) -> None:
        return pot.gradient(x, t=0).ustrip(pot.units["acceleration"])

    @pytest.mark.xfail(reason="WFF?")
    def test_density(self, pot: gp.TriaxialHernquistPotential, x: gt.QuSz3) -> None:
        assert pot.density(x, t=0).decompose(pot.units).value >= 0

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/triaxialhernquist", atol=1e-8
    )
    def test_hessian(self, pot: gp.TriaxialHernquistPotential, x: gt.QuSz3) -> None:
        return pot.hessian(x, t=0).ustrip("1/Myr2")

    # ---------------------------------
    # Convenience methods

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/triaxialhernquist", atol=1e-8
    )
    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        return pot.tidal_tensor(x, t=0).ustrip("1/Myr2")


# ==============================================================================
# The module functions


def test_isotropic_limit_is_hernquist() -> None:
    """With ``q1 = q2 = 1`` the potential is the Hernquist potential of the radius."""
    kw = {"m_tot": u.Q(1e12, "Msun"), "r_s": u.Q(8.0, "kpc"), "units": "galactic"}
    pot = gp.TriaxialHernquistPotential(q1=1.0, q2=1.0, **kw)
    p = pot.params(0.0)
    xyz = jnp.array([1.0, 2.0, 3.0])

    np.testing.assert_allclose(
        gp.triaxialhernquist.potential(p, xyz),
        gp.hernquist.potential(p, jnp.linalg.norm(xyz)),
        rtol=1e-14,
    )


def test_density_is_laplacian_over_4piG() -> None:
    """The class density equals the trace of its Hessian over ``4 pi G``."""
    pot = gp.TriaxialHernquistPotential(
        m_tot=u.Q(1e12, "Msun"), r_s=u.Q(8.0, "kpc"), q1=0.9, q2=0.7, units="galactic"
    )
    xyz = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.3, 0.2, 0.1]])
    four_pi_G = 4 * np.pi * pot.params(0.0)["G"]

    np.testing.assert_allclose(
        pot.density(xyz, 0.0),
        jnp.trace(pot.hessian(xyz, 0.0), axis1=-2, axis2=-1) / four_pi_G,
        rtol=1e-12,
    )
