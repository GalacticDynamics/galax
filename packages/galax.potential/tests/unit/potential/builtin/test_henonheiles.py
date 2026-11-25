"""Test the `galax.potential.HenonHeilesPotential` class."""

from typing import Any, ClassVar

import numpy as np
import pytest
from unxts.parametric import ParametricQuantity

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.potential.custom_types as gt
from ..param.test_field import ParameterFieldMixin
from ..test_core import AbstractSinglePotential_Test


class ParameterCoeffMixin(ParameterFieldMixin):
    """Test the coeff parameter."""

    @pytest.fixture(scope="class")
    def field_coeff(self) -> u.Quantity["wavenumber"]:
        return u.Q(1.0, "1/kpc")

    # =====================================================

    def test_coeff_constant(self, pot_cls, fields):
        """Test the `coeff` parameter."""
        fields["coeff"] = 1 / u.Q(1.0, "kpc")
        pot = pot_cls(**fields)
        assert pot.coeff(t=0) == 1 / u.Q(1.0, "kpc")

    def test_coeff_userfunc(self, pot_cls, fields):
        """Test the `coeff` parameter."""

        def cos_coeff(t: u.Quantity["time"]) -> ParametricQuantity["wavenumber"]:
            return u.Q(10 * jnp.cos(t.ustrip("Myr")), "1/kpc")

        fields["coeff"] = cos_coeff
        pot = pot_cls(**fields)
        assert pot.coeff(t=u.Q(0, "Myr")) == u.Q(10, "1/kpc")


# ===================================================================


class ParameterTimeScaleMixin(ParameterFieldMixin):
    """Test the timescale parameter."""

    @pytest.fixture(scope="class")
    def field_timescale(self) -> u.Quantity["time"]:
        return u.Q(1.0, "Myr")

    # =====================================================

    def test_timescale_constant(self, pot_cls, fields):
        """Test the `timescale` parameter."""
        fields["timescale"] = u.Q(1.0, "Myr")
        pot = pot_cls(**fields)
        assert pot.timescale(t=0) == u.Q(1.0, "Myr")

    def test_timescale_userfunc(self, pot_cls, fields):
        """Test the `timescale` parameter."""

        def func(t: u.Quantity["time"]) -> ParametricQuantity["time"]:
            return u.Q.from_(t * 1.2, "Myr")

        fields["timescale"] = func
        pot = pot_cls(**fields)
        assert pot.timescale(t=1) == u.Q(1.2, "Myr")


#####################################################################


class TestHenonHeilesPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterCoeffMixin,
    ParameterTimeScaleMixin,
):
    """Test the `galax.potential.HenonHeilesPotential` class."""

    HAS_GALA_COUNTERPART: ClassVar[bool] = False
    # TODO: figure out how to convert this to Gala given the different parameters

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.HenonHeilesPotential]:
        return gp.HenonHeilesPotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_coeff: u.Quantity,
        field_timescale: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "coeff": field_coeff,
            "timescale": field_timescale,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(self, pot: gp.HenonHeilesPotential, x: gt.Sz3) -> None:
        got = pot.potential(x, t=0)
        exp = u.Q(1.83333333, unit="kpc2 / Myr2")
        assert jnp.isclose(got, exp, atol=u.Q(1e-8, exp.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/henonheiles", atol=1e-8
    )
    def test_gradient(self, pot: gp.HenonHeilesPotential, x: gt.Sz3) -> None:
        return pot.gradient(x, t=0).ustrip(pot.units["acceleration"])

    def test_density(self, pot: gp.HenonHeilesPotential, x: gt.Sz3) -> None:
        got = pot.density(x, t=0)
        exp = u.Q(3.53795414e10, "solMass / kpc3")
        assert jnp.isclose(got, exp, atol=u.Q(1e-8, exp.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/henonheiles", atol=1e-8
    )
    def test_hessian(self, pot: gp.HenonHeilesPotential, x: gt.Sz3) -> None:
        return pot.hessian(x, t=0).ustrip("1/Myr2")

    # ---------------------------------
    # Convenience methods

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/henonheiles", atol=1e-8
    )
    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.Sz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        return pot.tidal_tensor(x, t=0).ustrip("1/Myr2")


# ==============================================================================
# The module functions


def test_module_closed_forms() -> None:
    """`galax.potential.henonheiles` equals the derivatives written out by hand."""
    hh = gp.henonheiles
    lam, ts, G = 1.0, 1.5, 3.0
    p = {"coeff": lam, "timescale": ts, "G": G}
    x, y, z = 1.0, 2.0, 3.0
    xyz = jnp.array([x, y, z])

    phi = (0.5 * (x**2 + y**2) + lam * (x**2 * y - y**3 / 3)) / ts**2
    np.testing.assert_allclose(hh.potential(p, xyz), phi)

    grad = np.array([x + 2 * lam * x * y, y + lam * (x**2 - y**2), 0.0]) / ts**2
    np.testing.assert_allclose(hh.gradient(p, xyz), grad)

    hess = np.array(
        [
            [1 + 2 * lam * y, 2 * lam * x, 0.0],
            [2 * lam * x, 1 - 2 * lam * y, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(hh.hessian(p, xyz), hess / ts**2, atol=1e-15)

    # The cubic terms are harmonic, so the Laplacian and the density are uniform.
    np.testing.assert_allclose(hh.laplacian(p, xyz), 2 / ts**2)
    np.testing.assert_allclose(hh.density(p, xyz), 2 / (4 * np.pi * G * ts**2))
    assert hh.density(p, jnp.ones((4, 3))).shape == (4,)


def test_laplacian_does_not_need_G() -> None:
    """The Laplacian is ``2 / timescale**2`` with or without ``G`` in ``constants``."""
    hh = gp.henonheiles
    p = {"coeff": 1.0, "timescale": 1.5}  # no "G"
    np.testing.assert_allclose(hh.laplacian(p, jnp.array([1.0, 2.0, 3.0])), 2 / 1.5**2)

    pot = gp.HenonHeilesPotential(
        coeff=u.Q(1.0, "1/kpc"),
        timescale=u.Q(1.5, "Myr"),
        units="galactic",
        constants={},
    )
    q, t = u.Q([1.0, 2.0, 3.0], "kpc"), u.Q(0.0, "Gyr")
    np.testing.assert_allclose(pot.laplacian(q, t).ustrip("1/Myr2"), 2 / 1.5**2)
