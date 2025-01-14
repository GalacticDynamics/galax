"""Unit tests for the `HarmonicOscillatorPotential` class."""

from typing import Any
from typing_extensions import override

import pytest
from plum import convert

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.typing as gt
from ...param.test_field import ParameterFieldMixin
from ...test_core import AbstractPotential_Test
from galax._interop.optional_deps import OptDeps
from galax.potential._src.base import AbstractBasePotential


class ParameterOmegaMixin(ParameterFieldMixin):
    """Test the omega parameter."""

    @pytest.fixture(scope="class")
    def field_omega(self) -> u.Quantity["frequency"]:
        return u.Quantity(1.0, "Hz")

    # =====================================================

    def test_omega_constant(self, pot_cls, fields):
        """Test the `omega` parameter."""
        fields["omega"] = u.Quantity(1.0, "Hz")
        pot = pot_cls(**fields)
        assert pot.omega(t=0) == u.Quantity(1.0, "Hz")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_omega_userfunc(self, pot_cls, fields):
        """Test the `omega` parameter."""
        fields["omega"] = lambda t: t * 1.2
        pot = pot_cls(**fields)
        assert pot.omega(t=0) == 2


################################################################################


class TestHarmonicOscillatorPotential(
    AbstractPotential_Test,
    # Parameters
    ParameterOmegaMixin,
):
    @pytest.fixture(scope="class")
    @override
    def pot_cls(self) -> type[gp.HarmonicOscillatorPotential]:
        return gp.HarmonicOscillatorPotential

    @pytest.fixture(scope="class")
    @override
    def fields_(self, field_omega, field_units) -> dict[str, Any]:
        return {"omega": field_omega, "units": field_units}

    # ==========================================================================

    def test_potential(self, pot: gp.HarmonicOscillatorPotential, x: gt.QuSz3) -> None:
        got = pot.potential(x, t=0)
        expect = u.Quantity(6.97117482e27, pot.units["specific energy"])
        assert jnp.isclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_gradient(self, pot: gp.HarmonicOscillatorPotential, x: gt.Sz3) -> None:
        got = convert(pot.gradient(x, t=0), u.Quantity)
        expect = u.Quantity([9.95882118e26, 1.99176424e27, 2.98764635e27], "kpc / Myr2")
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: gp.HarmonicOscillatorPotential, x: gt.QuSz3) -> None:
        got = pot.density(x, t=0)
        expect = u.Quantity(1.76169263e37, unit="solMass / kpc3")
        assert jnp.isclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_hessian(self, pot: gp.HarmonicOscillatorPotential, x: gt.QuSz3) -> None:
        got = pot.hessian(x, t=0)
        expect = u.Quantity(
            [
                [9.95882118e26, 0.00000000e00, 0.00000000e00],
                [0.00000000e00, 9.95882118e26, 0.00000000e00],
                [0.00000000e00, 0.00000000e00, 9.95882118e26],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: AbstractBasePotential, x: gt.Sz3) -> None:
        """Test the `AbstractBasePotential.tidal_tensor` method."""
        expect = u.Quantity(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Interoperability

    @pytest.mark.skipif(not OptDeps.GALA.installed, reason="requires gala")
    @pytest.mark.parametrize(
        ("method0", "method1", "atol"),
        [
            ("potential", "energy", 1e-8),
            ("gradient", "gradient", 1e-8),
            # ("density", "density", 5e-7),  # Gala doesn't have the density
            # ("hessian", "hessian", 1e-8),  # TODO: why doesn't this match?
        ],
    )
    def test_method_gala(
        self,
        pot: gp.HarmonicOscillatorPotential,
        method0: str,
        method1: str,
        x: gt.QuSz3,
        atol: float,
    ) -> None:
        """Test the equivalence of methods between gala and galax.

        This test only runs if the potential can be mapped to gala.
        """
        # First we need to check that the potential is gala-compatible
        if not self.HAS_GALA_COUNTERPART:
            pytest.skip("potential does not have a gala counterpart")

        # Evaluate the galax method. Gala is in 1D, so we take the norm.
        galax = convert(getattr(pot, method0)(x, t=0), u.Quantity)
        galax1d = jnp.linalg.vector_norm(jnp.atleast_1d(galax), axis=-1)

        # Evaluate the gala method. This works in 1D on Astropy quantities.
        galap = gp.io.convert_potential(gp.io.GalaLibrary, pot)
        r = convert(jnp.linalg.vector_norm(x, axis=-1), u.Quantity)
        gala = getattr(galap, method1)(r, t=0)

        assert jnp.allclose(
            jnp.ravel(galax1d),
            jnp.ravel(convert(gala, u.Quantity)),
            atol=u.Quantity(atol, galax.unit),
        )

    # ==========================================================================
    # TODO: Implement these tests

    @pytest.mark.skip("TODO")
    def test_evaluate_orbit(self, pot: gp.AbstractBasePotential, xv: gt.Sz6) -> None:
        """Test the `AbstractBasePotential.evaluate_orbit` method."""

    @pytest.mark.skip("TODO")
    def test_evaluate_orbit_batch(
        self, pot: gp.AbstractBasePotential, xv: gt.Sz6
    ) -> None:
        """Test the `AbstractBasePotential.evaluate_orbit` method."""
