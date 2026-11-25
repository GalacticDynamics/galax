"""Unit tests for the `HarmonicOscillatorPotential` class."""

from typing import Any, override

import numpy as np
import pytest
from plum import convert
from unxts.parametric import ParametricQuantity

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.potential.custom_types as gt
from ..optional_deps import OptDeps
from ..param.test_field import ParameterFieldMixin
from ..test_core import AbstractSinglePotential_Test
from galax.potential._src.base import AbstractPotential


class ParameterOmegaMixin(ParameterFieldMixin):
    """Test the omega parameter."""

    @pytest.fixture(scope="class")
    def field_omega(self) -> u.Quantity["frequency"]:
        return u.Q(1.0, "Hz")

    # =====================================================

    def test_omega_constant(self, pot_cls, fields):
        """Test the `omega` parameter."""
        fields["omega"] = u.Q(1.0, "Hz")
        pot = pot_cls(**fields)
        assert pot.omega(t=0) == u.Q(1.0, "Hz")

    def test_omega_userfunc(self, pot_cls, fields):
        """Test the `omega` parameter."""

        def cos_omega(t: u.Quantity["time"]) -> ParametricQuantity["frequency"]:
            return u.Q(10 * jnp.cos(t.ustrip("Myr")), "Hz")

        fields["omega"] = cos_omega
        pot = pot_cls(**fields)
        assert pot.omega(t=u.Q(0, "Myr")) == u.Q(10, "Hz")


################################################################################


class TestHarmonicOscillatorPotential(
    AbstractSinglePotential_Test,
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
        expect = u.Q(6.97117482e27, pot.units["specific energy"])
        assert jnp.isclose(got, expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/harmonicoscillator", atol=1e-8
    )
    def test_gradient(self, pot: gp.HarmonicOscillatorPotential, x: gt.Sz3) -> None:
        return pot.gradient(x, t=0).ustrip(pot.units["acceleration"])

    def test_density(self, pot: gp.HarmonicOscillatorPotential, x: gt.QuSz3) -> None:
        got = pot.density(x, t=0)
        expect = u.Q(5.28507789e37, unit="solMass / kpc3")
        assert jnp.isclose(got, expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/harmonicoscillator", atol=1e-8
    )
    def test_hessian(self, pot: gp.HarmonicOscillatorPotential, x: gt.QuSz3) -> None:
        return pot.hessian(x, t=0).ustrip("1/Myr2")

    # ---------------------------------
    # Convenience methods

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/harmonicoscillator", atol=1e-8
    )
    def test_tidal_tensor(self, pot: AbstractPotential, x: gt.Sz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        return pot.tidal_tensor(x, t=0).ustrip("1/Myr2")

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
        r = jnp.linalg.vector_norm(x, axis=-1)
        gala = getattr(galap, method1)(r, t=0)

        assert jnp.allclose(
            jnp.ravel(galax1d),
            jnp.ravel(convert(gala, u.Quantity)),
            atol=u.Q(atol, galax.unit),
        )

    # ==========================================================================
    # TODO: Implement these tests

    @pytest.mark.skip("TODO")
    def test_evaluate_orbit(self, pot: gp.AbstractPotential, xv: gt.Sz6) -> None:
        """Test the `AbstractPotential.evaluate_orbit` method."""

    @pytest.mark.skip("TODO")
    def test_evaluate_orbit_batch(self, pot: gp.AbstractPotential, xv: gt.Sz6) -> None:
        """Test the `AbstractPotential.evaluate_orbit` method."""


################################################################################


class TestHarmonicOscillatorVectorOmega:
    """A 3-vector omega gives an anisotropic oscillator."""

    @pytest.fixture(scope="class")
    def pot(self) -> gp.HarmonicOscillatorPotential:
        return gp.HarmonicOscillatorPotential(
            omega=u.Q([1.0, 2.0, 3.0], "1/Myr"), units="galactic"
        )

    @pytest.fixture(scope="class")
    def q(self) -> u.Q:
        return u.Q([1.0, 1.0, 1.0], "kpc")

    @pytest.fixture(scope="class")
    def t(self) -> u.Q:
        return u.Q(0.0, "Myr")

    def test_omega_shape(self, pot, t) -> None:
        assert pot.omega(t).shape == (3,)

    def test_potential(self, pot, q, t) -> None:
        assert jnp.allclose(pot.potential(q, t).ustrip("kpc2/Myr2"), 7.0)  # 1/2 (1+4+9)

    def test_gradient(self, pot, q, t) -> None:
        got = pot.gradient(q, t).ustrip("kpc/Myr2")
        assert jnp.allclose(got, jnp.asarray([1.0, 4.0, 9.0]))

    def test_hessian(self, pot, q, t) -> None:
        got = pot.hessian(q, t).ustrip("1/Myr2")
        assert jnp.allclose(got, jnp.diag(jnp.asarray([1.0, 4.0, 9.0])))

    def test_density(self, pot, q, t) -> None:
        G = pot.constants["G"].ustrip("kpc3/(Msun Myr2)")
        got = pot.density(q, t).ustrip("Msun/kpc3")
        assert jnp.allclose(got, 14.0 / (4 * jnp.pi * G))

    def test_batch(self, pot, t) -> None:
        qs = u.Q(jnp.ones((5, 3)), "kpc")
        assert pot.potential(qs, t).shape == (5,)
        assert pot.density(qs, t).shape == (5,)

    def test_scalar_omega_unchanged(self, t) -> None:
        pot = gp.HarmonicOscillatorPotential(omega=u.Q(2.0, "1/Myr"), units="galactic")
        G = pot.constants["G"].ustrip("kpc3/(Msun Myr2)")
        cases = [
            ([1.0, 0.0, 0.0], 2.0),
            ([1.0, 2.0, 0.0], 10.0),
            ([1.0, 2.0, 3.0], 28.0),
        ]
        for xyz, phi in cases:
            q = u.Q(xyz, "kpc")
            assert jnp.allclose(pot.potential(q, t).ustrip("kpc2/Myr2"), phi)
            assert jnp.allclose(
                pot.density(q, t).ustrip("Msun/kpc3"), 3 * 4.0 / (4 * jnp.pi * G)
            )

    def test_vector_omega_time_dependent(self, q) -> None:
        def omega(t: u.Quantity["time"]) -> ParametricQuantity["frequency"]:
            return u.Q(jnp.asarray([1.0, 2.0, 3.0]) * (1 + t.ustrip("Myr")), "1/Myr")

        pot = gp.HarmonicOscillatorPotential(omega=omega, units="galactic")
        t = u.Q(1.0, "Myr")
        assert pot.omega(t).shape == (3,)
        assert jnp.allclose(pot.potential(q, t).ustrip("kpc2/Myr2"), 28.0)


# ==============================================================================
# The module functions


def test_module_closed_forms() -> None:
    """`galax.potential.harmonicoscillator` equals the closed forms."""
    ho = gp.harmonicoscillator
    xyz = jnp.array([1.0, 1.0, 1.0])

    # A vector omega: one frequency per axis.
    p = {"omega": jnp.array([1.0, 2.0, 3.0]), "G": 2.0}
    assert ho.potential(p, xyz) == 7.0
    np.testing.assert_array_equal(ho.gradient(p, xyz), [1.0, 4.0, 9.0])
    np.testing.assert_array_equal(ho.hessian(p, xyz), np.diag([1.0, 4.0, 9.0]))
    np.testing.assert_allclose(ho.laplacian(p, xyz), 14.0)
    np.testing.assert_allclose(ho.density(p, xyz), 14.0 / (4 * np.pi * 2.0))

    # A scalar omega is shared by the three axes.
    p = {"omega": jnp.asarray(2.0), "G": 2.0}
    assert ho.potential(p, xyz) == 6.0
    np.testing.assert_array_equal(ho.gradient(p, xyz), [4.0, 4.0, 4.0])
    np.testing.assert_allclose(ho.laplacian(p, xyz), 12.0)
    np.testing.assert_allclose(ho.density(p, xyz), 12.0 / (4 * np.pi * 2.0))


def test_density_without_G_names_the_constant() -> None:
    """Without ``G`` in ``constants`` the density fails clearly; the Laplacian works."""
    pot = gp.HarmonicOscillatorPotential(
        omega=u.Q(1.0, "1/Myr"), units="galactic", constants={}
    )
    q, t = u.Q([1.0, 0.0, 0.0], "kpc"), u.Q(0.0, "Gyr")

    with pytest.raises(ValueError, match="'G'"):
        pot.density(q, t)
    np.testing.assert_allclose(pot.laplacian(q, t).ustrip("1/Myr2"), 3.0)
