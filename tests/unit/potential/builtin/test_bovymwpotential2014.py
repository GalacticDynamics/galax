"""Unit tests for the `galax.potential.BovyMWPotential2014` class."""

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.potential.custom_types as gt
from ..io.test_gala import parametrize_test_method_gala
from .test_composite import AbstractSpecialCompositePotential_Test
from galax.interop.optional_deps import GSL_ENABLED, OptDeps


class TestBovyMWPotential2014(AbstractSpecialCompositePotential_Test):
    """Test the `galax.potential.BovyMWPotential2014` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.BovyMWPotential2014]:
        return gp.BovyMWPotential2014

    @pytest.fixture(scope="class")
    def pot_map(
        self, pot_cls: type[gp.BovyMWPotential2014]
    ) -> dict[str, dict[str, u.Quantity]]:
        """Composite potential."""
        return {"disk": pot_cls.disk, "bulge": pot_cls.bulge, "halo": pot_cls.halo}

    # ==========================================================================

    def test_potential(self, pot: gp.BovyMWPotential2014, x: gt.QuSz3) -> None:
        expect = u.Q(-0.16359185, unit="kpc2 / Myr2")
        assert jnp.isclose(pot.potential(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/bovymwpotential2014", atol=1e-8
    )
    def test_gradient(self, pot: gp.BovyMWPotential2014, x: gt.QuSz3) -> None:
        return pot.gradient(x, t=0).ustrip(pot.units["acceleration"])

    def test_density(self, pot: gp.BovyMWPotential2014, x: gt.QuSz3) -> None:
        got = pot.density(x, t=0)
        exp = u.Q(0.024911277, "Msun / pc3")
        assert jnp.isclose(got, exp, atol=u.Q(1e-8, exp.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/bovymwpotential2014", atol=1e-8
    )
    def test_hessian(self, pot: gp.BovyMWPotential2014, x: gt.QuSz3) -> None:
        return pot.hessian(x, t=0).ustrip("1/Myr2")

    # ---------------------------------
    # Convenience methods

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/bovymwpotential2014", atol=1e-8
    )
    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        return pot.tidal_tensor(x, t=0).ustrip("1/Myr2")

    # ---------------------------------
    # Interoperability

    @pytest.mark.skipif(
        not OptDeps.GALA.installed or not GSL_ENABLED, reason="requires gala + GSL"
    )
    def test_galax_to_gala_to_galax_roundtrip(
        self, pot: gp.AbstractPotential, x: gt.QuSz3
    ) -> None:
        super().test_galax_to_gala_to_galax_roundtrip(pot, x)

    @pytest.mark.skipif(
        not OptDeps.GALA.installed or not GSL_ENABLED, reason="requires gala + GSL"
    )
    @parametrize_test_method_gala
    def test_method_gala(
        self,
        pot: gp.BovyMWPotential2014,
        method0: str,
        method1: str,
        x: gt.QuSz3,
        atol: float,
    ) -> None:
        super().test_method_gala(pot, method0, method1, x, atol)
