"""Tests for the `galax.potential.LM10Potential` class."""

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.potential.custom_types as gt
from .test_composite import AbstractSpecialCompositePotential_Test
from galax.interop.optional_deps import OptDeps


class TestLM10Potential(AbstractSpecialCompositePotential_Test):
    """Test the `galax.potential.LM10Potential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.LM10Potential]:
        return gp.LM10Potential

    @pytest.fixture(scope="class")
    def pot_map(
        self, pot_cls: type[gp.LM10Potential]
    ) -> dict[str, dict[str, gp.AbstractPotential]]:
        """Composite potential."""
        return {"disk": pot_cls.disk, "bulge": pot_cls.bulge, "halo": pot_cls.halo}

    # ==========================================================================

    def test_potential(self, pot: gp.LM10Potential, x: gt.QuSz3) -> None:
        expect = u.Q(-0.00242568, unit="kpc2 / Myr2")
        assert jnp.isclose(pot.potential(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/lm10", atol=1e-8
    )
    def test_gradient(self, pot: gp.LM10Potential, x: gt.QuSz3) -> None:
        return pot.gradient(x, t=0).ustrip(pot.units["acceleration"])

    def test_density(self, pot: gp.LM10Potential, x: gt.QuSz3) -> None:
        expect = u.Q(19085831.78310305, "solMass / kpc3")
        assert jnp.isclose(pot.density(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/lm10", atol=1e-8
    )
    def test_hessian(self, pot: gp.LM10Potential, x: gt.QuSz3) -> None:
        return pot.hessian(x, t=0).ustrip("1/Myr2")

    # ---------------------------------
    # Convenience methods

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/lm10", atol=1e-8
    )
    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        return pot.tidal_tensor(x, t=0).ustrip("1/Myr2")

    # ==========================================================================
    # Interoperability

    @pytest.mark.skipif(not OptDeps.GALA.installed, reason="requires gala")
    @pytest.mark.parametrize(
        ("method0", "method1", "atol"),
        [
            ("potential", "energy", 1e-8),
            ("gradient", "gradient", 1e-8),
            # ("density", "density", 1e-8),  # TODO: get gala and galax to agree
            # ("hessian", "hessian", 1e-8),  # TODO: get gala and galax to agree
        ],
    )
    def test_method_gala(
        self,
        pot: gp.AbstractPotential,
        method0: str,
        method1: str,
        x: gt.QuSz3,
        atol: float,
    ) -> None:
        """Test the equivalence of methods between gala and galax.

        This test only runs if the potential can be mapped to gala.
        """
        super().test_method_gala(pot, method0, method1, x, atol)
