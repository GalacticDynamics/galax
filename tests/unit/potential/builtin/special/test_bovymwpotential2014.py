"""Unit tests for the `galax.potential.BovyMWPotential2014` class."""

import pytest
from plum import convert

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.typing as gt
from ...io.test_gala import parametrize_test_method_gala
from .test_composite import AbstractSpecialCompositePotential_Test
from galax._interop.optional_deps import GSL_ENABLED, OptDeps


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
        expect = u.Quantity(-0.09550731, unit="kpc2 / Myr2")
        assert jnp.isclose(
            pot.potential(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: gp.BovyMWPotential2014, x: gt.QuSz3) -> None:
        expect = u.Quantity([0.00231875, 0.0046375, 0.01042675], "kpc / Myr2")
        got = convert(pot.gradient(x, t=0), u.Quantity)
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: gp.BovyMWPotential2014, x: gt.QuSz3) -> None:
        expect = u.Quantity(24_911_277.33877818, "solMass / kpc3")
        assert jnp.isclose(
            pot.density(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: gp.BovyMWPotential2014, x: gt.QuSz3) -> None:
        expect = u.Quantity(
            [
                [0.00208414, -0.00046922, -0.0009568],
                [-0.00046922, 0.00138031, -0.00191361],
                [-0.0009568, -0.00191361, -0.00205622],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.hessian(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        expect = u.Quantity(
            [
                [0.00161473, -0.00046922, -0.0009568],
                [-0.00046922, 0.0009109, -0.00191361],
                [-0.0009568, -0.00191361, -0.00252563],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

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
