from typing import Any

import jax.numpy as xp
import pytest

import galdynamix.potential as gp

from .test_core import TestAbstractPotential


class TestMiyamotoNagaiPotential(TestAbstractPotential):
    """Test the `galdynamix.potential.MiyamotoNagaiPotential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.MiyamotoNagaiPotential]:
        return gp.MiyamotoNagaiPotential

    @pytest.fixture(scope="class")
    def field_m(self) -> dict[str, Any]:
        return 1e12

    @pytest.fixture(scope="class")
    def field_a(self) -> dict[str, Any]:
        return 1

    @pytest.fixture(scope="class")
    def field_b(self) -> dict[str, Any]:
        return 1

    @pytest.fixture(scope="class")
    def fields_(self, field_m, field_a, field_b, field_units) -> dict[str, Any]:
        return {"m": field_m, "a": field_a, "b": field_b, "units": field_units}

    # ==========================================================================

    def test_potential_energy(self, pot, x) -> None:
        assert xp.isclose(pot.potential_energy(x, t=0), xp.array(-0.95208676))

    def test_gradient(self, pot, x):
        assert xp.allclose(
            pot.gradient(x, t=0), xp.array([0.04264751, 0.08529503, 0.16840152])
        )

    def test_density(self, pot, x):
        assert xp.isclose(pot.density(x, t=0), 1.9949418e08)

    def test_hessian(self, pot, x):
        assert xp.allclose(
            pot.hessian(x, t=0),
            xp.array(
                [
                    [0.03691649, -0.01146205, -0.02262999],
                    [-0.01146205, 0.01972342, -0.04525999],
                    [-0.02262999, -0.04525999, -0.04536254],
                ]
            ),
        )

    def test_acceleration(self, pot, x):
        assert xp.allclose(pot.acceleration(x, t=0), -pot.gradient(x, t=0))


# ==========================================================================


class TestBarPotential(TestAbstractPotential):
    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.BarPotential]:
        return gp.BarPotential

    @pytest.fixture(scope="class")
    def field_m(self) -> dict[str, Any]:
        return 1e12

    @pytest.fixture(scope="class")
    def field_a(self) -> dict[str, Any]:
        return 1

    @pytest.fixture(scope="class")
    def field_b(self) -> dict[str, Any]:
        return 1

    @pytest.fixture(scope="class")
    def field_c(self) -> dict[str, Any]:
        return 1

    @pytest.fixture(scope="class")
    def field_Omega(self) -> dict[str, Any]:
        return 0

    @pytest.fixture(scope="class")
    def fields_(
        self, field_m, field_a, field_b, field_c, field_Omega, field_units
    ) -> dict[str, Any]:
        return {
            "m": field_m,
            "a": field_a,
            "b": field_b,
            "c": field_c,
            "Omega": field_Omega,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential_energy(self, pot, x) -> None:
        assert xp.isclose(pot.potential_energy(x, t=0), xp.array(-0.94601574))

    def test_gradient(self, pot, x):
        assert xp.allclose(
            pot.gradient(x, t=0), xp.array([0.04011905, 0.08383918, 0.16552719])
        )

    def test_density(self, pot, x):
        assert xp.isclose(pot.density(x, t=0), 1.94669274e08)

    def test_hessian(self, pot, x):
        assert xp.allclose(
            pot.hessian(x, t=0),
            xp.array(
                [
                    [0.03529841, -0.01038389, -0.02050134],
                    [-0.01038389, 0.0195721, -0.04412159],
                    [-0.02050134, -0.04412159, -0.04386589],
                ]
            ),
        )

    def test_acceleration(self, pot, x):
        assert xp.allclose(pot.acceleration(x, t=0), -pot.gradient(x, t=0))


# ==========================================================================


class TestIsochronePotential(TestAbstractPotential):
    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.IsochronePotential]:
        return gp.IsochronePotential

    @pytest.fixture(scope="class")
    def field_m(self) -> dict[str, Any]:
        return 1e12

    @pytest.fixture(scope="class")
    def field_a(self) -> dict[str, Any]:
        return 1

    @pytest.fixture(scope="class")
    def fields_(self, field_m, field_a, field_units) -> dict[str, Any]:
        return {
            "m": field_m,
            "a": field_a,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential_energy(self, pot, x) -> None:
        assert xp.isclose(pot.potential_energy(x, t=0), xp.array(-0.9231515))

    def test_gradient(self, pot, x):
        assert xp.allclose(
            pot.gradient(x, t=0), xp.array([0.04891392, 0.09782784, 0.14674175])
        )

    def test_density(self, pot, x):
        assert xp.isclose(pot.density(x, t=0), 5.04511665e08)

    def test_hessian(self, pot, x):
        assert xp.allclose(
            pot.hessian(x, t=0),
            xp.array(
                [
                    [0.0404695, -0.01688883, -0.02533324],
                    [-0.01688883, 0.01513626, -0.05066648],
                    [-0.02533324, -0.05066648, -0.0270858],
                ]
            ),
        )

    def test_acceleration(self, pot, x):
        assert xp.allclose(pot.acceleration(x, t=0), -pot.gradient(x, t=0))


# ==========================================================================


class TestNFWPotential(TestAbstractPotential):
    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.NFWPotential]:
        return gp.NFWPotential

    @pytest.fixture(scope="class")
    def field_m(self) -> dict[str, Any]:
        return 1e12

    @pytest.fixture(scope="class")
    def field_r_s(self) -> dict[str, Any]:
        return 1

    @pytest.fixture(scope="class")
    def field_softening_length(self) -> dict[str, Any]:
        return 0.001

    @pytest.fixture(scope="class")
    def fields_(
        self, field_m, field_r_s, field_softening_length, field_units
    ) -> dict[str, Any]:
        return {
            "m": field_m,
            "r_s": field_r_s,
            "softening_length": field_softening_length,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential_energy(self, pot, x) -> None:
        assert xp.isclose(pot.potential_energy(x, t=0), xp.array(-1.87117234))

    def test_gradient(self, pot, x):
        assert xp.allclose(
            pot.gradient(x, t=0), xp.array([0.0658867, 0.1317734, 0.19766011])
        )

    def test_density(self, pot, x):
        assert xp.isclose(pot.density(x, t=0), 9.46039849e08)

    def test_hessian(self, pot, x):
        assert xp.allclose(
            pot.hessian(x, t=0),
            xp.array(
                [
                    [0.05558809, -0.02059723, -0.03089585],
                    [-0.02059723, 0.02469224, -0.06179169],
                    [-0.03089585, -0.06179169, -0.02680084],
                ]
            ),
        )

    def test_acceleration(self, pot, x):
        assert xp.allclose(pot.acceleration(x, t=0), -pot.gradient(x, t=0))
