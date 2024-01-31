from typing import Any

import jax.experimental.array_api as xp
import jax.numpy as jnp
import pytest

import galax.potential as gp
from galax.typing import Vec3
from galax.units import UnitSystem

from ..test_core import TestAbstractPotential as AbstractPotential_Test


class TestNullPotential(AbstractPotential_Test):
    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.NullPotential]:
        return gp.NullPotential

    @pytest.fixture(scope="class")
    def fields_(self, field_units: UnitSystem) -> dict[str, Any]:
        return {"units": field_units}

    # ==========================================================================

    def test_potential_energy(self, pot: gp.NullPotential, x: Vec3) -> None:
        """Test :meth:`NullPotential.potential_energy`."""
        assert jnp.isclose(pot.potential_energy(x, t=0), xp.asarray(0.0))

    def test_gradient(self, pot: gp.NullPotential, x: Vec3) -> None:
        """Test :meth:`NullPotential.gradient`."""
        assert jnp.allclose(pot.gradient(x, t=0), xp.asarray([0.0, 0.0, 0.0]))

    def test_density(self, pot: gp.NullPotential, x: Vec3) -> None:
        """Test :meth:`NullPotential.density`."""
        assert jnp.isclose(pot.density(x, t=0), 0.0)

    def test_hessian(self, pot: gp.NullPotential, x: Vec3) -> None:
        """Test :meth:`NullPotential.hessian`."""
        assert jnp.allclose(
            pot.hessian(x, t=0),
            xp.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        )
