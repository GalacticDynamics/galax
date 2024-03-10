from typing import Any

import jax.numpy as jnp
import pytest

import quaxed.array_api as xp
import quaxed.numpy as qnp
from jax_quantity import Quantity

from ..test_core import TestAbstractPotential as AbstractPotential_Test
from galax.potential import AbstractPotentialBase, NullPotential
from galax.typing import Vec3
from galax.units import UnitSystem


class TestNullPotential(AbstractPotential_Test):
    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[NullPotential]:
        return NullPotential

    @pytest.fixture(scope="class")
    def fields_(self, field_units: UnitSystem) -> dict[str, Any]:
        return {"units": field_units}

    # ==========================================================================

    def test_potential_energy(self, pot: NullPotential, x: Vec3) -> None:
        """Test :meth:`NullPotential.potential_energy`."""
        assert jnp.isclose(pot.potential_energy(x, t=0).value, xp.asarray(0.0))

    def test_gradient(self, pot: NullPotential, x: Vec3) -> None:
        """Test :meth:`NullPotential.gradient`."""
        expected = Quantity([0.0, 0.0, 0.0], pot.units["acceleration"])
        assert qnp.allclose(pot.gradient(x, t=0).value, expected.value)  # TODO: value

    def test_density(self, pot: NullPotential, x: Vec3) -> None:
        """Test :meth:`NullPotential.density`."""
        assert jnp.isclose(pot.density(x, t=0).value, 0.0)

    def test_hessian(self, pot: NullPotential, x: Vec3) -> None:
        """Test :meth:`NullPotential.hessian`."""
        assert jnp.allclose(
            pot.hessian(x, t=0),
            xp.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: AbstractPotentialBase, x: Vec3) -> None:
        """Test the `AbstractPotentialBase.tidal_tensor` method."""
        expect = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        assert qnp.allclose(pot.tidal_tensor(x, t=0), xp.asarray(expect))
