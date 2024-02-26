from typing import Any

import array_api_jax_compat as xp
import astropy.units as u
import jax.numpy as jnp
import pytest
from quax import quaxify

from jax_quantity import Quantity

import galax.potential as gp
from ..test_core import TestAbstractPotential as AbstractPotential_Test
from .test_common import MassParameterMixin, ShapeAParameterMixin, ShapeBParameterMixin
from galax.potential import AbstractPotentialBase, MiyamotoNagaiPotential
from galax.typing import Vec3
from galax.units import UnitSystem

allclose = quaxify(jnp.allclose)


class TestMiyamotoNagaiPotential(
    AbstractPotential_Test,
    # Parameters
    MassParameterMixin,
    ShapeAParameterMixin,
    ShapeBParameterMixin,
):
    """Test the `galax.potential.MiyamotoNagaiPotential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.MiyamotoNagaiPotential]:
        return gp.MiyamotoNagaiPotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m: u.Quantity,
        field_a: u.Quantity,
        field_b: u.Quantity,
        field_units: UnitSystem,
    ) -> dict[str, Any]:
        return {"m": field_m, "a": field_a, "b": field_b, "units": field_units}

    # ==========================================================================

    def test_potential_energy(self, pot: MiyamotoNagaiPotential, x: Vec3) -> None:
        assert jnp.isclose(pot.potential_energy(x, t=0).value, xp.asarray(-0.95208676))

    def test_gradient(self, pot: MiyamotoNagaiPotential, x: Vec3) -> None:
        expected = Quantity(
            [0.04264751, 0.08529503, 0.16840152], pot.units["acceleration"]
        )
        assert allclose(pot.gradient(x, t=0).value, expected.value)  # TODO: not .value

    def test_density(self, pot: MiyamotoNagaiPotential, x: Vec3) -> None:
        assert jnp.isclose(pot.density(x, t=0).value, 1.9949418e08)

    def test_hessian(self, pot: MiyamotoNagaiPotential, x: Vec3) -> None:
        assert jnp.allclose(
            pot.hessian(x, t=0),
            xp.asarray(
                [
                    [0.03691649, -0.01146205, -0.02262999],
                    [-0.01146205, 0.01972342, -0.04525999],
                    [-0.02262999, -0.04525999, -0.04536254],
                ]
            ),
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: AbstractPotentialBase, x: Vec3) -> None:
        """Test the `AbstractPotentialBase.tidal_tensor` method."""
        expect = [
            [0.03315736, -0.01146205, -0.02262999],
            [-0.01146205, 0.0159643, -0.04525999],
            [-0.02262999, -0.04525999, -0.04912166],
        ]
        assert allclose(pot.tidal_tensor(x, t=0), xp.asarray(expect))
