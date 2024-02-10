from typing import Any

import array_api_jax_compat as xp
import astropy.units as u
import jax.numpy as jnp
import pytest

import galax.potential as gp
from galax.potential import MiyamotoNagaiPotential
from galax.typing import Vec3
from galax.units import UnitSystem

from ..test_core import TestAbstractPotential as AbstractPotential_Test
from .test_common import MassParameterMixin, ShapeAParameterMixin, ShapeBParameterMixin


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
        assert jnp.isclose(pot.potential_energy(x, t=0), xp.asarray(-0.95208676))

    def test_gradient(self, pot: MiyamotoNagaiPotential, x: Vec3) -> None:
        assert jnp.allclose(
            pot.gradient(x, t=0), xp.asarray([0.04264751, 0.08529503, 0.16840152])
        )

    def test_density(self, pot: MiyamotoNagaiPotential, x: Vec3) -> None:
        assert jnp.isclose(pot.density(x, t=0), 1.9949418e08)

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
