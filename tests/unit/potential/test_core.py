from functools import partial
from typing import Any, ClassVar

import equinox as eqx
import jax
import pytest
from plum import convert

import quaxed.numpy as jnp
import unxt as u
from unxt.unitsystems import AbstractUnitSystem, galactic
from xmmutablemap import ImmutableMap

import galax.potential as gp
import galax.potential.params as gpp
import galax.typing as gt
from .test_base import AbstractPotential_Test
from .test_utils import FieldUnitSystemMixin
from galax.potential._src.base import default_constants


class AbstractSinglePotential_Test(AbstractPotential_Test, FieldUnitSystemMixin):
    """Test the `galax.potential.AbstractPotential` class."""

    @pytest.fixture(scope="class")
    def units(self) -> AbstractUnitSystem:
        return galactic

    @pytest.fixture(scope="class")
    def fields_(self, field_units: AbstractUnitSystem) -> dict[str, Any]:
        return {"units": field_units}


###############################################################################


class TestAbstractSinglePotential(AbstractSinglePotential_Test):
    """Test the `galax.potential.AbstractPotential` class."""

    HAS_GALA_COUNTERPART: ClassVar[bool] = False

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.AbstractPotential]:
        class TestPotential(gp.AbstractPotential):
            m_tot: gpp.AbstractParameter = gpp.ParameterField(
                dimensions="mass", default=u.Quantity(1e12, "Msun")
            )
            units: AbstractUnitSystem = eqx.field(
                default=galactic, converter=u.unitsystem, static=True
            )
            constants: ImmutableMap[str, u.Quantity] = eqx.field(
                default=default_constants, converter=ImmutableMap
            )

            @partial(jax.jit, inline=True)
            def _potential(  # TODO: inputs w/ units
                self, q: gt.BtQuSz3, t: gt.BBtRealQuSz0, /
            ) -> gt.SpecificEnergyBtSz0:
                return (
                    self.constants["G"]
                    * self.m_tot(t)
                    / jnp.linalg.vector_norm(q, axis=-1)
                )

        return TestPotential

    ###########################################################################

    def test_init(self, pot_cls) -> None:
        """Test the initialization of `AbstractPotential`."""
        # Test that the abstract class cannot be instantiated
        with pytest.raises(TypeError):
            gp.AbstractPotential()

        # Test that the concrete class can be instantiated
        pot = pot_cls()
        assert isinstance(pot, gp.AbstractPotential)

    # =========================================================================

    # ---------------------------------

    def test_potential(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.potential` method."""
        assert jnp.allclose(
            pot.potential(x, t=0),
            u.Quantity(1.20227527, "kpc2/Myr2"),
            atol=u.Quantity(1e-8, "kpc2/Myr2"),
        )

    # ---------------------------------

    def test_gradient(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.gradient` method."""
        expect = u.Quantity(
            [-0.08587681, -0.17175361, -0.25763042], pot.units["acceleration"]
        )
        got = convert(pot.gradient(x, t=0), u.Quantity)
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.density` method."""
        # TODO: fix negative density!!!
        expect = u.Quantity(-2.647e-7, pot.units["mass density"])
        assert jnp.allclose(
            pot.density(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.hessian` method."""
        expected = u.Quantity(
            jnp.asarray(
                [
                    [-0.06747463, 0.03680435, 0.05520652],
                    [0.03680435, -0.01226812, 0.11041304],
                    [0.05520652, 0.11041304, 0.07974275],
                ]
            ),
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.hessian(x, t=0), expected, atol=u.Quantity(1e-8, "1/Myr2")
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        expect = u.Quantity(
            [
                [-0.06747463, 0.03680435, 0.05520652],
                [0.03680435, -0.01226812, 0.11041304],
                [0.05520652, 0.11041304, 0.07974275],
            ],
            pot.units["frequency drift"],
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )
