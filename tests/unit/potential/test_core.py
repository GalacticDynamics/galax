from functools import partial
from typing import Any, ClassVar

import equinox as eqx
import jax
import pytest
from plum import convert

import quaxed.numpy as jnp
from unxt import Quantity
from unxt.unitsystems import AbstractUnitSystem, galactic, unitsystem
from xmmutablemap import ImmutableMap

import galax.potential as gp
import galax.potential.params as gpp
import galax.typing as gt
from .test_base import AbstractPotentialBase_Test
from .test_utils import FieldUnitSystemMixin
from galax.potential._src.base import default_constants


class AbstractPotential_Test(AbstractPotentialBase_Test, FieldUnitSystemMixin):
    """Test the `galax.potential.AbstractPotentialBase` class."""

    @pytest.fixture(scope="class")
    def units(self) -> AbstractUnitSystem:
        return galactic

    @pytest.fixture(scope="class")
    def fields_(self, field_units: AbstractUnitSystem) -> dict[str, Any]:
        return {"units": field_units}


###############################################################################


class TestAbstractPotential(AbstractPotential_Test):
    """Test the `galax.potential.AbstractPotentialBase` class."""

    HAS_GALA_COUNTERPART: ClassVar[bool] = False

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.AbstractPotentialBase]:
        class TestPotential(gp.AbstractPotentialBase):
            m_tot: gpp.AbstractParameter = gpp.ParameterField(
                dimensions="mass", default=Quantity(1e12, "Msun")
            )
            units: AbstractUnitSystem = eqx.field(
                default=galactic, converter=unitsystem, static=True
            )
            constants: ImmutableMap[str, Quantity] = eqx.field(
                default=default_constants, converter=ImmutableMap
            )

            @partial(jax.jit, inline=True)
            def _potential(  # TODO: inputs w/ units
                self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
            ) -> gt.SpecificEnergyBatchScalar:
                return (
                    self.constants["G"]
                    * self.m_tot(t)
                    / jnp.linalg.vector_norm(q, axis=-1)
                )

        return TestPotential

    ###########################################################################

    def test_init(self, pot_cls) -> None:
        """Test the initialization of `AbstractPotentialBase`."""
        # Test that the abstract class cannot be instantiated
        with pytest.raises(TypeError):
            gp.AbstractPotentialBase()

        # Test that the concrete class can be instantiated
        pot = pot_cls()
        assert isinstance(pot, gp.AbstractPotentialBase)

    # =========================================================================

    # ---------------------------------

    def test_potential(self, pot: gp.AbstractPotentialBase, x: gt.QVec3) -> None:
        """Test the `AbstractPotentialBase.potential` method."""
        assert jnp.allclose(
            pot.potential(x, t=0),
            Quantity(1.20227527, "kpc2/Myr2"),
            atol=Quantity(1e-8, "kpc2/Myr2"),
        )

    # ---------------------------------

    def test_gradient(self, pot: gp.AbstractPotentialBase, x: gt.QVec3) -> None:
        """Test the `AbstractPotentialBase.gradient` method."""
        expect = Quantity(
            [-0.08587681, -0.17175361, -0.25763042], pot.units["acceleration"]
        )
        got = convert(pot.gradient(x, t=0), Quantity)
        assert jnp.allclose(got, expect, atol=Quantity(1e-8, expect.unit))

    def test_density(self, pot: gp.AbstractPotentialBase, x: gt.QVec3) -> None:
        """Test the `AbstractPotentialBase.density` method."""
        # TODO: fix negative density!!!
        expect = Quantity(-2.647e-7, pot.units["mass density"])
        assert jnp.allclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: gp.AbstractPotentialBase, x: gt.QVec3) -> None:
        """Test the `AbstractPotentialBase.hessian` method."""
        expected = Quantity(
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
            pot.hessian(x, t=0), expected, atol=Quantity(1e-8, "1/Myr2")
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: gp.AbstractPotentialBase, x: gt.QVec3) -> None:
        """Test the `AbstractPotentialBase.tidal_tensor` method."""
        expect = Quantity(
            [
                [-0.06747463, 0.03680435, 0.05520652],
                [0.03680435, -0.01226812, 0.11041304],
                [0.05520652, 0.11041304, 0.07974275],
            ],
            pot.units["frequency drift"],
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )
