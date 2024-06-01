import copy
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Any, ClassVar, final

import astropy.units as u
import equinox as eqx
import jax
import pytest

import quaxed.array_api as xp
import quaxed.numpy as qnp
from unxt import AbstractUnitSystem, Quantity
from unxt.unitsystems import galactic

import galax.dynamics as gd
import galax.typing as gt
from .io.test_gala import GalaIOMixin
from galax.potential import AbstractParameter, AbstractPotentialBase, ParameterField
from galax.potential._potential.base import default_constants
from galax.utils import ImmutableDict


class AbstractPotentialBase_Test(GalaIOMixin, metaclass=ABCMeta):
    """Test the `galax.potential.AbstractPotentialBase` class."""

    @pytest.fixture(scope="class")
    @abstractmethod
    def pot_cls(self) -> type[AbstractPotentialBase]: ...

    @pytest.fixture(scope="class")
    def units(self) -> AbstractUnitSystem:
        return galactic

    @pytest.fixture(scope="class")
    def field_units(self, units: AbstractUnitSystem) -> AbstractUnitSystem:
        return units

    @pytest.fixture(scope="class")
    def fields_(self, field_units: AbstractUnitSystem) -> dict[str, Any]:
        return {"units": field_units}

    @pytest.fixture()
    def fields(self, fields_) -> dict[str, Any]:
        return copy.copy(fields_)

    @pytest.fixture(scope="class")
    def pot(
        self, pot_cls: type[AbstractPotentialBase], fields_: dict[str, Any]
    ) -> AbstractPotentialBase:
        """Create a concrete potential instance for testing."""
        return pot_cls(**fields_)

    # ---------------------------------

    @pytest.fixture(scope="class")
    def x(self, units: AbstractUnitSystem) -> gt.QVec3:
        """Create a position vector for testing."""
        return Quantity(xp.asarray([1, 2, 3], dtype=float), units["length"])

    @pytest.fixture(scope="class")
    def v(sel, units: AbstractUnitSystem) -> gt.QVec3:
        """Create a velocity vector for testing."""
        return Quantity(xp.asarray([4, 5, 6], dtype=float), units["speed"])

    @pytest.fixture(scope="class")
    def xv(self, x: gt.QVec3, v: gt.QVec3) -> gt.Vec6:
        """Create a phase-space vector for testing."""
        return xp.concat([x.value, v.value])

    # ---------------------------------

    @pytest.fixture(scope="class")
    def batchx(self, units: AbstractUnitSystem) -> gt.BatchQVec3:
        """Create a batch of position vectors for testing."""
        return Quantity(
            xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float), units["length"]
        )

    @pytest.fixture(scope="class")
    def batchv(self, units: AbstractUnitSystem) -> gt.BatchQVec3:
        """Create a batch of velocity vectors for testing."""
        return Quantity(
            xp.asarray([[4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=float),
            units["speed"],
        )

    @pytest.fixture(scope="class")
    def batchxv(self, batchx: gt.BatchQVec3, batchv: gt.BatchQVec3) -> gt.BatchVec3:
        """Create a batch of phase-space vectors for testing."""
        return xp.concatenate([batchx.value, batchv.value], axis=-1)

    # ---------------------------------

    @pytest.fixture(scope="class")
    def t(self) -> Quantity["time"]:
        """Create a time for testing."""
        return Quantity(0.0, "Gyr")

    ###########################################################################

    def test_init(
        self, pot_cls: type[AbstractPotentialBase], fields_: dict[str, Any]
    ) -> AbstractPotentialBase:
        """Create a concrete potential instance for testing."""
        pot = pot_cls(**fields_)
        assert isinstance(pot, pot_cls)

        # TODO: more tests

    # =========================================================================

    # ---------------------------------

    @abstractmethod
    def test_potential(self, pot: AbstractPotentialBase, x: gt.QVec3) -> None:
        """Test the `potential` method."""
        ...

    def test_potential_batch(
        self, pot: AbstractPotentialBase, batchx: gt.BatchQVec3
    ) -> None:
        """Test the `AbstractPotentialBase.potential` method."""
        # Test that the method works on batches.
        assert pot.potential(batchx, t=0).shape == batchx.shape[:-1]
        # Test that the batched method is equivalent to the scalar method
        assert qnp.allclose(
            pot.potential(batchx, t=0)[0],
            pot.potential(batchx[0], t=0),
            atol=Quantity(1e-15, pot.units["specific energy"]),
        )

    # ---------------------------------

    def test_call(self, pot: AbstractPotentialBase, x: gt.QVec3) -> None:
        """Test the `AbstractPotentialBase.__call__` method."""
        assert xp.equal(pot(x, 0), pot.potential(x, 0))

    @abstractmethod
    def test_gradient(self, pot: AbstractPotentialBase, x: gt.QVec3) -> None:
        """Test the `AbstractPotentialBase.gradient` method."""
        ...

    @abstractmethod
    def test_density(self, pot: AbstractPotentialBase, x: gt.QVec3) -> None:
        """Test the `AbstractPotentialBase.density` method."""
        ...

    @abstractmethod
    def test_hessian(self, pot: AbstractPotentialBase, x: gt.QVec3) -> None:
        """Test the `AbstractPotentialBase.hessian` method."""
        ...

    def test_acceleration(self, pot: AbstractPotentialBase, x: gt.QVec3) -> None:
        """Test the `AbstractPotentialBase.acceleration` method."""
        assert qnp.array_equal(pot.acceleration(x, t=0), -pot.gradient(x, t=0))

    # ---------------------------------
    # Convenience methods

    @abstractmethod
    def test_tidal_tensor(self, pot: AbstractPotentialBase, x: gt.QVec3) -> None:
        """Test the `AbstractPotentialBase.tidal_tensor` method."""
        ...

    # =========================================================================

    def test_evaluate_orbit(self, pot: AbstractPotentialBase, xv: gt.Vec6) -> None:
        """Test the `AbstractPotentialBase.evaluate_orbit` method."""
        ts = Quantity(xp.linspace(0.0, 1.0, 100), "Myr")

        orbit = pot.evaluate_orbit(xv, ts)
        assert isinstance(orbit, gd.Orbit)
        assert orbit.shape == (len(ts.value),)  # TODO: don't use .value
        assert qnp.array_equal(orbit.t, ts)

    def test_evaluate_orbit_batch(
        self, pot: AbstractPotentialBase, xv: gt.Vec6
    ) -> None:
        """Test the `AbstractPotentialBase.evaluate_orbit` method."""
        ts = Quantity(xp.linspace(0.0, 1.0, 100), "Myr")

        # Simple batch
        orbits = pot.evaluate_orbit(xv[None, :], ts)
        assert isinstance(orbits, gd.Orbit)
        assert orbits.shape == (1, len(ts))
        assert qnp.allclose(orbits.t, ts, atol=Quantity(1e-16, "Myr"))

        # More complicated batch
        xv2 = xp.stack([xv, xv], axis=0)
        orbits = pot.evaluate_orbit(xv2, ts)
        assert isinstance(orbits, gd.Orbit)
        assert orbits.shape == (2, len(ts))
        assert qnp.allclose(orbits.t, ts, atol=Quantity(1e-16, "Myr"))


##############################################################################


@final
class TestAbstractPotentialBase(AbstractPotentialBase_Test):
    """Test the `galax.potential.AbstractPotentialBase` class."""

    HAS_GALA_COUNTERPART: ClassVar[bool] = False

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[AbstractPotentialBase]:
        class TestPotential(AbstractPotentialBase):
            m_tot: AbstractParameter = ParameterField(
                dimensions="mass", default=1e12 * u.Msun
            )
            units: AbstractUnitSystem = eqx.field(default=galactic, static=True)
            constants: ImmutableDict[Quantity] = eqx.field(
                default=default_constants, converter=ImmutableDict
            )

            @partial(jax.jit)
            def _potential(  # TODO: inputs w/ units
                self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
            ) -> gt.BatchFloatQScalar:
                return (
                    self.constants["G"]
                    * self.m_tot(t)
                    / xp.linalg.vector_norm(q, axis=-1)
                )

        return TestPotential

    ###########################################################################

    def test_init(self, pot_cls) -> None:
        """Test the initialization of `AbstractPotentialBase`."""
        # Test that the abstract class cannot be instantiated
        with pytest.raises(TypeError):
            AbstractPotentialBase()

        # Test that the concrete class can be instantiated
        pot = pot_cls()
        assert isinstance(pot, AbstractPotentialBase)

    # =========================================================================

    # ---------------------------------

    def test_potential(self, pot: AbstractPotentialBase, x: gt.QVec3) -> None:
        """Test the `AbstractPotentialBase.potential` method."""
        assert qnp.allclose(
            pot.potential(x, t=0),
            Quantity(1.20227527, "kpc2/Myr2"),
            atol=Quantity(1e-8, "kpc2/Myr2"),
        )

    # ---------------------------------

    def test_gradient(self, pot: AbstractPotentialBase, x: gt.QVec3) -> None:
        """Test the `AbstractPotentialBase.gradient` method."""
        expect = Quantity(
            [-0.08587681, -0.17175361, -0.25763042], pot.units["acceleration"]
        )
        assert qnp.allclose(
            pot.gradient(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_density(self, pot: AbstractPotentialBase, x: gt.QVec3) -> None:
        """Test the `AbstractPotentialBase.density` method."""
        # TODO: fix negative density!!!
        expect = Quantity(-2.647e-7, pot.units["mass density"])
        assert qnp.allclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: AbstractPotentialBase, x: gt.QVec3) -> None:
        """Test the `AbstractPotentialBase.hessian` method."""
        expected = Quantity(
            xp.asarray(
                [
                    [-0.06747463, 0.03680435, 0.05520652],
                    [0.03680435, -0.01226812, 0.11041304],
                    [0.05520652, 0.11041304, 0.07974275],
                ]
            ),
            "1/Myr2",
        )
        assert qnp.allclose(
            pot.hessian(x, t=0), expected, atol=Quantity(1e-8, "1/Myr2")
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: AbstractPotentialBase, x: gt.QVec3) -> None:
        """Test the `AbstractPotentialBase.tidal_tensor` method."""
        expect = Quantity(
            [
                [-0.06747463, 0.03680435, 0.05520652],
                [0.03680435, -0.01226812, 0.11041304],
                [0.05520652, 0.11041304, 0.07974275],
            ],
            pot.units["frequency drift"],
        )
        assert qnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )
