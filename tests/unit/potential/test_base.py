import copy
from typing import Any

import equinox as eqx
import jax.numpy as xp
import pytest
from jaxtyping import Array, Float

import galax.dynamics as gd
import galax.potential as gp
from galax.typing import BatchableFloatOrIntScalarLike, BatchFloatScalar, BatchVec3
from galax.utils import partial_jit, vectorize_method


class TestAbstractPotentialBase:
    """Test the `galax.potential.AbstractPotentialBase` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.AbstractPotentialBase]:
        class TestPotential(gp.AbstractPotentialBase):
            units: float = 2
            _G: float = eqx.field(init=False, static=True, repr=False, converter=float)

            def __post_init__(self):
                object.__setattr__(self, "_G", 1.0)

            @partial_jit()
            @vectorize_method(signature="(3),()->()")
            def _potential_energy(
                self, q: BatchVec3, t: BatchableFloatOrIntScalarLike
            ) -> BatchFloatScalar:
                return xp.sum(q, axis=-1)

        return TestPotential

    @pytest.fixture(scope="class")
    def fields_(self) -> dict[str, Any]:
        return {}

    @pytest.fixture()
    def fields(self, fields_) -> dict[str, Any]:
        return copy.copy(fields_)

    @pytest.fixture(scope="class")
    def pot(
        self, pot_cls: type[gp.AbstractPotentialBase], fields_: dict[str, Any]
    ) -> gp.AbstractPotentialBase:
        """Create a concrete potential instance for testing."""
        return pot_cls(**fields_)

    # ---------------------------------

    @pytest.fixture(scope="class")
    def x(self) -> Float[Array, "3"]:
        """Create a position vector for testing."""
        return xp.array([1, 2, 3], dtype=float)

    @pytest.fixture(scope="class")
    def v(self) -> Float[Array, "3"]:
        """Create a velocity vector for testing."""
        return xp.array([4, 5, 6], dtype=float)

    @pytest.fixture(scope="class")
    def xv(self, x: Float[Array, "3"], v: Float[Array, "3"]) -> Float[Array, "6"]:
        """Create a phase-space vector for testing."""
        return xp.concatenate([x, v])

    @pytest.fixture(scope="class")
    def t(self) -> float:
        """Create a time for testing."""
        return 0.0

    ###########################################################################

    def test_init(self):
        """Test the initialization of `AbstractPotentialBase`."""
        # Test that the abstract class cannot be instantiated
        with pytest.raises(TypeError):
            gp.AbstractPotentialBase()

        # Test that the concrete class can be instantiated
        class TestPotential(gp.AbstractPotentialBase):
            units: float = 2

            def _potential_energy(self, q, t):
                return xp.sum(q, axis=-1)

        pot = TestPotential()
        assert isinstance(pot, gp.AbstractPotentialBase)

    # =========================================================================

    def test_potential_energy(self, pot, x):
        """Test the `AbstractPotentialBase.potential_energy` method."""
        assert pot.potential_energy(x, t=0) == 6

    def test_call(self, pot, x):
        """Test the `AbstractPotentialBase.__call__` method."""
        assert xp.equal(pot(x, t=0), pot.potential_energy(x, t=0))

    def test_gradient(self, pot, x):
        """Test the `AbstractPotentialBase.gradient` method."""
        assert xp.array_equal(pot.gradient(x, t=0), xp.ones_like(x))

    def test_density(self, pot, x):
        """Test the `AbstractPotentialBase.density` method."""
        assert pot.density(x, t=0) == 0.0

    def test_hessian(self, pot, x):
        """Test the `AbstractPotentialBase.hessian` method."""
        assert xp.array_equal(
            pot.hessian(x, t=0),
            xp.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            ),
        )

    def test_acceleration(self, pot, x):
        """Test the `AbstractPotentialBase.acceleration` method."""
        assert xp.array_equal(pot.acceleration(x, t=0), xp.array([-1.0, -1, -1]))
        assert xp.array_equal(pot.acceleration(x, t=0), -pot.gradient(x, t=0))

    def test_integrate_orbit(self, pot, xv):
        """Test the `AbstractPotentialBase.integrate_orbit` method."""
        ts = xp.linspace(0.0, 1.0, 100)

        orbit = pot.integrate_orbit(xv, ts)
        assert isinstance(orbit, gd.Orbit)
        assert orbit.shape == (len(ts), 7)
        assert xp.array_equal(orbit.t, ts)
