import copy
import functools as ft
from abc import ABCMeta, abstractmethod

from typing import Any, ClassVar, final

import equinox as eqx
import jax
import pytest

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue
from xmmutablemap import ImmutableMap

import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp
import galax.potential.custom_types as gt
import galax.potential.params as gpp
from .io.test_gala import GalaIOMixin
from galax.potential._src.base import default_constants


class AbstractPotential_Test(GalaIOMixin, metaclass=ABCMeta):
    """Test the `galax.potential.AbstractPotential` class."""

    @pytest.fixture(scope="class")
    @abstractmethod
    def pot_cls(self) -> type[gp.AbstractPotential]: ...

    @pytest.fixture(scope="class")
    def units(self) -> u.AbstractUnitSystem:
        return u.unitsystems.galactic

    @pytest.fixture(scope="class")
    def field_units(self, units: u.AbstractUnitSystem) -> u.AbstractUnitSystem:
        return units

    @pytest.fixture(scope="class")
    def fields_(self, field_units: u.AbstractUnitSystem) -> dict[str, Any]:
        return {"units": field_units}

    @pytest.fixture
    def fields(self, fields_) -> dict[str, Any]:
        return copy.copy(fields_)

    @pytest.fixture(scope="class")
    def pot(
        self, pot_cls: type[gp.AbstractPotential], fields_: dict[str, Any]
    ) -> gp.AbstractPotential:
        """Create a concrete potential instance for testing."""
        return pot_cls(**fields_)

    # ---------------------------------

    @pytest.fixture(scope="class")
    def x(self, units: u.AbstractUnitSystem) -> gt.QuSz3:
        """Create a position vector for testing."""
        return u.Q(jnp.asarray([1, 2, 3], dtype=float), units["length"])

    @pytest.fixture(scope="class")
    def v(sel, units: u.AbstractUnitSystem) -> gt.QuSz3:
        """Create a velocity vector for testing."""
        return u.Q(jnp.asarray([4, 5, 6], dtype=float), units["speed"])

    @pytest.fixture(scope="class")
    def xv(self, x: gt.QuSz3, v: gt.QuSz3) -> gt.Sz6:
        """Create a phase-space vector for testing."""
        return jnp.concat([x.value, v.value])

    # ---------------------------------

    @pytest.fixture(scope="class")
    def batchx(self, units: u.AbstractUnitSystem) -> gt.BBtQuSz3:
        """Create a batch of position vectors for testing."""
        return u.Q(
            jnp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float), units["length"]
        )

    @pytest.fixture(scope="class")
    def batchv(self, units: u.AbstractUnitSystem) -> gt.BBtQuSz3:
        """Create a batch of velocity vectors for testing."""
        return u.Q(
            jnp.asarray([[4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=float),
            units["speed"],
        )

    @pytest.fixture(scope="class")
    def batchxv(self, batchx: gt.BBtQuSz3, batchv: gt.BBtQuSz3) -> gt.BtSz3:
        """Create a batch of phase-space vectors for testing."""
        return jnp.concatenate([batchx.value, batchv.value], axis=-1)

    # ---------------------------------

    @pytest.fixture(scope="class")
    def t(self) -> u.Quantity["time"]:
        """Create a time for testing."""
        return u.Q(0.0, "Gyr")

    ###########################################################################

    def test_init(
        self, pot_cls: type[gp.AbstractPotential], fields_: dict[str, Any]
    ) -> gp.AbstractPotential:
        """Create a concrete potential instance for testing."""
        pot = pot_cls(**fields_)
        assert isinstance(pot, pot_cls)

        # TODO: more tests

    # =========================================================================

    # ---------------------------------

    @abstractmethod
    def test_potential(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `potential` method."""
        ...

    def test_potential_batch(
        self, pot: gp.AbstractPotential, batchx: gt.BBtQuSz3
    ) -> None:
        """Test the `AbstractPotential.potential` method."""
        # Test that the method works on batches.
        assert pot.potential(batchx, t=0).shape == batchx.shape[:-1]
        # Test that the batched method is equivalent to the scalar method
        assert jnp.allclose(
            pot.potential(batchx, t=0)[0],
            pot.potential(batchx[0], t=0),
            atol=u.Q(1e-15, pot.units["specific energy"]),
        )

    def test_potential_density_correspondence(
        self, pot: gp.AbstractPotential, x: gt.QuSz3
    ) -> None:
        lhs = jnp.trace(pot.hessian(x, 0))
        rhs = 4 * jnp.pi * pot.constants["G"] * pot.density(x, 0)
        assert jnp.isclose(lhs, rhs, atol=u.Q(1e-15, pot.units["frequency drift"]))

    # ---------------------------------

    def test_call(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.__call__` method."""
        assert jnp.equal(pot(x, 0), pot.potential(x, 0))

    @abstractmethod
    def test_gradient(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.gradient` method."""
        ...

    @abstractmethod
    def test_density(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.density` method."""
        ...

    @abstractmethod
    def test_hessian(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.hessian` method."""
        ...

    def test_acceleration(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.acceleration` method."""
        acc = pot.acceleration(x, t=0)
        grad = pot.gradient(x, t=0)
        assert jnp.array_equal(acc, -grad)

    # ---------------------------------
    # Convenience methods

    @abstractmethod
    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        ...

    # =========================================================================

    def test_evaluate_orbit(self, pot: gp.AbstractPotential, xv: gt.Sz6) -> None:
        """Test the `AbstractPotential.evaluate_orbit` method."""
        ts = u.Q(jnp.linspace(0.0, 1.0, 100), "Myr")

        orbit = pot.evaluate_orbit(xv, ts)
        assert isinstance(orbit, gd.Orbit)
        assert orbit.shape == (len(ts.value),)  # TODO: don't use .value
        assert jnp.array_equal(orbit.t, ts)

    def test_evaluate_orbit_batch(self, pot: gp.AbstractPotential, xv: gt.Sz6) -> None:
        """Test the `AbstractPotential.evaluate_orbit` method."""
        ts = u.Q(jnp.linspace(0.0, 1.0, 100), "Myr")

        # Simple batch
        orbits = pot.evaluate_orbit(xv[None, :], ts)
        assert isinstance(orbits, gd.Orbit)
        assert orbits.shape == (1, len(ts))
        assert jnp.allclose(orbits.t, ts, atol=u.Q(1e-16, "Myr"))

        # More complicated batch
        xv2 = jnp.stack([xv, xv], axis=0)
        orbits = pot.evaluate_orbit(xv2, ts)
        assert isinstance(orbits, gd.Orbit)
        assert orbits.shape == (2, len(ts))
        assert jnp.allclose(orbits.t, ts, atol=u.Q(1e-16, "Myr"))

    # =========================================================================

    @pytest.mark.parametrize(
        "func",
        [
            gp.potential,
            gp.gradient,
            gp.laplacian,
            gp.density,
            gp.hessian,
            gp.acceleration,
            gp.tidal_tensor,
            gp.local_circular_velocity,
            gp.dpotential_dr,
            gp.d2potential_dr2,
        ],
    )
    def test_no_time(self, func: Any, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Omitting the time is allowed only for a time-independent potential."""
        if pot.is_time_dependent:
            with pytest.raises(TypeError, match="depends on time"):
                func(pot, x)
            return

        # Time-independent, so any time gives the same result.
        # Quantity
        got, exp = func(pot, x), func(pot, x, u.Q(3.7, "Gyr"))
        assert type(got) is type(exp)
        assert got.unit == exp.unit
        assert jnp.allclose(got.value, exp.value, equal_nan=True)
        # Array
        got, exp = func(pot, x.value), func(pot, x.value, 3.7)
        assert type(got) is type(exp)
        assert jnp.allclose(got, exp, equal_nan=True)


##############################################################################


@final
class TestAbstractPotential(AbstractPotential_Test):
    """Test the `galax.potential.AbstractPotential` class."""

    HAS_GALA_COUNTERPART: ClassVar[bool] = False

    @pytest.fixture(scope="class")
    def pot_cls(self, units) -> type[gp.AbstractPotential]:
        usys = units
        constants_in_usys = {k: v.decompose(usys) for k, v in default_constants.items()}

        class TestPotential(gp.AbstractPotential):
            m_tot: gpp.AbstractParameter = gpp.ParameterField(
                dimensions="mass", default=u.Q(1e12, "Msun")
            )
            units: u.AbstractUnitSystem = eqx.field(
                default=u.unitsystems.galactic, static=True
            )
            constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
                default=ImmutableMap(constants_in_usys),
                converter=ImmutableMap,
            )

            @ft.partial(jax.jit)
            def _potential(  # TODO: inputs w/ units
                self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /
            ) -> gt.BBtSz0:
                m_tot = self.m_tot(t, ustrip=self.units["mass"])
                xyz = u.ustrip(AllowValue, self.units["length"], xyz)
                return (
                    self.constants["G"].value
                    * m_tot
                    / jnp.linalg.vector_norm(xyz, axis=-1)
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
        exp = u.Q(1.20227527, "kpc2/Myr2")
        got = pot.potential(x, t=0)
        assert jnp.allclose(got, exp, atol=u.Q(1e-8, "kpc2/Myr2"))

    # ---------------------------------

    def test_gradient(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.gradient` method."""
        expect = u.Q([-0.08587681, -0.17175361, -0.25763042], pot.units["acceleration"])
        got = pot.gradient(x, t=0)
        assert jnp.allclose(got, expect, atol=u.Q(1e-8, expect.unit))

    def test_density(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.density` method."""
        # TODO: fix negative density!!!
        got = pot.density(x, t=0)
        exp = u.Q(-4.90989768e-07, pot.units["mass density"])
        assert jnp.allclose(got, exp, atol=u.Q(1e-8, exp.unit))

    def test_hessian(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.hessian` method."""
        expected = u.Q(
            jnp.asarray(
                [
                    [-0.06747463, 0.03680435, 0.05520652],
                    [0.03680435, -0.01226812, 0.11041304],
                    [0.05520652, 0.11041304, 0.07974275],
                ]
            ),
            "1/Myr2",
        )
        assert jnp.allclose(pot.hessian(x, t=0), expected, atol=u.Q(1e-8, "1/Myr2"))

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        expect = u.Q(
            [
                [-0.06747463, 0.03680435, 0.05520652],
                [0.03680435, -0.01226812, 0.11041304],
                [0.05520652, 0.11041304, 0.07974275],
            ],
            pot.units["frequency drift"],
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Q(1e-8, expect.unit)
        )


##############################################################################


class TestNoTime:
    """Evaluating a potential without a time."""

    @pytest.fixture(scope="class")
    def static(self) -> gp.AbstractPotential:
        return gp.KeplerPotential(m_tot=u.Q(1e12, "Msun"), units="galactic")

    @pytest.fixture(scope="class")
    def tdep(self) -> gp.AbstractPotential:
        m_tot = gpp.LinearParameter(
            slope=u.Q(1e10, "Msun / Myr"),
            point_time=u.Q(0, "Myr"),
            point_value=u.Q(1e12, "Msun"),
        )
        return gp.KeplerPotential(m_tot=m_tot, units="galactic")

    @pytest.fixture(scope="class")
    def q(self) -> gt.QuSz3:
        return u.Q([8.0, 0.0, 0.0], "kpc")

    def test_is_time_dependent(
        self, static: gp.AbstractPotential, tdep: gp.AbstractPotential
    ) -> None:
        """Time dependence is found in parameters, components, and operators."""
        assert not static.is_time_dependent
        assert tdep.is_time_dependent

        # Composite: time-dependent if any component is.
        assert not gp.CompositePotential(a=static, b=static).is_time_dependent
        assert gp.CompositePotential(a=static, b=tdep).is_time_dependent

        # Transformed: time-independent operators ...
        rot = cx.ops.GalileanRotation.from_euler("z", u.Q(10, "deg"))
        shift = cx.ops.GalileanSpatialTranslation.from_([1, 0, 0], "kpc")
        for op in (rot, shift, rot | shift):
            assert not gp.TransformedPotential(static, op).is_time_dependent
        assert gp.TransformedPotential(tdep, rot).is_time_dependent
        # ... and time-dependent ones, also when nested in another operator.
        boost = cx.ops.GalileanBoost.from_([1, 0, 0], "km/s")
        spin = gc.ops.ConstantRotationZOperator(Omega_z=u.Q(10, "deg / Myr"))
        galilean = cx.ops.GalileanOperator(
            translation=cx.ops.GalileanTranslation.from_([0, 1, 0, 0], "kpc"),
            velocity=boost,
        )
        for op in (boost, spin, galilean, rot | boost):
            assert gp.TransformedPotential(static, op).is_time_dependent

        # Translated by a time-dependent translation.
        path_t = u.Q(jnp.linspace(0, 1, 10), "Gyr")
        delta = gpp.TimeDependentTranslationParameter.from_(
            path_t, u.Q(jnp.zeros((10, 3)), "kpc"), units=static.units
        )
        assert gp.TranslatedPotential(static, translation=delta).is_time_dependent

    def test_time_dependent_requires_time(
        self, tdep: gp.AbstractPotential, q: gt.QuSz3
    ) -> None:
        """A time-dependent potential cannot be evaluated without a time."""
        cq = cx.CartesianPos3D.from_(q)
        for pos in (
            q,
            q.value,
            cq,
            cx.vecs.KinematicSpace(length=cq),
            cx.Coordinate({"length": cq}, frame=gc.frames.simulation_frame),
            gc.PhaseSpacePosition(q=q, p=u.Q([0.0, 0, 0], "km/s")),
        ):
            with pytest.raises(TypeError, match="depends on time"):
                tdep.potential(pos)
        # `t=None` is the same as omitting it.
        with pytest.raises(TypeError, match="depends on time"):
            tdep.potential(q, None)

        # Under `jax.jit` it fails at trace time.
        with pytest.raises(TypeError, match="depends on time"):
            jax.jit(gp.gradient)(tdep, q)

    def test_own_time(self, tdep: gp.AbstractPotential, q: gt.QuSz3) -> None:
        """Inputs that carry a time use it."""
        t = u.Q(10, "Myr")
        exp = tdep.potential(q, t)
        cq = cx.CartesianPos3D.from_(q)
        for pos in (
            cx.FourVector(q=cq, t=t),
            gc.PhaseSpaceCoordinate(q=q, p=u.Q([0.0, 0, 0], "km/s"), t=t),
        ):
            assert jnp.allclose(tdep.potential(pos).value, exp.value), type(pos)

    def test_time_independent(self, static: gp.AbstractPotential, q: gt.QuSz3) -> None:
        """A time-independent potential can be evaluated without a time."""
        exp = static.potential(q, u.Q(3.7, "Gyr"))
        cq = cx.CartesianPos3D.from_(q)
        for pos in (
            q,
            cq,
            cx.vecs.KinematicSpace(length=cq),
            cx.Coordinate({"length": cq}, frame=gc.frames.simulation_frame),
            gc.PhaseSpacePosition(q=q, p=u.Q([0.0, 0, 0], "km/s")),
        ):
            assert jnp.allclose(static.potential(pos).value, exp.value), type(pos)
        assert jnp.allclose(static.potential(q, None).value, exp.value)

        # Also under `jax.jit`.
        got = jax.jit(gp.gradient)(static, q)
        assert jnp.allclose(got.value, static.gradient(q, u.Q(0, "Myr")).value)
        got = jax.jit(gp.gradient)(static, q.value)
        assert jnp.allclose(got, static.gradient(q.value, 0.0))

    def test_placeholder_dtype(self, static: gp.AbstractPotential) -> None:
        """The placeholder time follows the requested ``dtype``, else the position's."""
        from galax.potential._src.utils import parse_pot_to_xyz_t

        xyz = jnp.ones(3, dtype=jnp.float32)
        _, t = parse_pot_to_xyz_t(static, xyz, None, dtype=jnp.float32)
        assert t.dtype == jnp.float32
        _, t = parse_pot_to_xyz_t(static, xyz, None)
        assert t.dtype == jnp.float32
        _, t = parse_pot_to_xyz_t(static, u.Q(xyz, "kpc"), None)
        assert t.dtype == jnp.float32
        _, t = parse_pot_to_xyz_t(static, jnp.ones(3, dtype=int), None)
        assert t.dtype == jnp.result_type(float)
