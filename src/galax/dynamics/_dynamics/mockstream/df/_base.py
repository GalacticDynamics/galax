"""galax: Galactic Dynamix in Jax."""

__all__ = ["AbstractStreamDF", "ProgenitorMassCallable", "ConstantMassProtenitor"]

import abc
from functools import partial
from typing import Protocol, TypeAlias, runtime_checkable

import equinox as eqx
import jax
import quax.examples.prng as jr
from jaxtyping import Shaped

import quaxed.array_api as xp
from unxt import Quantity

from galax.dynamics._dynamics.mockstream.core import MockStream
from galax.dynamics._dynamics.orbit import Orbit
from galax.potential._potential.base import AbstractPotentialBase
from galax.typing import BatchVec3, FloatQScalar, Vec3

Wif: TypeAlias = tuple[Vec3, Vec3, Vec3, Vec3]
Carry: TypeAlias = tuple[int, jr.PRNG, Vec3, Vec3, Vec3, Vec3]


@runtime_checkable
class ProgenitorMassCallable(Protocol):
    def __call__(
        self, t: Shaped[Quantity["time"], "*shape"], /
    ) -> Shaped[Quantity["mass"], "*shape"]: ...


class ConstantMassProtenitor(eqx.Module):  # type: ignore[misc]
    m: Shaped[Quantity["mass"], ""]

    def __call__(
        self, t: Shaped[Quantity["time"], "*shape"], /
    ) -> Shaped[Quantity["mass"], "*shape"]:
        return xp.ones(t.shape) * self.m


class AbstractStreamDF(eqx.Module, strict=True):  # type: ignore[call-arg, misc]
    """Abstract base class of Stream Distribution Functions."""

    lead: bool = eqx.field(default=True, static=True)
    trail: bool = eqx.field(default=True, static=True)

    def __post_init__(self) -> None:
        if not self.lead and not self.trail:
            msg = "You must generate either leading or trailing tails (or both!)"
            raise ValueError(msg)

    @partial(jax.jit)
    def sample(
        self,
        rng: jr.PRNG,
        # <\ parts of gala's ``prog_orbit``
        pot: AbstractPotentialBase,
        prog_orbit: Orbit,
        # />
        /,
        prog_mass: Shaped[Quantity["mass"], ""] | ProgenitorMassCallable,
    ) -> tuple[MockStream, MockStream]:
        """Generate stream particle initial conditions.

        Parameters
        ----------
        rng : `quax.examples.prng.PRNG`
            Pseudo-random number generator.
        pot : AbstractPotentialBase, positional-only
            The potential of the host galaxy.
        prog_orbit : Orbit, positional-only
            The orbit of the progenitor.

        prog_mass : Quantity[float, (), 'mass'] | ProgenitorMassCallable
            Mass of the progenitor.

        Returns
        -------
        mock_lead, mock_trail : MockStream
            Positions and velocities of the leading and trailing tails.
        """
        # Progenitor positions and times. The orbit times are used as the
        # release times for the mock stream.
        prog_w = prog_orbit.w(units=pot.units)  # TODO: keep as PSP
        x, v = prog_w[..., 0:3], prog_w[..., 3:6]
        ts = prog_orbit.t

        mprog: ProgenitorMassCallable = (
            ConstantMassProtenitor(m=prog_mass)
            if not callable(prog_mass)
            else prog_mass
        )

        # Scan over the release times to generate the stream particle initial
        # conditions at each release time.
        def scan_fn(carry: Carry, t: FloatQScalar) -> tuple[Carry, Wif]:
            i = carry[0]
            rng, subrng = carry[1].split(2)
            out = self._sample(subrng, pot, x[i], v[i], mprog(t), t)
            return (i + 1, rng, *out), out

        # TODO: use ``jax.vmap`` instead of ``jax.lax.scan`` for GPU usage
        init_carry = (0, rng, xp.zeros(3), xp.zeros(3), xp.zeros(3), xp.zeros(3))
        x_lead, x_trail, v_lead, v_trail = jax.lax.scan(scan_fn, init_carry, ts)[1]

        mock_lead = MockStream(
            q=Quantity(x_lead, pot.units["length"]),
            p=Quantity(v_lead, pot.units["speed"]),
            t=ts,
            release_time=ts,
        )
        mock_trail = MockStream(
            q=Quantity(x_trail, pot.units["length"]),
            p=Quantity(v_trail, pot.units["speed"]),
            t=ts,
            release_time=ts,
        )

        return mock_lead, mock_trail

    # TODO: keep units and PSP through this func
    @abc.abstractmethod
    def _sample(
        self,
        rng: jr.PRNG,
        pot: AbstractPotentialBase,
        x: Vec3,
        v: Vec3,
        prog_mass: FloatQScalar,
        t: FloatQScalar,
    ) -> tuple[BatchVec3, BatchVec3, BatchVec3, BatchVec3]:
        """Generate stream particle initial conditions.

        Parameters
        ----------
        rng : `quax.examples.prng.PRNG`
            Pseudo-random number generator.
        pot : AbstractPotentialBase
            The potential of the host galaxy.
        w : Array
            6d position (x, y, z) [kpc], (v_x, v_y, v_z) [kpc/Myr]
        prog_mass : Numeric
            Mass of the progenitor in [Msol]
        t : Quantity[float, (), "time"]
            The release time of the stream particles.

        Returns
        -------
        x_lead, x_trail, v_lead, v_trail : Array
            Positions and velocities of the leading and trailing tails.
        """
        ...
