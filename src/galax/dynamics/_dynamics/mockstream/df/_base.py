"""galax: Galactic Dynamix in Jax."""

__all__ = ["AbstractStreamDF"]

import abc
from functools import partial
from typing import TypeAlias

import equinox as eqx
import jax
import quax.examples.prng as jr
from plum import convert

import coordinax as cx
import quaxed.array_api as xp
from unxt import Quantity

import galax.typing as gt
from ._progenitor import ConstantMassProtenitor, ProgenitorMassCallable
from galax.dynamics._dynamics.mockstream.core import MockStream, MockStreamArm
from galax.dynamics._dynamics.orbit import Orbit
from galax.potential import AbstractPotentialBase

Wif: TypeAlias = tuple[gt.LengthVec3, gt.LengthVec3, gt.SpeedVec3, gt.SpeedVec3]
Carry: TypeAlias = tuple[
    int, jr.PRNG, gt.LengthVec3, gt.LengthVec3, gt.SpeedVec3, gt.SpeedVec3
]


class AbstractStreamDF(eqx.Module, strict=True):  # type: ignore[call-arg, misc]
    """Abstract base class of Stream Distribution Functions."""

    lead: bool = eqx.field(default=True, static=True)
    trail: bool = eqx.field(default=True, static=True)

    def __check_init__(self) -> None:
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
        prog_mass: gt.MassScalar | ProgenitorMassCallable,
    ) -> MockStream:
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
        `galax.dynamics.MockStream`
            Phase-space positions of the leading and trailing arms.

        Examples
        --------
        >>> import galax.coordinates as gc
        >>> import galax.dynamics as gd
        >>> import galax.potential as gp
        >>> import quax.examples.prng as jr

        >>> df = gd.FardalStreamDF()
        >>> pot = gp.MilkyWayPotential()
        >>> w = gc.PhaseSpacePosition(q=Quantity([8.3, 0, 0], "kpc"),
        ...                           p=Quantity([0, 220, 0], "km/s"),
        ...                           t=Quantity(0, "Gyr"))
        >>> prog_orbit = pot.evaluate_orbit(w, t=Quantity([0, 1, 2], "Gyr"))
        >>> stream_ic = df.sample(jr.ThreeFry(0), pot, prog_orbit,
        ...                       prog_mass=Quantity(1e4, "Msun"))
        """
        # Progenitor positions and times. The orbit times are used as the
        # release times for the mock stream.
        prog_orbit = prog_orbit.represent_as(cx.CartesianPosition3D)
        x = convert(prog_orbit.q, Quantity)
        v = convert(prog_orbit.p, Quantity)
        ts = prog_orbit.t

        # Progenitor mass
        mprog: ProgenitorMassCallable = (
            ConstantMassProtenitor(m_tot=prog_mass)
            if not callable(prog_mass)
            else prog_mass
        )

        # Scan over the release times to generate the stream particle initial
        # conditions at each release time.
        def scan_fn(carry: Carry, t: gt.FloatQScalar) -> tuple[Carry, Wif]:
            i = carry[0]
            rng, subrng = carry[1].split(2)
            out = self._sample(subrng, pot, x[i], v[i], mprog(t), t)
            return (i + 1, rng, *out), out

        # TODO: use ``jax.vmap`` instead of ``jax.lax.scan`` for GPU usage
        init_carry = (
            0,
            rng,
            xp.zeros_like(x[0]),
            xp.zeros_like(x[0]),
            xp.zeros_like(v[0]),
            xp.zeros_like(v[0]),
        )
        x_lead, x_trail, v_lead, v_trail = jax.lax.scan(scan_fn, init_carry, ts)[1]

        mock_lead = MockStreamArm(
            q=x_lead.to_units(pot.units["length"]),
            p=v_lead.to_units(pot.units["speed"]),
            t=ts.to_units(pot.units["time"]),
            release_time=ts.to_units(pot.units["time"]),
        )
        mock_trail = MockStreamArm(
            q=x_trail.to_units(pot.units["length"]),
            p=v_trail.to_units(pot.units["speed"]),
            t=ts.to_units(pot.units["time"]),
            release_time=ts.to_units(pot.units["time"]),
        )

        return MockStream(lead=mock_lead, trail=mock_trail)

    # TODO: keep units and PSP through this func
    @abc.abstractmethod
    def _sample(
        self,
        rng: jr.PRNG,
        pot: AbstractPotentialBase,
        x: gt.LengthVec3,
        v: gt.SpeedVec3,
        prog_mass: gt.FloatQScalar,
        t: gt.FloatQScalar,
    ) -> tuple[
        gt.LengthBatchVec3, gt.LengthBatchVec3, gt.SpeedBatchVec3, gt.SpeedBatchVec3
    ]:
        """Generate stream particle initial conditions.

        Parameters
        ----------
        rng : `quax.examples.prng.PRNG`
            Pseudo-random number generator.
        pot : AbstractPotentialBase
            The potential of the host galaxy.
        x : Quantity[float, (3,), "length"]
            3d position (x, y, z)
        v : Quantity[float, (3,), "speed"]
            3d velocity (v_x, v_y, v_z)
        prog_mass : Quantity[float, (), "mass"]
            Mass of the progenitor.
        t : Quantity[float, (), "time"]
            The release time of the stream particles.

        Returns
        -------
        x_lead, x_trail : Quantity[float, (*shape, 3), "length"]
            Positions of the leading and trailing tails.
        v_lead, v_trail : Quantity[float, (*shape, 3), "speed"]
            Velocities of the leading and trailing tails.
        """
        ...
