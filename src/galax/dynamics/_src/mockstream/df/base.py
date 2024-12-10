"""Stream Distribution Functions for ejecting mock stream particles."""

__all__ = ["AbstractStreamDF"]

import abc
from functools import partial
from typing import TypeAlias

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray
from plum import convert

import coordinax as cx
from unxt import Quantity, uconvert

import galax.coordinates as gc
import galax.potential as gp
import galax.typing as gt
from .progenitor import ConstantMassProtenitor, ProgenitorMassCallable
from galax.dynamics._src.mockstream.core import MockStreamArm
from galax.dynamics._src.orbit import Orbit

Carry: TypeAlias = tuple[gt.LengthVec3, gt.SpeedVec3, gt.LengthVec3, gt.SpeedVec3]


class AbstractStreamDF(eqx.Module, strict=True):  # type: ignore[call-arg, misc]
    """Abstract base class of Stream Distribution Functions."""

    @partial(jax.jit)
    def sample(
        self,
        rng: PRNGKeyArray,
        # <\ parts of gala's ``prog_orbit``
        pot: gp.AbstractBasePotential,
        prog_orbit: Orbit,
        # />
        /,
        prog_mass: gt.MassScalar | ProgenitorMassCallable,
    ) -> gc.CompositePhaseSpacePosition:
        """Generate stream particle initial conditions.

        Parameters
        ----------
        rng : :class:`jaxtyping.PRNGKeyArray`, positional-only
            Pseudo-random number generator. Not split, used as is.
        pot : :class:`~galax.potential.AbstractBasePotential`, positional-only
            The potential of the host galaxy.
        prog_orbit : :class:`~galax.dynamics.Orbit`, positional-only
            The orbit of the progenitor.

        prog_mass : Quantity[float, (), 'mass'] | ProgenitorMassCallable
            Mass of the progenitor.

        Returns
        -------
        `galax.coordinates.CompositePhaseSpacePosition`
            Phase-space positions of the leading and trailing arms.

        Examples
        --------
        >>> import galax.coordinates as gc
        >>> import galax.dynamics as gd
        >>> import galax.potential as gp
        >>> import jax.random as jr

        >>> df = gd.FardalStreamDF()
        >>> pot = gp.MilkyWayPotential()
        >>> w = gc.PhaseSpacePosition(q=Quantity([8.3, 0, 0], "kpc"),
        ...                           p=Quantity([0, 220, 0], "km/s"),
        ...                           t=Quantity(0, "Gyr"))
        >>> prog_orbit = pot.evaluate_orbit(w, t=Quantity([0, 1, 2], "Gyr"))
        >>> stream_ic = df.sample(jr.key(0), pot, prog_orbit,
        ...                       prog_mass=Quantity(1e4, "Msun"))
        >>> stream_ic
        CompositePhaseSpacePosition({'lead': MockStreamArm(
            q=CartesianPos3D( ... ),
            p=CartesianVel3D( ... ),
            t=Quantity...,
            release_time=Quantity... ),
          'trail': MockStreamArm(
            q=CartesianPos3D( ... ),
            p=CartesianVel3D( ... ),
            t=Quantity...,
            release_time=Quantity...
        )})
        """
        # Progenitor positions and times. The orbit times are used as the
        # release times for the mock stream.
        prog_orbit = prog_orbit.vconvert(cx.CartesianPos3D)
        ts = prog_orbit.t

        # Progenitor mass
        mprog: ProgenitorMassCallable = (
            ConstantMassProtenitor(m_tot=prog_mass)
            if not callable(prog_mass)
            else prog_mass
        )

        x_lead, v_lead, x_trail, v_trail = self._sample(
            rng,
            pot,
            convert(prog_orbit.q, Quantity),
            convert(prog_orbit.p, Quantity),
            mprog(ts),
            ts,
        )

        ts = uconvert(pot.units["time"], ts)
        mock_lead = MockStreamArm(
            q=uconvert(pot.units["length"], x_lead),
            p=uconvert(pot.units["speed"], v_lead),
            t=ts,
            release_time=ts,
        )
        mock_trail = MockStreamArm(
            q=uconvert(pot.units["length"], x_trail),
            p=uconvert(pot.units["speed"], v_trail),
            t=ts,
            release_time=ts,
        )

        return gc.CompositePhaseSpacePosition(lead=mock_lead, trail=mock_trail)

    # TODO: keep units and PSP through this func
    @abc.abstractmethod
    def _sample(
        self,
        key: PRNGKeyArray,
        potential: gp.AbstractBasePotential,
        x: gt.LengthBatchableVec3,
        v: gt.SpeedBatchableVec3,
        prog_mass: gt.BatchableFloatQScalar,
        t: gt.BatchableFloatQScalar,
    ) -> tuple[
        gt.LengthBatchVec3, gt.SpeedBatchVec3, gt.LengthBatchVec3, gt.SpeedBatchVec3
    ]:
        """Generate stream particle initial conditions.

        Parameters
        ----------
        rng : :class:`jaxtyping.PRNGKeyArray`
            Pseudo-random number generator.
        potential : :class:`galax.potential.AbstractBasePotential`
            The potential of the host galaxy.
        x : Quantity[float, (*#batch, 3), "length"]
            3d position (x, y, z)
        v : Quantity[float, (*#batch, 3), "speed"]
            3d velocity (v_x, v_y, v_z)
        prog_mass : Quantity[float, (*#batch), "mass"]
            Mass of the progenitor.
        t : Quantity[float, (*#batch), "time"]
            The release time of the stream particles.

        Returns
        -------
        x_lead, v_lead: Quantity[float, (*batch, 3), "length" | "speed"]
            Position and velocity of the leading arm.
        x_trail, v_trail : Quantity[float, (*batch, 3), "length" | "speed"]
            Position and velocity of the trailing arm.
        """
        ...
