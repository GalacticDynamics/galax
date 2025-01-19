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
import unxt as u

import galax.coordinates as gc
import galax.potential as gp
import galax.typing as gt
from .progenitor import ConstantMassProtenitor, ProgenitorMassCallable
from galax.dynamics._src.mockstream.core import MockStreamArm
from galax.dynamics._src.orbit import Orbit

Carry: TypeAlias = tuple[gt.LengthSz3, gt.SpeedSz3, gt.LengthSz3, gt.SpeedSz3]


class AbstractStreamDF(eqx.Module, strict=True):  # type: ignore[call-arg, misc]
    """Abstract base class of Stream Distribution Functions."""

    @partial(jax.jit)
    def sample(
        self,
        rng: PRNGKeyArray,
        # <\ parts of gala's ``prog_orbit``
        pot: gp.AbstractPotential,
        prog_orbit: Orbit,
        # />
        /,
        prog_mass: gt.MassSz0 | ProgenitorMassCallable,
    ) -> gc.CompositePhaseSpacePosition:
        """Generate stream particle initial conditions.

        Parameters
        ----------
        rng : :class:`jaxtyping.PRNGKeyArray`, positional-only
            Pseudo-random number generator. Not split, used as is.
        pot : :class:`~galax.potential.AbstractPotential`, positional-only
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
        >>> w = gc.PhaseSpacePosition(q=u.Quantity([8.3, 0, 0], "kpc"),
        ...                           p=u.Quantity([0, 220, 0], "km/s"),
        ...                           t=u.Quantity(0, "Gyr"))
        >>> prog_orbit = pot.evaluate_orbit(w, t=u.Quantity([0, 1, 2], "Gyr"))
        >>> stream_ic = df.sample(jr.key(0), pot, prog_orbit,
        ...                       prog_mass=u.Quantity(1e4, "Msun"))
        >>> stream_ic
        CompositePhaseSpacePosition({'lead': MockStreamArm(
            q=CartesianPos3D( ... ),
            p=CartesianVel3D( ... ),
            t=Quantity...,
            release_time=Quantity...,
            frame=SimulationFrame() ),
          'trail': MockStreamArm(
            q=CartesianPos3D( ... ),
            p=CartesianVel3D( ... ),
            t=Quantity...,
            release_time=Quantity...,
            frame=SimulationFrame()
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
            convert(prog_orbit.q, u.Quantity),
            convert(prog_orbit.p, u.Quantity),
            mprog(ts),
            ts,
        )

        ts = u.uconvert(pot.units["time"], ts)
        mock_lead = MockStreamArm(
            q=u.uconvert(pot.units["length"], x_lead),
            p=u.uconvert(pot.units["speed"], v_lead),
            t=ts,
            release_time=ts,
            frame=prog_orbit.frame,
        )
        mock_trail = MockStreamArm(
            q=u.uconvert(pot.units["length"], x_trail),
            p=u.uconvert(pot.units["speed"], v_trail),
            t=ts,
            release_time=ts,
            frame=prog_orbit.frame,
        )

        return gc.CompositePhaseSpacePosition(
            lead=mock_lead,
            trail=mock_trail,
            frame=prog_orbit.frame,
        )

    # TODO: keep units and PSP through this func
    @abc.abstractmethod
    def _sample(
        self,
        key: PRNGKeyArray,
        potential: gp.AbstractPotential,
        x: gt.LengthBBtSz3,
        v: gt.SpeedBBtSz3,
        prog_mass: gt.BBtFloatQuSz0,
        t: gt.BBtFloatQuSz0,
    ) -> tuple[gt.LengthBtSz3, gt.SpeedBtSz3, gt.LengthBtSz3, gt.SpeedBtSz3]:
        """Generate stream particle initial conditions.

        Parameters
        ----------
        rng : :class:`jaxtyping.PRNGKeyArray`
            Pseudo-random number generator.
        potential : :class:`galax.potential.AbstractPotential`
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
