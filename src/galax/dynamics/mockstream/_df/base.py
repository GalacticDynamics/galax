"""galax: Galactic Dynamix in Jax."""


__all__ = ["AbstractStreamDF"]

import abc
from functools import partial
from typing import TypeAlias

import equinox as eqx
import jax
import jax.experimental.array_api as xp
from jax.numpy import copy

from galax.dynamics._orbit import Orbit
from galax.dynamics.mockstream._core import MockStream
from galax.potential._potential.base import AbstractPotentialBase
from galax.typing import BatchVec3, FloatScalar, IntLike, Vec3, Vec6

Wif: TypeAlias = tuple[Vec3, Vec3, Vec3, Vec3]
Carry: TypeAlias = tuple[IntLike, Vec3, Vec3, Vec3, Vec3]

w_org = xp.asarray([0.0, 0.0, 0.0])


class AbstractStreamDF(eqx.Module, strict=True):  # type: ignore[call-arg, misc]
    """Abstract Base Class of Stream Distribution Functions."""

    lead: bool = eqx.field(default=True, static=True)
    trail: bool = eqx.field(default=True, static=True)

    def __post_init__(self) -> None:
        if not self.lead and not self.trail:
            msg = "You must generate either leading or trailing tails (or both!)"
            raise ValueError(msg)

    @partial(jax.jit, static_argnames=("seed_num",))
    def sample(
        self,
        # <\ parts of gala's ``prog_orbit``
        pot: AbstractPotentialBase,
        prog_orbit: Orbit,
        # />
        /,
        prog_mass: FloatScalar,
        *,
        seed_num: int,
    ) -> tuple[MockStream, MockStream]:
        """Generate stream particle initial conditions.

        Parameters
        ----------
        pot : AbstractPotentialBase, positional-only
            The potential of the host galaxy.
        prog_orbit : Orbit, positional-only
            The orbit of the progenitor.
        prog_mass : Numeric
            Mass of the progenitor in [Msol].
            TODO: allow this to be an array or function of time.

        seed_num : int, keyword-only
            PRNG seed

        Returns
        -------
        mock_lead, mock_trail : MockStream
            Positions and velocities of the leading and trailing tails.
        """
        # Progenitor positions and times. The orbit times are used as the
        # release times for the mock stream.
        prog_w = prog_orbit.w()
        ts = prog_orbit.t

        # Scan over the release times to generate the stream particle initial
        # conditions at each release time.
        def scan_fn(carry: Carry, t: FloatScalar) -> tuple[Carry, Wif]:
            i = carry[0]
            out = self._sample(pot, prog_w[i], prog_mass, t, i=i, seed_num=seed_num)
            return (i + 1, *out), out

        # TODO: use ``jax.vmap`` instead of ``jax.lax.scan`` for GPU usage
        init_carry = (0, copy(w_org), copy(w_org), copy(w_org), copy(w_org))
        x_lead, x_trail, v_lead, v_trail = jax.lax.scan(scan_fn, init_carry, ts)[1]

        mock_lead = MockStream(x_lead, v_lead, t=ts, release_time=ts)
        mock_trail = MockStream(x_trail, v_trail, t=ts, release_time=ts)

        return mock_lead, mock_trail

    @abc.abstractmethod
    def _sample(
        self,
        pot: AbstractPotentialBase,
        w: Vec6,
        prog_mass: FloatScalar,
        t: FloatScalar,
        *,
        i: IntLike,
        seed_num: int,
    ) -> tuple[BatchVec3, BatchVec3, BatchVec3, BatchVec3]:
        """Generate stream particle initial conditions.

        Parameters
        ----------
        pot : AbstractPotentialBase
            The potential of the host galaxy.
        w : Array
            6d position (x, y, z) [kpc], (v_x, v_y, v_z) [kpc/Myr]
        prog_mass : Numeric
            Mass of the progenitor in [Msol]
        t : Numeric
            Time in [Myr]

        i : int
            PRNG multiplier
        seed_num : int
            PRNG seed

        Returns
        -------
        x_lead, x_trail, v_lead, v_trail : Array
            Positions and velocities of the leading and trailing tails.
        """
        ...
