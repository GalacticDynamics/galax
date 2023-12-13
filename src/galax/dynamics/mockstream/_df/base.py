"""galax: Galactic Dynamix in Jax."""


__all__ = ["AbstractStreamDF"]

import abc
from typing import TypeAlias

import equinox as eqx
import jax
import jax.numpy as xp

from galax.dynamics._orbit import Orbit
from galax.dynamics.mockstream._core import MockStream
from galax.potential._potential.base import AbstractPotentialBase
from galax.typing import BatchVec3, FloatScalar, IntLike, Vec3, Vec6
from galax.utils import partial_jit

Wif: TypeAlias = tuple[Vec3, Vec3, Vec3, Vec3]
Carry: TypeAlias = tuple[IntLike, Vec3, Vec3, Vec3, Vec3]


class AbstractStreamDF(eqx.Module):  # type: ignore[misc]
    lead: bool = eqx.field(default=True, static=True)
    trail: bool = eqx.field(default=True, static=True)

    def __post_init__(self) -> None:
        if not self.lead and not self.trail:
            msg = "You must generate either leading or trailing tails (or both!)"
            raise ValueError(msg)

    @partial_jit(static_argnames=("seed_num",))
    def sample(
        self,
        # <\ parts of gala's ``prog_orbit``
        potential: AbstractPotentialBase,
        prog_orbit: Orbit,
        # />
        prog_mass: FloatScalar,
        *,
        seed_num: int,
    ) -> tuple[MockStream, MockStream]:
        """Generate stream particle initial conditions.

        Parameters
        ----------
        potential : AbstractPotentialBase
            The potential of the host galaxy.
        prog_orbit : Orbit
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
        prog_qps = prog_orbit.qp
        ts = prog_orbit.t

        def scan_fn(carry: Carry, t: FloatScalar) -> tuple[Carry, Wif]:
            i = carry[0]
            output = self._sample(
                potential,
                prog_qps[i],
                prog_mass,
                t,
                i=i,
                seed_num=seed_num,
            )
            return (i + 1, *output), tuple(output)

        init_carry = (
            0,
            xp.array([0.0, 0.0, 0.0]),
            xp.array([0.0, 0.0, 0.0]),
            xp.array([0.0, 0.0, 0.0]),
            xp.array([0.0, 0.0, 0.0]),
        )
        x_lead, x_trail, v_lead, v_trail = jax.lax.scan(scan_fn, init_carry, ts[1:])[1]

        mock_lead = MockStream(x_lead, v_lead, ts[1:])
        mock_trail = MockStream(x_trail, v_trail, ts[1:])

        return mock_lead, mock_trail

    @abc.abstractmethod
    def _sample(
        self,
        potential: AbstractPotentialBase,
        qp: Vec6,
        prog_mass: FloatScalar,
        t: FloatScalar,
        *,
        i: IntLike,
        seed_num: int,
    ) -> tuple[BatchVec3, BatchVec3, BatchVec3, BatchVec3]:
        """Generate stream particle initial conditions.

        Parameters
        ----------
        potential : AbstractPotentialBase
            The potential of the host galaxy.
        qp : Array
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
