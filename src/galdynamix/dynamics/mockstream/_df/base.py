"""galdynamix: Galactic Dynamix in Jax."""

from __future__ import annotations

__all__ = ["AbstractStreamDF"]

import abc
from typing import TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax
import jax.numpy as xp
import jax.typing as jt

from galdynamix.dynamics._orbit import Orbit
from galdynamix.dynamics.mockstream._core import MockStream
from galdynamix.potential._potential.base import AbstractPotentialBase
from galdynamix.utils import partial_jit

if TYPE_CHECKING:
    Wif: TypeAlias = tuple[jt.Array, jt.Array, jt.Array, jt.Array]
    Carry: TypeAlias = tuple[int, jt.Array, jt.Array, jt.Array, jt.Array]


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
        prog_mass: jt.Numeric,
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

        def scan_fn(carry: Carry, t: jt.Numeric) -> tuple[Carry, Wif]:
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
        w: jt.Array,
        prog_mass: jt.Numeric,
        t: jt.Numeric,
        *,
        i: int,
        seed_num: int,
    ) -> tuple[jt.Array, jt.Array, jt.Array, jt.Array]:
        """Generate stream particle initial conditions.

        Parameters
        ----------
        potential : AbstractPotentialBase
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
