"""galdynamix: Galactic Dynamix in Jax"""
# ruff: noqa: F403

from __future__ import annotations

__all__ = ["AbstractStreamDF"]

import abc
from typing import TYPE_CHECKING, Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as xp
import jax.typing as jt

from galdynamix.potential._potential.base import AbstractPotentialBase
from galdynamix.utils import partial_jit

if TYPE_CHECKING:
    _wifT: TypeAlias = tuple[jt.Array, jt.Array, jt.Array, jt.Array]
    _carryT: TypeAlias = tuple[int, jt.Array, jt.Array, jt.Array, jt.Array]


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
        prog_ws: jt.Array,
        ts: jt.Numeric,
        # />
        prog_mass: jt.Numeric,
        *,
        seed_num: int,
    ) -> tuple[jt.Array, jt.Array, jt.Array, jt.Array]:
        """Generate stream particle initial conditions.

        Parameters
        ----------
        potential : AbstractPotentialBase
            The potential of the host galaxy.
        prog_ws : Array[(N, 6), float]
            Columns are (x, y, z) [kpc], (v_x, v_y, v_z) [kpc/Myr]
            Rows are at times `ts`.
        prog_mass : Numeric
            Mass of the progenitor in [Msol].
            TODO: allow this to be an array or function of time.
        ts : Numeric
            Times in [Myr]

        seed_num : int, keyword-only
            PRNG seed

        Returns
        -------
        x_lead, x_trail, v_lead, v_trail : Array
            Positions and velocities of the leading and trailing tails.
        """

        def scan_fn(carry: _carryT, t: Any) -> tuple[_carryT, _wifT]:
            i = carry[0]
            output = self._sample(
                potential,
                prog_ws[i, :3],
                prog_ws[i, 3:],
                prog_mass,
                i,
                t,
                seed_num=seed_num,
            )
            return (i + 1, *output), tuple(output)  # type: ignore[return-value]

        init_carry = (
            0,
            xp.array([0.0, 0.0, 0.0]),
            xp.array([0.0, 0.0, 0.0]),
            xp.array([0.0, 0.0, 0.0]),
            xp.array([0.0, 0.0, 0.0]),
        )
        x_lead, x_trail, v_lead, v_trail = jax.lax.scan(scan_fn, init_carry, ts[1:])[1]
        return x_lead, x_trail, v_lead, v_trail

    @abc.abstractmethod
    def _sample(
        self,
        potential: AbstractPotentialBase,
        x: jt.Array,
        v: jt.Array,
        prog_mass: jt.Numeric,
        i: int,
        t: jt.Numeric,
        *,
        seed_num: int,
    ) -> tuple[jt.Array, jt.Array, jt.Array, jt.Array]:
        """Generate stream particle initial conditions.

        Parameters
        ----------
        potential : AbstractPotentialBase
            The potential of the host galaxy.
        x : Array
            3d position (x, y, z) in [kpc]
        v : Array
            3d velocity (v_x, v_y, v_z) in [kpc/Myr]
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
