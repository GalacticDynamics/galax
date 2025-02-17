"""Stream Distribution Functions for ejecting mock stream particles."""

__all__ = ["AbstractStreamDF"]

import abc
from typing import TypeAlias

import equinox as eqx
from jaxtyping import PRNGKeyArray

import galax._custom_types as gt
import galax.potential as gp

Carry: TypeAlias = tuple[gt.QuSz3, gt.QuSz3, gt.QuSz3, gt.QuSz3]


class AbstractStreamDF(eqx.Module, strict=True):  # type: ignore[call-arg, misc]
    """Abstract base class of Stream Distribution Functions."""

    # TODO: keep units and PSP through this func
    @abc.abstractmethod
    def sample(
        self,
        key: PRNGKeyArray,
        potential: gp.AbstractPotential,
        x: gt.BBtQuSz3,
        v: gt.BBtQuSz3,
        prog_mass: gt.BBtFloatQuSz0,
        t: gt.BBtFloatQuSz0,
    ) -> tuple[gt.BtQuSz3, gt.BtQuSz3, gt.BtQuSz3, gt.BtQuSz3]:
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
