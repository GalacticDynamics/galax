"""Dynamics Solvers.

This is private API.

"""

__all__ = ["AbstractDynamicsField", "HamiltonianField"]

import abc
from functools import partial
from typing import Any

import equinox as eqx
import jax
from plum import dispatch

import unxt as u
from unxt.quantity import UncheckedQuantity as FastQ

import galax.potential as gp
import galax.typing as gt


class AbstractDynamicsField(eqx.Module, strict=True):  # type: ignore[misc,call-arg]
    """ABC for dynamics fields."""

    @abc.abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError  # pragma: no cover


class HamiltonianField(AbstractDynamicsField, strict=True):  # type: ignore[call-arg]
    r"""Dynamics field for Hamiltonian EoM.

    This is for Hamilton's equations for motion for a particle in a potential.

    .. math::

            \\dot{q} = \frac{dH}{dp} \\ \\dot{p} = -\frac{dH}{dq}

    .. note::

        Calling this object in a jit context will provide a significant speedup.

    """

    #: Potential.
    potential: gp.AbstractBasePotential

    @property
    def units(self) -> u.AbstractUnitSystem:
        return self.potential.units

    # -----------------------
    # Call dispatches
    # TODO: not require unit munging

    @dispatch
    @partial(jax.jit, inline=True)
    def __call__(
        self: "HamiltonianField",
        t: gt.TimeBatchableScalar,
        qp: gt.BatchableQP,
        args: tuple[Any, ...],  # noqa: ARG002
        /,
    ) -> gt.BatchPAarr:
        """Call with Quantities."""
        units = self.units
        a = -self.potential._gradient(qp[0], t).ustrip(units["acceleration"])  # noqa: SLF001
        return qp[1].ustrip(units["speed"]), a

    @dispatch
    @partial(jax.jit, inline=True)
    def __call__(
        self: "HamiltonianField",
        t: gt.BatchableFloatScalar,
        qp: gt.BatchableQParr,
        args: tuple[Any, ...],  # noqa: ARG002
        /,
    ) -> gt.BatchPAarr:
        """Call with arrays."""
        units = self.units
        a = -self.potential._gradient(  # noqa: SLF001
            FastQ(qp[0], units["length"]),
            FastQ(t, units["time"]),
        ).ustrip(units["acceleration"])
        return qp[1], a
