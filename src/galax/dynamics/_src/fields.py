"""Dynamics Solvers.

This is private API.

"""

__all__ = ["AbstractDynamicsField", "HamiltonianField"]

import abc
from functools import partial
from typing import Any

import diffrax
import equinox as eqx
import jax
from plum import dispatch

import unxt as u
from unxt.quantity import UncheckedQuantity as FastQ

import galax.coordinates as gc
import galax.potential as gp
import galax.typing as gt


class AbstractDynamicsField(eqx.Module, strict=True):  # type: ignore[misc,call-arg]
    """ABC for dynamics fields."""

    @abc.abstractmethod
    def __call__(
        self, t: Any, qp: tuple[Any, Any], args: tuple[Any, ...], /
    ) -> tuple[Any, Any]:
        raise NotImplementedError  # pragma: no cover

    @property
    @abc.abstractmethod
    def terms(self) -> diffrax.AbstractTerm:
        # TODO: should this be concrete?
        raise NotImplementedError  # pragma: no cover


##############################################################################


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

    @dispatch.abstract
    def __call__(
        self, t: Any, qp: tuple[Any, Any], args: tuple[Any, ...], /
    ) -> tuple[Any, Any]:
        raise NotImplementedError  # pragma: no cover

    @property
    def terms(self) -> diffrax.ODETerm:
        """Return the ODE term."""
        return diffrax.ODETerm(jax.jit(self.__call__))


# ---------------------------
# Call dispatches


@HamiltonianField.__call__.dispatch
@partial(jax.jit, inline=True)
def call(
    self: HamiltonianField,
    t: gt.TimeBatchableScalar,
    qp: gt.BatchableQP,
    args: tuple[Any, ...],  # noqa: ARG001
    /,
) -> gt.BatchPAarr:
    """Call with Quantities."""
    # TODO: not require unit munging
    units = self.units
    a = -self.potential._gradient(qp[0], t).ustrip(units["acceleration"])  # noqa: SLF001
    return qp[1].ustrip(units["speed"]), a


@HamiltonianField.__call__.dispatch
@partial(jax.jit, inline=True)
def call(
    self: HamiltonianField,
    t: gt.BatchableFloatScalar,
    qp: gt.BatchableQParr,
    args: tuple[Any, ...],  # noqa: ARG001
    /,
) -> gt.BatchPAarr:
    """Call with arrays."""
    # TODO: not require unit munging
    units = self.units
    a = -self.potential._gradient(  # noqa: SLF001
        FastQ(qp[0], units["length"]),
        FastQ(t, units["time"]),
    ).ustrip(units["acceleration"])
    return qp[1], a


@HamiltonianField.__call__.dispatch
@partial(jax.jit, inline=True)
def call(
    self: HamiltonianField,
    t: gt.BatchableFloatScalar,
    qp: gt.BatchableVec6,
    args: tuple[Any, ...],
    /,
) -> gt.BatchPAarr:
    """Call with 6-vector."""
    return self(t, (qp[0:3], qp[3:6]), args)


@HamiltonianField.__call__.dispatch
@partial(jax.jit, inline=True)
def call(
    self: HamiltonianField,
    t: gt.BatchableFloatScalar,
    w: gc.AbstractBasePhaseSpacePosition,
    args: tuple[Any, ...],
    /,
) -> gt.BatchPAarr:
    """Call with PhaseSpacePosition."""
    return self(t, w._qp(units=self.units), args)  # noqa: SLF001
