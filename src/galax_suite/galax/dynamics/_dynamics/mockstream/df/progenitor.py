"""galax: Galactic Dynamix in Jax."""

__all__ = ["ProgenitorMassCallable", "ConstantMassProtenitor"]

from typing import Protocol, runtime_checkable

import equinox as eqx

import quaxed.array_api as xp
from unxt import Quantity

import galax.typing as gt


@runtime_checkable
class ProgenitorMassCallable(Protocol):
    """Callable that returns the progenitor mass at the given times."""

    def __call__(self, t: gt.TimeBatchScalar, /) -> gt.MassBatchScalar:
        """Return the progenitor mass at the times.

        Parameters
        ----------
        t : TimeBatchScalar
            The times at which to evaluate the progenitor mass.
        """
        ...


class ConstantMassProtenitor(eqx.Module):  # type: ignore[misc]
    """Progenitor mass callable that returns a constant mass.

    Parameters
    ----------
    m : Quantity[float, (), 'mass']
        The progenitor mass.
    """

    m_tot: gt.MassScalar = eqx.field(converter=Quantity["mass"].constructor)
    """The progenitor mass."""

    def __call__(self, t: gt.TimeBatchScalar, /) -> gt.MassBatchScalar:
        """Return the constant mass at the times.

        Parameters
        ----------
        t : TimeBatchScalar
            The times at which to evaluate the progenitor mass.
        """
        return xp.ones(t.shape) * self.m_tot
