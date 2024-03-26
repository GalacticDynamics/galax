"""galax: Galactic Dynamix in Jax."""

__all__ = ["ProgenitorMassCallable", "ConstantMassProtenitor"]

from typing import Protocol, runtime_checkable

import equinox as eqx
from jaxtyping import Shaped

import quaxed.array_api as xp
from unxt import Quantity

TimeBatchScalar: TypeAlias = Shaped[Quantity["time"], "*batch"]
MassBatchScalar: TypeAlias = Shaped[Quantity["mass"], "*batch"]


@runtime_checkable
class ProgenitorMassCallable(Protocol):
    """Callable that returns the progenitor mass at the given times."""

    def __call__(self, t: TimeBatchScalar, /) -> MassBatchScalar:
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

    m: Shaped[Quantity["mass"], ""] = eqx.field(converter=Quantity["mass"].constructor)
    """The progenitor mass."""

    def __call__(self, t: TimeBatchScalar, /) -> MassBatchScalar:
        """Return the constant mass at the times.

        Parameters
        ----------
        t : TimeBatchScalar
            The times at which to evaluate the progenitor mass.
        """
        return xp.ones(t.shape) * self.m
