"""galax: Galactic Dynamix in Jax."""

__all__ = ["ProgenitorMassCallable", "ConstantMassProtenitor"]

from typing import Protocol, runtime_checkable

import equinox as eqx

import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt


@runtime_checkable
class ProgenitorMassCallable(Protocol):
    """Callable that returns the progenitor mass at the given times."""

    def __call__(self, t: gt.TimeBtSz0, /) -> gt.MassBtSz0:
        """Return the progenitor mass at the times.

        Parameters
        ----------
        t : TimeBtSz0
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

    m_tot: gt.MassSz0 = eqx.field(converter=u.Quantity["mass"].from_)
    """The progenitor mass."""

    def __call__(self, t: gt.TimeBtSz0, /) -> gt.MassBtSz0:
        """Return the constant mass at the times.

        Parameters
        ----------
        t : TimeBtSz0
            The times at which to evaluate the progenitor mass.
        """
        return jnp.ones(t.shape) * self.m_tot
