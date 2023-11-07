from __future__ import annotations

__all__ = ["AbstractParameter", "ConstantParameter", "UserParameter"]

import abc
from dataclasses import KW_ONLY
from typing import Protocol

import astropy.units as u
import equinox as eqx
import jax.typing as jt

from galdynamix.utils import partial_jit


class ParameterCallable(Protocol):
    """Protocol for a Parameter callable."""

    def __call__(self, t: jt.Array) -> jt.Array:
        """Compute the parameter value at the given time(s).

        Parameters
        ----------
        t : Array
            Time(s) at which to compute the parameter value.

        Returns
        -------
        Array
            Parameter value(s) at the given time(s).
        """
        ...


# ====================================================================


class AbstractParameter(eqx.Module):  # type: ignore[misc]
    """Abstract Base Class for Parameters on a Potential.

    Parameters are time-dependent quantities that are used to define a
    Potential. They can be constant (see `ConstantParameter`), or they can be
    functions of time.

    Parameters
    ----------
    unit : Unit
        The output unit of the parameter.
    """

    _: KW_ONLY
    unit: u.Unit = eqx.field(static=True)  # TODO? move this to an annotation?

    @abc.abstractmethod
    def __call__(self, t: jt.Array) -> jt.Array:
        ...


class ConstantParameter(AbstractParameter):
    """Time-independent potential parameter."""

    # TODO: unit handling
    value: jt.Array

    @partial_jit()
    def __call__(self, t: jt.Array) -> jt.Array:
        return self.value


class UserParameter(AbstractParameter):
    """User-defined Parameter."""

    # TODO: unit handling
    func: ParameterCallable

    @partial_jit()
    def __call__(self, t: jt.Array) -> jt.Array:
        return self.func(t)
