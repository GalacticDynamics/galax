"""Dynamics Solvers.

This is private API.

"""

__all__ = ["AbstractDynamicsField"]

import abc
from typing import Any

import diffrax
import equinox as eqx
import jax
from jaxtyping import PyTree


class AbstractDynamicsField(eqx.Module, strict=True):  # type: ignore[misc,call-arg]
    """ABC for dynamics fields.

    Note that this provides a default implementation for the `terms` property,
    which is a jitted `diffrax.ODETerm` object. This is a convenience for the
    user and may be overridden, e.g. to support an SDE or other differential
    equation types.

    """

    @abc.abstractmethod
    def __call__(
        self, t: Any, qp: tuple[Any, Any], args: tuple[Any, ...], /
    ) -> tuple[Any, Any]:
        raise NotImplementedError  # pragma: no cover

    @property
    def terms(self) -> PyTree[diffrax.AbstractTerm]:
        """Return the AbstractTerm."""
        return diffrax.ODETerm(jax.jit(self.__call__))
