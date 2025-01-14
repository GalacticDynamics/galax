"""Dynamics Solvers.

This is private API.

"""

__all__ = ["AbstractSolver"]


import abc
from typing import Any

import equinox as eqx


class AbstractSolver(eqx.Module, strict=True):  # type: ignore[call-arg,misc]
    """ABC for solvers.

    Notes
    -----
    The ``init``, ``step``, and ``solve`` methods are abstract and should be
    implemented by subclasses.

    """

    @abc.abstractmethod
    def init(self, *args: Any, **kwargs: Any) -> Any:
        """Initialize the solver."""
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, *args: Any, **kwargs: Any) -> Any:
        """Step the solver."""
        raise NotImplementedError

    @abc.abstractmethod
    def solve(self, *args: Any, **kwargs: Any) -> Any:
        """Solve, initializing and stepping to the solution."""
        raise NotImplementedError
