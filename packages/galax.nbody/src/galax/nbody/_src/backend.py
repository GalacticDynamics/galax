"""The force-backend interface, plus the particle sets it acts on."""

__all__ = ["AbstractForceBackend", "Sources", "Targets"]

import abc

from jaxtyping import Array
from typing import Any

import equinox as eqx

from .gradpolicy import AbstractGradPolicy, Autodiff, as_grad_policy
from .particles import Sources, Targets


class AbstractForceBackend(eqx.Module):
    """Base class for N-body force backends.

    ``prepare`` turns sources into whatever the backend evaluates against, for
    example a tree. ``accelerations`` and ``potential`` evaluate it at targets
    and route the computation through the gradient policy ``grad``.
    """

    grad: AbstractGradPolicy = eqx.field(default=Autodiff(), converter=as_grad_policy)

    @abc.abstractmethod
    def prepare(self, sources: Sources) -> Any:
        """Return the prepared form of ``sources``."""

    @abc.abstractmethod
    def accelerations(
        self, prepared: Any, targets: Targets, *, G: float | Array
    ) -> Array:
        """Return the accelerations (Nt, 3) at ``targets``."""

    @abc.abstractmethod
    def potential(self, prepared: Any, targets: Targets, *, G: float | Array) -> Array:
        """Return the potentials (Nt,) at ``targets``."""
