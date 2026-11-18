"""Source and target particle sets for N-body force backends."""

__all__ = ["AbstractForceBackend", "Sources", "Targets"]

import abc

from jaxtyping import Array
from typing import Any

import equinox as eqx

from .gradpolicy import AbstractGradPolicy, Autodiff, as_grad_policy


class Sources(eqx.Module):
    """Particles that exert force.

    Positions ``q`` (Ns, 3), masses ``m`` and softenings ``h`` (Ns,).
    """

    q: Array
    m: Array
    h: Array


class Targets(eqx.Module):
    """Particles that feel force: positions ``q`` (Nt, 3) and softenings ``h`` (Nt,).

    ``self_index`` (Nt,) gives each target's index among the sources, or -1 if
    the target is not a source.
    """

    q: Array
    h: Array
    self_index: Array


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
    def accelerations(self, prepared: Any, targets: Targets, *, G: float) -> Array:
        """Return the accelerations (Nt, 3) at ``targets``."""

    @abc.abstractmethod
    def potential(self, prepared: Any, targets: Targets, *, G: float) -> Array:
        """Return the potentials (Nt,) at ``targets``."""
