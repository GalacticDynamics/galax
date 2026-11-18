"""Gradient policies: how gradients flow through a force backend."""

__all__ = ["AbstractGradPolicy", "Autodiff", "NoGrad", "StopGrad", "as_grad_policy"]

import abc
import operator

from collections.abc import Callable
from jaxtyping import Array
from typing import Any, TypeVar

import equinox as eqx
import jax

from .backend import Sources, Targets

_T = TypeVar("_T", Sources, Targets)
_FIELDS = frozenset({"q", "m", "h"})
_NOGRAD_MSG = (
    "This force backend has gradient policy NoGrad, so it can't be "
    "differentiated. Use Autodiff or StopGrad."
)


class AbstractGradPolicy(eqx.Module):
    """Decides how gradients flow through ``fn(sources, targets)``."""

    @abc.abstractmethod
    def apply(
        self,
        fn: Callable[[Sources, Targets], Array],
        sources: Sources,
        targets: Targets,
    ) -> Array:
        """Evaluate ``fn(sources, targets)`` under this policy."""


class Autodiff(AbstractGradPolicy):
    """Differentiate straight through ``fn``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from galax.nbody._src.backend import Sources, Targets
    >>> from galax.nbody._src.gradpolicy import Autodiff
    >>> s = Sources(q=jnp.zeros((1, 3)), m=jnp.ones(1), h=jnp.zeros(1))
    >>> t = Targets(q=jnp.ones((1, 3)), h=jnp.zeros(1), self_index=jnp.array([-1]))
    >>> float(Autodiff().apply(lambda s, t: t.q.sum() * s.m.sum(), s, t))
    3.0

    """

    def apply(
        self,
        fn: Callable[[Sources, Targets], Array],
        sources: Sources,
        targets: Targets,
    ) -> Array:
        return fn(sources, targets)


class StopGrad(AbstractGradPolicy):
    """Stop gradients through the named fields (``q``, ``m``, ``h``).

    The fields are those of both sources and targets that have them.
    """

    wrt: frozenset[str] = eqx.field(default=_FIELDS, static=True)

    def __check_init__(self) -> None:
        bad = set(self.wrt) - _FIELDS
        if bad:
            msg = (
                f"wrt must be a subset of {sorted(_FIELDS)}, got unknown {sorted(bad)}"
            )
            raise ValueError(msg)

    def apply(
        self,
        fn: Callable[[Sources, Targets], Array],
        sources: Sources,
        targets: Targets,
    ) -> Array:
        def stop(obj: _T) -> _T:
            for name in self.wrt & set(obj.__dataclass_fields__):
                value = jax.lax.stop_gradient(getattr(obj, name))
                obj = eqx.tree_at(operator.attrgetter(name), obj, value)
            return obj

        return fn(stop(sources), stop(targets))


class NoGrad(AbstractGradPolicy):
    """Forward evaluation only; differentiating raises ``NotImplementedError``."""

    def apply(
        self,
        fn: Callable[[Sources, Targets], Array],
        sources: Sources,
        targets: Targets,
    ) -> Array:
        @jax.custom_jvp
        def run(s: Sources, t: Targets) -> Array:
            return fn(s, t)

        @run.defjvp
        def _(
            primals: tuple[Any, ...], tangents: tuple[Any, ...]
        ) -> tuple[Array, Array]:
            raise NotImplementedError(_NOGRAD_MSG)

        return run(sources, targets)


_POLICIES: dict[str, type[AbstractGradPolicy]] = {
    "autodiff": Autodiff,
    "stop": StopGrad,
    "none": NoGrad,
}


def as_grad_policy(x: "str | AbstractGradPolicy") -> AbstractGradPolicy:
    """Return ``x`` if it is a policy, else build one from its string name.

    Examples
    --------
    >>> from galax.nbody._src.gradpolicy import as_grad_policy
    >>> as_grad_policy("none")
    NoGrad()

    """
    if isinstance(x, AbstractGradPolicy):
        return x
    try:
        return _POLICIES[x]()
    except KeyError:
        msg = f"unknown gradient policy {x!r}; valid strings are {sorted(_POLICIES)}"
        raise ValueError(msg) from None
