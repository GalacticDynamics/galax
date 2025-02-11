"""Fields."""

__all__ = ["AbstractField"]

from abc import abstractmethod
from typing import Any

import diffrax as dfx
import equinox as eqx
from jaxtyping import PyTree
from plum import dispatch

import diffraxtra as dfxtra


class AbstractField(eqx.Module, strict=True):  # type: ignore[misc,call-arg]
    """Abstract base class for fields.

    Methods
    -------
    - `__call__` : evaluates the field.
    - `terms` : returns the `diffrax.AbstractTerm` wrapper of the `jax.jit`-ed
      ``__call__`` for integration with `diffrax.diffeqsolve`. `terms` takes as
      input the `diffrax.AbstractSolver` object (or something that wraps it,
      like a `diffraxtra.DiffEqSolver`), to determine the correct
      `diffrax.AbstractTerm` and its `jaxtyping.PyTree` structure.

    Examples
    --------
    >>> import diffrax as dfx
    >>> import unxt as u
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    Define a Hamiltonian field:

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    Evaluate the field at a given coordinate:

    >>> field(u.Quantity(0, "Gyr"), u.Quantity([8., 0, 0], "kpc"), u.Quantity([0, 22, 0], "km/s"))
    (Array([0.        , 0.02249967, 0.        ], dtype=float64, ...),
     Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    For integration with `diffrax.diffeqsolve` the ``terms`` method returns the
    correctly-structured `diffrax.AbstractTerm` `jaxtyping.PyTree`. The term is
    a wrapper around the ``__call__`` method, which is `jax.jit`-ed for
    performance.

    >>> field.terms(dfx.Dopri8())
    ODETerm(vector_field=<wrapped function __call__>)

    >>> field.terms(dfx.SemiImplicitEuler())
    (ODETerm( ... ), ODETerm( ... ))

    """  # noqa: E501

    @abstractmethod
    def __call__(self, t: Any, *args: Any, **kwargs: Any) -> Any:
        """Evaluate the field at time `t`."""
        raise NotImplementedError  # pragma: no cover

    @dispatch.abstract
    def terms(self, solver: Any, /) -> PyTree[dfx.AbstractTerm]:
        """Return the `diffrax.AbstractTerm` `jaxtyping.PyTree` for integration."""
        raise NotImplementedError  # pragma: no cover


# ==================================================================


@AbstractField.terms.dispatch  # type: ignore[misc]
def terms(
    self: AbstractField, wrapper: dfxtra.DiffEqSolver, /
) -> PyTree[dfx.AbstractTerm]:
    """Return diffeq terms, redispatching to the solver.

    Examples
    --------
    >>> import diffrax as dfx
    >>> import unxt as u
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> solver = gd.solve.DiffEqSolver(dfx.Dopri8())

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> field.terms(solver)
    ODETerm(vector_field=<wrapped function __call__>)

    """
    return self.terms(wrapper.solver)
