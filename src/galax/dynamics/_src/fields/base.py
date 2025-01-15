"""Dynamics Solvers.

This is private API.

"""

__all__ = ["AbstractDynamicsField"]

import abc
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
from jaxtyping import PyTree
from plum import dispatch

import unxt as u

from galax.dynamics._src.utils import DiffEqSolver


class AbstractDynamicsField(eqx.Module, strict=True):  # type: ignore[misc,call-arg]
    """ABC for dynamics fields.

    Note that this provides a default implementation for the `terms` property,
    which is a jitted `diffrax.ODETerm` object. This is a convenience for the
    user and may be overridden, e.g. to support an SDE or other differential
    equation types.

    """

    units: eqx.AbstractVar[u.AbstractUnitSystem]

    @abc.abstractmethod
    def __call__(
        self, t: Any, qp: tuple[Any, Any], args: tuple[Any, ...], /
    ) -> tuple[Any, Any]:
        raise NotImplementedError  # pragma: no cover

    @dispatch
    def terms(
        self: "AbstractDynamicsField",
        solver: dfx.AbstractSolver,  # noqa: ARG002
        /,
    ) -> PyTree[dfx.AbstractTerm]:
        """Return the AbstractTerm PyTree for integration with `diffrax`.

        Examples
        --------
        >>> import diffrax as dfx
        >>> import unxt as u
        >>> import galax.potential as gp
        >>> import galax.dynamics as gd

        >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")
        >>> field = gd.fields.HamiltonianField(pot)

        >>> solver = dfx.Dopri8()

        >>> field.terms(solver)
        ODETerm(vector_field=<wrapped function __call__>)

        """
        return dfx.ODETerm(jax.jit(self.__call__))


# =========================================================
# Interface with AbstractDynamicsField


@AbstractDynamicsField.terms.dispatch  # type: ignore[misc,attr-defined]
def terms(
    self: AbstractDynamicsField, wrapper: DiffEqSolver, /
) -> PyTree[dfx.AbstractTerm]:
    """Return diffeq terms, redispatching to the solver.

    Examples
    --------
    >>> import diffrax as dfx
    >>> import unxt as u
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> solver = gd.integrate.DiffEqSolver(dfx.Dopri8())

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> field.terms(solver)
    ODETerm(vector_field=<wrapped function __call__>)

    """
    return self.terms(wrapper.solver)
