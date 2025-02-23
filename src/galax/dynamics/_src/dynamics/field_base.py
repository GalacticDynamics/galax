"""Dynamics fields.

This is private API.

"""

__all__ = ["AbstractDynamicsField"]

from typing import Any
from typing_extensions import override

import diffrax as dfx
import equinox as eqx
import jax
from jaxtyping import PyTree
from plum import dispatch

import unxt as u

from galax.dynamics._src.fields import AbstractField


class AbstractDynamicsField(AbstractField, strict=True):  # type: ignore[call-arg]
    """ABC for dynamics fields.

    Note that this provides a default implementation for the `terms` property,
    which is a jitted `diffrax.ODETerm` object. This is a convenience for the
    user and may be overridden, e.g. to support an SDE or other differential
    equation types.

    """

    #: unit system of the field.
    units: eqx.AbstractVar[u.AbstractUnitSystem]

    @override  # specify the signature of the `__call__` method.
    @dispatch.abstract
    def __call__(self, *_: Any, **kw: Any) -> tuple[Any, Any]:
        raise NotImplementedError  # pragma: no cover


@AbstractField.terms.dispatch  # type: ignore[misc]
def terms(
    self: "AbstractDynamicsField", _: dfx.AbstractSolver, /
) -> PyTree[dfx.AbstractTerm]:
    """Return the `diffrax.AbstractTerm` `jaxtyping.PyTree` for integration.

    Examples
    --------
    >>> import diffrax as dfx
    >>> import unxt as u
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> solver = dfx.Dopri8()

    >>> field.terms(solver)
    ODETerm(vector_field=<wrapped function __call__>)

    """
    return dfx.ODETerm(jax.jit(self.__call__))
