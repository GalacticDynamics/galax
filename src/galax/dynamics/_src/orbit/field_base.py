"""Dynamics fields.

This is private API.

"""

__all__ = ["AbstractOrbitField"]

from typing import Any
from typing_extensions import override

import diffrax as dfx
import jax
from jaxtyping import Array, PyTree
from plum import dispatch

from galax.dynamics._src.fields import AbstractField
from galax.dynamics._src.utils import parse_to_t_y


class AbstractOrbitField(AbstractField, strict=True):  # type: ignore[call-arg]
    """ABC for fields for computing orbits.

    Note that this provides a default implementation for the `terms` property,
    which is a jitted `diffrax.ODETerm` object. This is a convenience for the
    user and may be overridden, e.g. to support an SDE or other differential
    equation types.

    """

    @override  # specify the signature of the `__call__` method.
    @dispatch.abstract
    def __call__(self, *_: Any, **kw: Any) -> tuple[Any, Any]:
        raise NotImplementedError  # pragma: no cover

    @AbstractField.parse_inputs.dispatch  # type: ignore[misc]
    def parse_inputs(
        self, *args: Any, ustrip: bool = True, **kwargs: Any
    ) -> tuple[Array, PyTree[Array]]:
        """Parse inputs for the field.

        TODO: consolidate with ``parse_to_t_y``.

        """
        del ustrip
        t0, y0 = parse_to_t_y(None, *args, ustrip=self.units, **kwargs)
        return t0, y0


@AbstractField.terms.dispatch  # type: ignore[misc]
def terms(
    self: "AbstractOrbitField", _: dfx.AbstractSolver, /
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
