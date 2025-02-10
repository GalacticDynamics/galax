"""Fields for mass evolution."""

__all__ = [
    "MassVectorField",
    "AbstractMassField",
    "UserMassField",
    "ConstantMass",
]

from abc import abstractmethod
from dataclasses import KW_ONLY
from typing import Any, Protocol, TypeAlias, TypedDict, runtime_checkable

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from galax.dynamics._src.fields import AbstractField

Time: TypeAlias = Any
ClusterMass: TypeAlias = Any


class Args(TypedDict, total=False):
    units: Any
    # Add other optional keys here if needed


@runtime_checkable
class MassVectorField(Protocol):
    """Protocol for mass vector field.

    This is a function that returns the derivative of the mass vector with
    respect to time.

    Examples
    --------
    >>> from galax.dynamics.cluster import MassVectorField

    >>> def mass_deriv(t, Mc, args, **kwargs): pass

    >>> isinstance(mass_deriv, MassVectorField)
    True

    """

    def __call__(
        self, t: Time, Mc: ClusterMass, args: Args, /, **kwargs: Any
    ) -> Array: ...


class AbstractMassField(AbstractField):
    """ABC for mass fields.

    Methods
    -------
    __call__ : `galax.dynamics.cluster.MassVectorField`
        Compute the mass field.
    terms : the `diffrax.AbstractTerm` `jaxtyping.PyTree` for integration.

    """

    @abstractmethod
    def __call__(self, t: Time, Mc: ClusterMass, args: Args, /, **kwargs: Any) -> Array:  # type: ignore[override]
        raise NotImplementedError  # pragma: no cover

    @AbstractField.terms.dispatch  # type: ignore[misc]
    def terms(
        self: "AbstractMassField", _: dfx.AbstractSolver, /
    ) -> PyTree[dfx.AbstractTerm]:
        """Return diffeq terms for integration.

        Examples
        --------
        >>> import diffrax as dfx
        >>> import galax.dynamics as gd

        >>> field = gd.cluster.ConstantMass()
        >>> field.terms(dfx.Dopri8())
        ODETerm(
            vector_field=_JitWrapper( fn='ConstantMass.__call__', ... ) )

        """
        return dfx.ODETerm(eqx.filter_jit(self.__call__))


#####################################################


class UserMassField(AbstractMassField):
    """User-defined mass field.

    This takes a user-defined function of type
    `galax.dynamics.cluster.MassVectorField`.

    """

    #: User-defined mass derivative function of type
    #: `galax.dynamics.cluster.MassVectorField`
    mass_deriv: MassVectorField

    _: KW_ONLY

    def __call__(self, t: Time, Mc: ClusterMass, args: Args, /, **kwargs: Any) -> Array:  # type: ignore[override]
        return self.mass_deriv(t, Mc, args, **kwargs)


#####################################################


class ConstantMass(AbstractMassField):
    """Constant mass field.

    This is a constant mass field.

    """

    def __call__(self, t: Time, Mc: ClusterMass, args: Args, /, **kwargs: Any) -> Array:  # type: ignore[override]  # noqa: ARG002
        return jnp.zeros_like(Mc)
