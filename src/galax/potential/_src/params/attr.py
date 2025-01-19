"""Descriptor for a Parameters attributes."""

__all__ = [
    "AbstractParametersAttribute",
    "ParametersAttribute",
    "CompositeParametersAttribute",
]

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NoReturn, final

from .field import ParameterField

if TYPE_CHECKING:
    import galax.potential  # noqa: ICN001


@dataclass(frozen=True, slots=True)
class AbstractParametersAttribute:
    """Mapping of the :class:`~galax.potential.ParameterField` values.

    Examples
    --------
    The normal usage of this class is the ``parameters`` attribute of
    :class:`~galax.potential.AbstractPotential`.

    >>> import galax.potential as gp
    >>> gp.KeplerPotential.parameters
    mappingproxy({'m_tot': ParameterField(...)})

    >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")
    >>> pot.parameters
    mappingproxy({'m_tot': ConstantParameter(Quantity['mass'](Array(1.e+12, dtype=float64, ...), unit='solMass'))})

    """  # noqa: E501

    parameters: "MappingProxyType[str, ParameterField]"  # TODO: specify type hint
    """Class attribute name on Potential."""

    _name: str = field(init=False)
    """The name of the descriptor on the containing class."""

    def __set_name__(self, _: Any, name: str) -> None:
        object.__setattr__(self, "_name", name)

    def __set__(self, instance: Any, _: Any) -> NoReturn:
        msg = f"cannot set {self._name!r} of {instance!r}."
        raise AttributeError(msg)


@final
@dataclass(frozen=True, slots=True)
class ParametersAttribute(AbstractParametersAttribute):
    """Mapping of the :class:`~galax.potential.ParameterField` values.

    If accessed from the :class:`~galax.potential.AbstractPotential` class,
    this returns a mapping of the :class:`~galax.potential.AbstractParameter`
    objects themselves.  If accessed from an instance, this returns a mapping of
    the values of the Parameters.

    This class is used to implement
    :obj:`galax.potential.AbstractPotential.parameters`.

    Examples
    --------
    The normal usage of this class is the ``parameters`` attribute of
    :class:`~galax.potential.AbstractPotential`.

    >>> import unxt as u
    >>> import galax.potential as gp

    >>> gp.KeplerPotential.parameters
    mappingproxy({'m_tot': ParameterField(...)})

    >>> kepler = gp.KeplerPotential(m_tot=1e12, units="galactic")
    >>> kepler.parameters
    mappingproxy({'m_tot': ConstantParameter(Quantity['mass'](Array(1.e+12, dtype=float64, ...), unit='solMass'))})

    """  # noqa: E501

    def __get__(
        self,
        instance: "galax.potential.AbstractPotential | None",
        owner: "type[galax.potential.AbstractPotential] | None",
    ) -> "MappingProxyType[str, ParameterField]":
        # Called from the class
        if instance is None:
            return self.parameters
        # Called from the instance
        return MappingProxyType({n: getattr(instance, n) for n in self.parameters})


@final
@dataclass(frozen=True, slots=True)
class CompositeParametersAttribute(AbstractParametersAttribute):
    """Mapping of the :class:`~galax.potential.ParameterField` values.

    If accessed from the :class:`~galax.potential.CompositePotential` class,
    this returns a mapping of the :class:`~galax.potential.AbstractParameter`
    objects themselves.  If accessed from an instance, this returns a mapping of
    the values of the Parameters.

    This class is used to implement
    :obj:`galax.potential.CompositePotential.parameters`.

    Examples
    --------
    The normal usage of this class is the ``parameters`` attribute of
    :class:`~galax.potential.AbstractPotential`.

    >>> import unxt as u
    >>> import galax.potential as gp

    >>> gp.CompositePotential.parameters
    mappingproxy({})

    >>> kepler = gp.KeplerPotential(m_tot=1e12, units="galactic")
    >>> composite = gp.CompositePotential(kepler=kepler)
    >>> composite.parameters
    mappingproxy({'kepler':
        mappingproxy({'m_tot':
            ConstantParameter(Quantity['mass'](Array(1.e+12, dtype=float64, ...), unit='solMass'))})})

    """  # noqa: E501

    def __get__(
        self,
        instance: "galax.potential.AbstractCompositePotential | None",
        owner: "type[galax.potential.AbstractCompositePotential] | None",
    ) -> "MappingProxyType[str, ParameterField]":
        # Called from the class
        if instance is None:
            return self.parameters
        # Called from the instance
        return MappingProxyType({k: p.parameters for k, p in instance.items()})
