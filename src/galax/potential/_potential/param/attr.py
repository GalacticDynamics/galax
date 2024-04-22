"""Descriptor for a Parameters attributes."""

__all__ = ["ParametersAttribute"]

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NoReturn, final

from .field import ParameterField

if TYPE_CHECKING:
    from galax.potential._potential.base import AbstractPotentialBase


@final
@dataclass(frozen=True, slots=True)
class ParametersAttribute:
    """Mapping of the :class:`~galax.potential.ParameterField` values.

    If accessed from the :class:`~galax.potential.AbstractPotentialBase` class,
    this returns a mapping of the :class:`~galax.potential.AbstractParameter`
    objects themselves.  If accessed from an instance, this returns a mapping of
    the values of the Parameters.

    This class is used to implement
    :obj:`galax.potential.AbstractPotentialBase.parameters`.

    Examples
    --------
    The normal usage of this class is the ``parameters`` attribute of
    :class:`~galax.potential.AbstractPotentialBase`.

    >>> from galax.potential import KeplerPotential

    >>> KeplerPotential.parameters
    mappingproxy({'m_tot': ParameterField(...)})

    >>> import astropy.units as u
    >>> kepler = KeplerPotential(m_tot=1e12 * u.solMass, units="galactic")
    >>> kepler.parameters
    mappingproxy({'m_tot': ConstantParameter(
        unit=Unit("solMass"), value=Quantity[...](value=f64[], unit=Unit("solMass"))
        )})
    """

    parameters: "MappingProxyType[str, ParameterField]"  # TODO: specify type hint
    """Class attribute name on Potential."""

    _name: str = field(init=False)
    """The name of the descriptor on the containing class."""

    def __set_name__(self, _: Any, name: str) -> None:
        object.__setattr__(self, "_name", name)

    def __get__(
        self,
        instance: "AbstractPotentialBase | None",
        owner: "type[AbstractPotentialBase] | None",
    ) -> "MappingProxyType[str, ParameterField]":
        # Called from the class
        if instance is None:
            return self.parameters
        # Called from the instance
        return MappingProxyType({n: getattr(instance, n) for n in self.parameters})

    def __set__(self, instance: Any, _: Any) -> NoReturn:
        msg = f"cannot set {self._name!r} of {instance!r}."
        raise AttributeError(msg)
