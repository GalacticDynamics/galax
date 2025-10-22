"""Composite Potential."""

__all__ = ["CompositePotential"]


from dataclasses import KW_ONLY
from types import MappingProxyType
from typing import Any, ClassVar, TypeAlias, cast, final

import equinox as eqx

import unxt as u
from xmmutablemap import ImmutableMap
from zeroth import zeroth

from .base import AbstractPotential, default_constants
from .base_multi import AbstractCompositePotential
from .params.attr import CompositeParametersAttribute

ArgPotential: TypeAlias = (
    dict[str, AbstractPotential] | tuple[tuple[str, AbstractPotential], ...]
)


class UnitsOptionEnum(eqx.Enumeration):  # type: ignore[misc]
    """Enum for units option."""

    FIRST = "first"


@final
class CompositePotential(
    AbstractCompositePotential,
    ImmutableMap[str, AbstractPotential],  # type: ignore[misc]
):
    """Composite Potential.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.CompositePotential(
    ...     bulge=gp.HernquistPotential(m_tot=1e11, r_s=2, units="galactic"),
    ...     disk=gp.KuzminPotential(m_tot=1e12, r_s=10, units="galactic"),
    ...     halo=gp.NFWPotential(m=1e12, r_s=10, units="galactic"),
    ... )

    >>> pot.keys()
    dict_keys(['bulge', 'disk', 'halo'])

    >>> "halo" in pot
    True

    >>> len(pot)
    3

    >>> pot["disk"]
    KuzminPotential(
      units=..., constants=ImmutableMap({'G': ...}),
      m_tot=ConstantParameter(...),
      r_s=ConstantParameter(...)
    )

    """

    parameters: ClassVar = CompositeParametersAttribute(MappingProxyType({}))

    _data: dict[str, AbstractPotential]
    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(static=True, converter=u.unitsystem)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    def __init__(
        self,
        potentials: ArgPotential = (),
        /,
        *,
        units: Any = UnitsOptionEnum.FIRST,
        constants: Any = default_constants,
        **kwargs: AbstractPotential,
    ) -> None:
        ImmutableMap.__init__(self, potentials, **kwargs)  # <- ImmutableMap.__init__

        # __post_init__ stuff:
        # Check that all potentials have the same unit system
        usys = (
            zeroth(self.values()).units
            if units is UnitsOptionEnum.FIRST
            else u.unitsystem(units)
        )
        if not all(p.units == usys for p in self.values()):
            msg = "all potentials must have the same unit system"
            raise ValueError(msg)
        object.__setattr__(self, "units", usys)  # TODO: not call `object.__setattr__`

        # Constants
        object.__setattr__(
            self,
            "constants",
            ImmutableMap({k: v.decompose(usys) for k, v in constants.items()}),
        )

        # Apply the unit system to any parameters.
        self._apply_unitsystem()

    def __repr__(self) -> str:  # TODO: not need this hack
        return cast(str, ImmutableMap.__repr__(self))
