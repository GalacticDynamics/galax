__all__ = ["AbstractCompositePotential", "CompositePotential"]


import uuid
from collections.abc import Hashable, Mapping
from dataclasses import KW_ONLY
from functools import partial
from types import MappingProxyType
from typing import Any, ClassVar, TypeVar, final

import equinox as eqx
import jax
from plum import dispatch

import quaxed.numpy as jnp
from unxt import AbstractUnitSystem, Quantity, unitsystem
from xmmutablemap import ImmutableMap
from zeroth import zeroth

import galax.typing as gt
from .base import AbstractPotentialBase, default_constants
from .params.attr import CompositeParametersAttribute

K = TypeVar("K")
V = TypeVar("V")


# Note: cannot have `strict=True` because of inheriting from ImmutableMap.
class AbstractCompositePotential(
    AbstractPotentialBase,
    ImmutableMap[str, AbstractPotentialBase],  # type: ignore[misc]
    strict=False,
):
    def __init__(
        self,
        potentials: (
            dict[str, AbstractPotentialBase]
            | tuple[tuple[str, AbstractPotentialBase], ...]
        ) = (),
        /,
        *,
        units: Any = None,
        constants: Any = default_constants,
        **kwargs: AbstractPotentialBase,
    ) -> None:
        ImmutableMap.__init__(self, potentials, **kwargs)  # <- ImmutableMap.__init__

        # __post_init__ stuff:
        # Check that all potentials have the same unit system
        units_ = units if units is not None else zeroth(self.values()).units
        usys = unitsystem(units_)
        if not all(p.units == usys for p in self.values()):
            msg = "all potentials must have the same unit system"
            raise ValueError(msg)
        object.__setattr__(self, "units", usys)  # TODO: not call `object.__setattr__`

        # TODO: some similar check that the same constants are the same, e.g.
        #       `G` is the same for all potentials. Or use `constants` to update
        #       the `constants` of every potential (before `super().__init__`)
        object.__setattr__(self, "constants", constants)

        # Apply the unit system to any parameters.
        self._apply_unitsystem()

    # === Potential ===

    @partial(jax.jit, inline=True)
    def _potential(  # TODO: inputs w/ units
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.SpecificEnergyBatchScalar:
        return jnp.sum(
            jnp.asarray(
                [p._potential(q, t) for p in self.values()]  # noqa: SLF001
            ),
            axis=0,
        )

    ###########################################################################
    # Composite potentials

    def __or__(self, other: Any) -> "CompositePotential":
        if not isinstance(other, AbstractPotentialBase):
            return NotImplemented

        return CompositePotential(  # combine the two dictionaries
            self._data
            | (  # make `other` into a compatible dictionary.
                other._data
                if isinstance(other, CompositePotential)
                else {str(uuid.uuid4()): other}
            )
        )

    def __ror__(self, other: Any) -> "CompositePotential":
        if not isinstance(other, AbstractPotentialBase):
            return NotImplemented

        return CompositePotential(  # combine the two dictionaries
            (  # make `other` into a compatible dictionary.
                other._data
                if isinstance(other, CompositePotential)
                else {str(uuid.uuid4()): other}
            )
            | self._data
        )

    def __add__(self, other: AbstractPotentialBase) -> "CompositePotential":
        return self | other


# =================


@dispatch(precedence=1)
def replace(
    obj: AbstractCompositePotential, /, **kwargs: Any
) -> AbstractCompositePotential:
    """Replace the parameters of a composite potential.

    Examples
    --------
    >>> from dataclassish import replace
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.CompositePotential(
    ...     disk=gp.MiyamotoNagaiPotential(m_tot=Quantity(1e11, "Msun"), a=6.5, b=0.26, units="galactic"),
    ...     halo=gp.NFWPotential(m=Quantity(1e12, "Msun"), r_s=20, units="galactic"),
    ... )

    >>> new_pot = replace(pot, disk=gp.MiyamotoNagaiPotential(m_tot=Quantity(1e12, "Msun"), a=6.5, b=0.26, units="galactic"))
    >>> new_pot["disk"].m_tot.value
    Quantity['mass'](Array(1.e+12, dtype=float64), unit='solMass')

    """  # noqa: E501
    # TODO: directly call the Mapping implementation
    extra_keys = set(kwargs) - set(obj)
    kwargs = eqx.error_if(kwargs, any(extra_keys), "invalid keys {extra_keys}.")

    return type(obj)(**{**obj, **kwargs})


@dispatch(precedence=1)
def replace(
    obj: AbstractCompositePotential,
    replacements: Mapping[str, Mapping[str, Any]],
    /,
) -> AbstractCompositePotential:
    """Replace the parameters of a composite potential.

    Examples
    --------
    >>> from dataclassish import replace
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.CompositePotential(
    ...     disk=gp.MiyamotoNagaiPotential(m_tot=Quantity(1e11, "Msun"), a=6.5, b=0.26, units="galactic"),
    ...     halo=gp.NFWPotential(m=Quantity(1e12, "Msun"), r_s=20, units="galactic"),
    ... )

    >>> new_pot = replace(pot, {"disk": {"m_tot": Quantity(1e12, "Msun")}})
    >>> new_pot["disk"].m_tot.value
    Quantity['mass'](Array(1.e+12, dtype=float64), unit='solMass')

    """  # noqa: E501
    # AbstractCompositePhaseSpacePosition is both a Mapping and a dataclass
    # so we need to disambiguate the method to call
    method = replace.invoke(Mapping[Hashable, Any], Mapping[str, Any])
    return method(obj, replacements)


###########################################################################


@final
class CompositePotential(AbstractCompositePotential):
    """Composite Potential."""

    parameters: ClassVar = CompositeParametersAttribute(MappingProxyType({}))

    _data: dict[str, AbstractPotentialBase]
    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(init=False, static=True, converter=unitsystem)
    constants: ImmutableMap[str, Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )
