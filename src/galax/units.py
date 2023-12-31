"""Tools for representing systems of units using ``astropy.units``."""

__all__ = [
    "UnitSystem",
    "DimensionlessUnitSystem",
    "galactic",
    "dimensionless",
    "solarsystem",
]

from collections.abc import Iterator
from typing import Any, ClassVar, no_type_check

import astropy.units as u
from astropy.units.physical import _physical_unit_mapping


@no_type_check  # TODO: get beartype working with this
class UnitSystem:
    """Represents a system of units."""

    _core_units: list[u.UnitBase]
    _registry: dict[u.PhysicalType, u.UnitBase]

    _required_dimensions: ClassVar[list[u.PhysicalType]] = [
        u.get_physical_type("length"),
        u.get_physical_type("time"),
        u.get_physical_type("mass"),
        u.get_physical_type("angle"),
    ]

    # TODO: type hint `units`
    def __init__(self, units: Any, *args: u.UnitBase) -> None:
        if isinstance(units, UnitSystem):
            if len(args) > 0:
                msg = "If passing in a UnitSystem, cannot pass in additional units."
                raise ValueError(msg)

            self._registry = units._registry.copy()  # noqa: SLF001
            self._core_units = units._core_units  # noqa: SLF001
            return

        units = (units, *args)

        self._registry = {}
        for unit in units:
            unit_ = (  # TODO: better detection of allowed unit base classes
                unit if isinstance(unit, u.UnitBase) else u.def_unit(f"{unit!s}", unit)
            )
            if unit_.physical_type in self._registry:
                msg = f"Multiple units passed in with type {unit_.physical_type!r}"
                raise ValueError(msg)
            self._registry[unit_.physical_type] = unit_

        self._core_units = []
        for phys_type in self._required_dimensions:
            if phys_type not in self._registry:
                msg = f"You must specify a unit for the physical type {phys_type!r}"
                raise ValueError(msg)
            self._core_units.append(self._registry[phys_type])

    def __getitem__(self, key: str | u.PhysicalType) -> u.UnitBase:
        if key in self._registry:
            return self._registry[key]

        unit = None
        for k, v in _physical_unit_mapping.items():
            if v == key:
                unit = u.Unit(" ".join([f"{x}**{y}" for x, y in k]))
                break

        if unit is None:
            msg = f"Physical type '{key}' doesn't exist in unit registry."
            raise ValueError(msg)

        unit = unit.decompose(self._core_units)
        unit._scale = 1.0  # noqa: SLF001
        return unit

    def __len__(self) -> int:
        return len(self._core_units)

    def __iter__(self) -> Iterator[u.UnitBase]:
        yield from self._core_units

    def __repr__(self) -> str:
        return f"UnitSystem({', '.join(str(uu) for uu in self._core_units)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnitSystem):
            return NotImplemented
        return bool(self._registry == other._registry)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """Hash the unit system."""
        return hash(tuple(self._core_units) + tuple(self._required_dimensions))


class DimensionlessUnitSystem(UnitSystem):
    """A unit system with only dimensionless units."""

    _required_dimensions: ClassVar[list[u.PhysicalType]] = []

    def __init__(self) -> None:
        self._core_units = [u.one]
        self._registry = {"dimensionless": u.one}

    def __getitem__(self, key: str | u.PhysicalType) -> u.UnitBase:
        return u.one

    def __str__(self) -> str:
        return "UnitSystem(dimensionless)"

    def __repr__(self) -> str:
        return "DimensionlessUnitSystem()"


# define galactic unit system
galactic = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian, u.km / u.s)

# solar system units
solarsystem = UnitSystem(u.au, u.M_sun, u.yr, u.radian)

# dimensionless
dimensionless = DimensionlessUnitSystem()
