"""Paired down UnitSystem class from gala.

See gala's license below.

```
The MIT License (MIT)

Copyright (c) 2012-2023 Adrian M. Price-Whelan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
"""

from __future__ import annotations

from typing import Any, ClassVar

__all__ = [
    "UnitSystem",
    "DimensionlessUnitSystem",
    "galactic",
    "dimensionless",
    "solarsystem",
]


import astropy.units as u


class UnitSystem:
    """Represents a system of units."""

    _core_units: list[u.UnitBase]
    _registry: dict[u.PhysicalType, u.UnitBase]

    _required_physical_types: ClassVar[list[u.PhysicalType]] = [
        u.get_physical_type("length"),
        u.get_physical_type("time"),
        u.get_physical_type("mass"),
        u.get_physical_type("angle"),
    ]

    def __init__(self, units: UnitSystem | u.UnitBase, *args: u.UnitBase):
        if isinstance(units, UnitSystem):
            if len(args) > 0:
                msg = "If passing in a UnitSystem instance, you cannot pass in additional units."
                raise ValueError(msg)

            self._registry = units._registry.copy()
            self._core_units = units._core_units
            return

        units = (units, *args)

        self._registry = {}
        for unit in units:
            unit_ = (
                unit if isinstance(unit, u.UnitBase) else u.def_unit(f"{unit!s}", unit)
            )
            if unit_.physical_type in self._registry:
                msg = f"Multiple units passed in with type {unit_.physical_type!r}"
                raise ValueError(msg)
            self._registry[unit_.physical_type] = unit_

        self._core_units = []
        for phys_type in self._required_physical_types:
            if phys_type not in self._registry:
                msg = f"You must specify a unit for the physical type {phys_type!r}"
                raise ValueError(msg)
            self._core_units.append(self._registry[phys_type])

    def __getitem__(self, key: str | u.PhysicalType) -> u.UnitBase:
        key = u.get_physical_type(key)
        return self._registry[key]

    def __len__(self) -> int:
        return len(self._core_units)

    def __iter__(self) -> u.UnitBase:
        yield from self._core_units

    def __repr__(self) -> str:
        return f"UnitSystem({', '.join(str(uu) for uu in self._core_units)})"

    def __eq__(self, other: Any) -> bool:
        return bool(self._registry == other._registry)

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)


class DimensionlessUnitSystem(UnitSystem):
    _required_physical_types: ClassVar[list[u.PhysicalType]] = []

    def __init__(self) -> None:
        self._core_units = [u.one]
        self._registry = {"dimensionless": u.one}

    def __getitem__(self, key: str) -> u.UnitBase:
        return u.one

    def __str__(self) -> str:
        return "UnitSystem(dimensionless)"


# define galactic unit system
galactic = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian, u.km / u.s)

# solar system units
solarsystem = UnitSystem(u.au, u.M_sun, u.yr, u.radian)

# dimensionless
dimensionless = DimensionlessUnitSystem()
