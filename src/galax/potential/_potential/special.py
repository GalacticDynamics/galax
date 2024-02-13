"""galax: Galactic Dynamix in Jax."""

__all__ = ["BovyMWPotential2014", "LM10Potential", "MilkyWayPotential"]


from collections.abc import Mapping
from dataclasses import KW_ONLY
from math import sqrt
from types import MappingProxyType
from typing import Any, ClassVar, TypeVar, final

import astropy.units as u
import equinox as eqx

from jax_quantity import Quantity

from .base import AbstractPotentialBase
from .builtin import (
    HernquistPotential,
    LogarithmicPotential,
    MiyamotoNagaiPotential,
    NFWPotential,
)
from .composite import AbstractCompositePotential
from .utils import converter_to_usys
from galax.units import UnitSystem, dimensionless, galactic

T = TypeVar("T", bound=AbstractPotentialBase)

_sqrt2 = sqrt(2)


def _parse_input_comp(
    cls: type[T],
    instance: T | Mapping[str, Any] | None,
    default: Mapping[str, Any],
    units: UnitSystem,
) -> T:
    if isinstance(instance, cls):
        return instance

    if units == dimensionless:
        default = {k: v.value for k, v in default.items()}

    return cls(units=units, **dict(default) | (dict(instance or {})))


@final
class BovyMWPotential2014(AbstractCompositePotential):
    """``MWPotential2014`` from Bovy (2015).

    An implementation of the ``MWPotential2014``
    `from galpy <https://galpy.readthedocs.io/en/latest/potential.html>`_
    and described in `Bovy (2015)
    <https://ui.adsabs.harvard.edu/#abs/2015ApJS..216...29B/abstract>`_.

    This potential consists of a spherical bulge and dark matter halo, and a
    Miyamoto-Nagai disk component.

    .. note::

        Because it internally uses the PowerLawCutoffPotential,
        this potential requires GSL to be installed, and Gala must have been
        built and installed with GSL support enaled (the default behavior).
        See http://gala.adrian.pw/en/latest/install.html for more information.

    Parameters
    ----------
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.
    disk : dict (optional)
        Parameters to be passed to the :class:`~gala.potential.MiyamotoNagaiPotential`.
    bulge : dict (optional)
        Parameters to be passed to the :class:`~gala.potential.PowerLawCutoffPotential`.
    halo : dict (optional)
        Parameters to be passed to the :class:`~gala.potential.NFWPotential`.

    Note: in subclassing, order of arguments must match order of potential
    components added at bottom of init.
    """

    _data: dict[str, AbstractPotentialBase] = eqx.field(init=False)
    _: KW_ONLY
    units: UnitSystem = eqx.field(init=True, static=True, converter=converter_to_usys)
    _G: float = eqx.field(init=False, static=True, repr=False, converter=float)

    _default_disk: ClassVar[MappingProxyType[str, Quantity]] = MappingProxyType(
        {
            "m": Quantity(68193902782.346756, u.Msun),
            "a": Quantity(3.0, u.kpc),
            "b": Quantity(280, u.pc),
        }
    )
    _default_bulge: ClassVar[MappingProxyType[str, Any]] = MappingProxyType(
        {
            "m": Quantity(4501365375.06545, u.Msun),
            "alpha": 1.8,
            "r_c": Quantity(1.9, u.kpc),
        }
    )
    _default_halo: ClassVar[MappingProxyType[str, Quantity]] = MappingProxyType(
        {"m": Quantity(4.3683325e11, u.Msun), "r_s": Quantity(16, u.kpc)}
    )

    def __init__(
        self,
        *,
        units: Any = galactic,
        disk: MiyamotoNagaiPotential | Mapping[str, Any] | None = None,
        bulge: HernquistPotential | Mapping[str, Any] | None = None,
        halo: NFWPotential | Mapping[str, Any] | None = None,
    ) -> None:
        units_ = converter_to_usys(units) if units is not None else galactic

        super().__init__(
            disk=_parse_input_comp(
                MiyamotoNagaiPotential, disk, self._default_disk, units_
            ),
            bulge=_parse_input_comp(
                HernquistPotential, bulge, self._default_bulge, units_
            ),
            halo=_parse_input_comp(NFWPotential, halo, self._default_halo, units_),
        )

        # __post_init__ stuff:
        # Check that all potentials have the same unit system
        if not all(p.units == units_ for p in self.values()):
            msg = "all potentials must have the same unit system"
            raise ValueError(msg)
        object.__setattr__(self, "units", units_)

        # Apply the unit system to any parameters.
        self._init_units()


@final
class LM10Potential(AbstractCompositePotential):
    """Law & Majewski (2010) Milky Way mass model.

    The Galactic potential used by Law and Majewski (2010) to represent
    the Milky Way as a three-component sum of disk, bulge, and halo.

    The disk potential is an axisymmetric
    :class:`~galax.potential.MiyamotoNagaiPotential`, the bulge potential
    is a spherical :class:`~galax.potential.HernquistPotential`, and the
    halo potential is a triaxial :class:`~galax.potential.LogarithmicPotential`.

    Default parameters are fixed to those found in LM10 by fitting N-body
    simulations to the Sagittarius stream.

    Parameters
    ----------
    units : `~galax.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.
    disk : dict (optional)
        Parameters to be passed to the :class:`~galax.potential.MiyamotoNagaiPotential`.
    bulge : dict (optional)
        Parameters to be passed to the :class:`~galax.potential.HernquistPotential`.
    halo : dict (optional)
        Parameters to be passed to the :class:`~galax.potential.LogarithmicPotential`.

    Note: in subclassing, order of arguments must match order of potential
    components added at bottom of init.
    """

    _data: dict[str, AbstractPotentialBase] = eqx.field(init=False)
    _: KW_ONLY
    units: UnitSystem = eqx.field(init=True, static=True, converter=converter_to_usys)
    _G: float = eqx.field(init=False, static=True, repr=False, converter=float)

    _default_disk: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            "m": Quantity(1e11, u.Msun),
            "a": Quantity(6.5, u.kpc),
            "b": Quantity(0.26, u.kpc),
        }
    )
    _default_bulge: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {"m": Quantity(3.4e10, u.Msun), "c": Quantity(0.7, u.kpc)}
    )
    _default_halo: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            "q1": 1.38,
            "q2": 1.0,
            "q3": 1.36,
            "r_h": Quantity(12.0, u.kpc),
            "phi": Quantity(97, u.degree),
            "v_c": Quantity(_sqrt2 * 121.858, u.km / u.s),
        }
    )

    def __init__(
        self,
        *,
        units: Any = galactic,
        disk: MiyamotoNagaiPotential | Mapping[str, Any] | None = None,
        bulge: HernquistPotential | Mapping[str, Any] | None = None,
        halo: LogarithmicPotential | Mapping[str, Any] | None = None,
    ) -> None:
        units_ = converter_to_usys(units) if units is not None else galactic

        super().__init__(
            disk=_parse_input_comp(
                MiyamotoNagaiPotential, disk, self._default_disk, units_
            ),
            bulge=_parse_input_comp(
                HernquistPotential, bulge, self._default_bulge, units_
            ),
            halo=_parse_input_comp(
                LogarithmicPotential, halo, self._default_halo, units_
            ),
        )

        # __post_init__ stuff:
        # Check that all potentials have the same unit system
        if not all(p.units == units_ for p in self.values()):
            msg = "all potentials must have the same unit system"
            raise ValueError(msg)
        object.__setattr__(self, "units", units_)

        # Apply the unit system to any parameters.
        self._init_units()


@final
class MilkyWayPotential(AbstractCompositePotential):
    """Milky Way mass model.

    A simple mass-model for the Milky Way consisting of a spherical nucleus and
    bulge, a Miyamoto-Nagai disk, and a spherical NFW dark matter halo.

    The disk model is taken from `Bovy (2015)
    <https://ui.adsabs.harvard.edu/#abs/2015ApJS..216...29B/abstract>`_ - if you
    use this potential, please also cite that work.

    Default parameters are fixed by fitting to a compilation of recent mass
    measurements of the Milky Way, from 10 pc to ~150 kpc.

    Parameters
    ----------
    units : `~galax.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.
    disk : dict (optional)
        Parameters to be passed to the :class:`~galax.potential.MiyamotoNagaiPotential`.
    bulge : dict (optional)
        Parameters to be passed to the :class:`~galax.potential.HernquistPotential`.
    halo : dict (optional)
        Parameters to be passed to the :class:`~galax.potential.NFWPotential`.
    nucleus : dict (optional)
        Parameters to be passed to the :class:`~galax.potential.HernquistPotential`.

    Note: in subclassing, order of arguments must match order of potential
    components added at bottom of init.
    """

    _data: dict[str, AbstractPotentialBase] = eqx.field(init=False)
    _: KW_ONLY
    units: UnitSystem = eqx.field(init=True, static=True, converter=converter_to_usys)
    _G: float = eqx.field(init=False, static=True, repr=False, converter=float)

    _default_disk: ClassVar[MappingProxyType[str, Quantity]] = MappingProxyType(
        {
            "m": Quantity(6.8e10, u.Msun),
            "a": Quantity(3.0, u.kpc),
            "b": Quantity(0.28, u.kpc),
        }
    )
    _default_halo: ClassVar[MappingProxyType[str, Quantity]] = MappingProxyType(
        {"m": Quantity(5.4e11, u.Msun), "r_s": Quantity(15.62, u.kpc)}
    )
    _default_bulge: ClassVar[MappingProxyType[str, Quantity]] = MappingProxyType(
        {"m": Quantity(5e9, u.Msun), "c": Quantity(1.0, u.kpc)}
    )
    _default_nucleus: ClassVar[MappingProxyType[str, Quantity]] = MappingProxyType(
        {"m": Quantity(1.71e9, u.Msun), "c": Quantity(0.07, u.kpc)}
    )

    def __init__(
        self,
        *,
        units: Any = galactic,
        disk: MiyamotoNagaiPotential | Mapping[str, Any] | None = None,
        halo: NFWPotential | Mapping[str, Any] | None = None,
        bulge: HernquistPotential | Mapping[str, Any] | None = None,
        nucleus: HernquistPotential | Mapping[str, Any] | None = None,
    ) -> None:
        units_ = converter_to_usys(units) if units is not None else galactic

        super().__init__(
            disk=_parse_input_comp(
                MiyamotoNagaiPotential, disk, self._default_disk, units_
            ),
            halo=_parse_input_comp(NFWPotential, halo, self._default_halo, units_),
            bulge=_parse_input_comp(
                HernquistPotential, bulge, self._default_bulge, units_
            ),
            nucleus=_parse_input_comp(
                HernquistPotential, nucleus, self._default_nucleus, units_
            ),
        )

        # __post_init__ stuff:
        # Check that all potentials have the same unit system
        if not all(p.units == units_ for p in self.values()):
            msg = "all potentials must have the same unit system"
            raise ValueError(msg)
        object.__setattr__(self, "units", units_)

        # Apply the unit system to any parameters.
        self._init_units()
