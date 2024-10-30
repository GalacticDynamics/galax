"""galax: Galactic Dynamix in Jax."""

__all__ = [
    "BovyMWPotential2014",
    "LM10Potential",
    "MilkyWayPotential",
    "MilkyWayPotential2022",
]


from collections.abc import Mapping
from dataclasses import KW_ONLY
from typing import Any, ClassVar, TypeVar, cast, final

import equinox as eqx

from dataclassish import replace
from unxt import Quantity
from unxt.unitsystems import AbstractUnitSystem, galactic, unitsystem
from xmmutablemap import ImmutableMap

from .builtin import (
    HernquistPotential,
    MiyamotoNagaiPotential,
    MN3Sech2Potential,
    PowerLawCutoffPotential,
)
from .const import _sqrt2
from .logarithmic import LMJ09LogarithmicPotential
from .nfw import NFWPotential
from galax.potential._src.base import AbstractBasePotential, default_constants
from galax.potential._src.base_multi import AbstractCompositePotential

T = TypeVar("T", bound=AbstractBasePotential)


def _parse_potential(
    base: AbstractBasePotential,
    instance: AbstractBasePotential | Mapping[str, Any] | None,
    units: Any,
) -> AbstractBasePotential:
    if isinstance(instance, type(base)):
        return instance
    if isinstance(instance, AbstractBasePotential):
        msg = f"instance must be of type {type(base)}, not {type(instance)}"
        raise TypeError(msg)

    return cast(AbstractBasePotential, replace(base, **(instance or {}), units=units))


# =============================================================================


@final
class BovyMWPotential2014(AbstractCompositePotential):
    """``MWPotential2014`` from Bovy (2015).

    An implementation of the ``MWPotential2014`` `from galpy
    <https://galpy.readthedocs.io/en/latest/potential.html>`_ and described in
    `Bovy (2015)
    <https://ui.adsabs.harvard.edu/#abs/2015ApJS..216...29B/abstract>`_.

    This potential consists of a spherical bulge and dark matter halo, and a
    Miyamoto-Nagai disk component.

    Parameters
    ----------
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducible units that specify (at minimum) the length, mass,
        time, and angle units.
    disk : dict (optional)
        Parameters to be passed to the
        :class:`~gala.potential.MiyamotoNagaiPotential`.
    bulge : dict (optional)
        Parameters to be passed to the
        :class:`~gala.potential.PowerLawCutoffPotential`.
    halo : dict (optional)
        Parameters to be passed to the :class:`~gala.potential.NFWPotential`.

    Note: in subclassing, order of arguments must match order of potential
    components added at bottom of init.
    """

    _data: dict[str, AbstractBasePotential] = eqx.field(init=False)
    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(
        default=galactic, static=True, converter=unitsystem
    )
    constants: ImmutableMap[str, Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    _default_disk: ClassVar[MiyamotoNagaiPotential] = MiyamotoNagaiPotential(
        m_tot=Quantity(68_193_902_782.346756, "Msun"),
        a=Quantity(3.0, "kpc"),
        b=Quantity(280, "pc"),
        units=galactic,
    )
    _default_bulge: ClassVar[PowerLawCutoffPotential] = PowerLawCutoffPotential(
        m_tot=Quantity(4501365375.06545, "Msun"),
        alpha=1.8,
        r_c=Quantity(1.9, "kpc"),
        units=galactic,
    )
    _default_halo: ClassVar[NFWPotential] = NFWPotential(
        m=Quantity(4.3683325e11, "Msun"), r_s=Quantity(16, "kpc"), units=galactic
    )

    def __init__(
        self,
        *,
        disk: MiyamotoNagaiPotential | Mapping[str, Any] | None = None,
        bulge: PowerLawCutoffPotential | Mapping[str, Any] | None = None,
        halo: NFWPotential | Mapping[str, Any] | None = None,
        units: Any = galactic,
        constants: Any = default_constants,
    ) -> None:
        units_ = unitsystem(units)

        super().__init__(
            disk=_parse_potential(self._default_disk, disk, units_),
            bulge=_parse_potential(self._default_bulge, bulge, units_),
            halo=_parse_potential(self._default_halo, halo, units_),
            units=units_,
            constants=constants,
        )


@final
class LM10Potential(AbstractCompositePotential):
    """Law & Majewski (2010) Milky Way mass model.

    The Galactic potential used by Law and Majewski (2010) to represent the
    Milky Way as a three-component sum of disk, bulge, and halo.

    The disk potential is an axisymmetric
    :class:`~galax.potential.MiyamotoNagaiPotential`, the bulge potential is a
    spherical :class:`~galax.potential.HernquistPotential`, and the halo
    potential is a triaxial :class:`~galax.potential.LMJ09LogarithmicPotential`.

    Default parameters are fixed to those found in LM10 by fitting N-body
    simulations to the Sagittarius stream.

    Parameters
    ----------
    units : `~galax.units.UnitSystem` (optional)
        Set of non-reducible units that specify (at minimum) the length, mass,
        time, and angle units.
    disk : dict (optional)
        Parameters to be passed to the
        :class:`~galax.potential.MiyamotoNagaiPotential`.
    bulge : dict (optional)
        Parameters to be passed to the
        :class:`~galax.potential.HernquistPotential`.
    halo : dict (optional)
        Parameters to be passed to the
        :class:`~galax.potential.LMJ09LogarithmicPotential`.

    Note: in subclassing, order of arguments must match order of potential
    components added at bottom of init.
    """

    _data: dict[str, AbstractBasePotential] = eqx.field(init=False)
    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(
        default=galactic, static=True, converter=unitsystem
    )
    constants: ImmutableMap[str, Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    _default_disk: ClassVar[MiyamotoNagaiPotential] = MiyamotoNagaiPotential(
        m_tot=Quantity(1e11, "Msun"),
        a=Quantity(6.5, "kpc"),
        b=Quantity(0.26, "kpc"),
        units=galactic,
    )
    _default_bulge: ClassVar[HernquistPotential] = HernquistPotential(
        m_tot=Quantity(3.4e10, "Msun"), r_s=Quantity(0.7, "kpc"), units=galactic
    )
    _default_halo: ClassVar[LMJ09LogarithmicPotential] = LMJ09LogarithmicPotential(
        v_c=Quantity(_sqrt2 * 121.858, "km / s"),
        r_s=Quantity(12.0, "kpc"),
        q1=1.38,
        q2=1.0,
        q3=1.36,
        phi=Quantity(97, "degree"),
        units=galactic,
    )

    def __init__(
        self,
        *,
        disk: MiyamotoNagaiPotential | Mapping[str, Any] | None = None,
        bulge: HernquistPotential | Mapping[str, Any] | None = None,
        halo: LMJ09LogarithmicPotential | Mapping[str, Any] | None = None,
        units: Any = galactic,
        constants: Any = default_constants,
    ) -> None:
        units_ = unitsystem(units)

        super().__init__(
            disk=_parse_potential(self._default_disk, disk, units_),
            bulge=_parse_potential(self._default_bulge, bulge, units_),
            halo=_parse_potential(self._default_halo, halo, units_),
            units=units_,
            constants=constants,
        )


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
    units : `~unxt.AbstractUnitSystem` (optional)
        Set of non-reducible units.
    disk : dict (optional)
        Parameters to be passed to the
        :class:`~galax.potential.MiyamotoNagaiPotential`.
    bulge : dict (optional)
        Parameters to be passed to the
        :class:`~galax.potential.HernquistPotential`.
    halo : dict (optional)
        Parameters to be passed to the :class:`~galax.potential.NFWPotential`.
    nucleus : dict (optional)
        Parameters to be passed to the
        :class:`~galax.potential.HernquistPotential`.

    Note: in subclassing, order of arguments must match order of potential
    components added at bottom of init.
    """

    _data: dict[str, AbstractBasePotential] = eqx.field(init=False)
    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(init=True, static=True, converter=unitsystem)
    constants: ImmutableMap[str, Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    _default_disk: ClassVar[MiyamotoNagaiPotential] = MiyamotoNagaiPotential(
        m_tot=Quantity(6.8e10, "Msun"),
        a=Quantity(3.0, "kpc"),
        b=Quantity(0.28, "kpc"),
        units=galactic,
    )
    _default_halo: ClassVar[NFWPotential] = NFWPotential(
        m=Quantity(5.4e11, "Msun"), r_s=Quantity(15.62, "kpc"), units=galactic
    )
    _default_bulge: ClassVar[HernquistPotential] = HernquistPotential(
        m_tot=Quantity(5e9, "Msun"), r_s=Quantity(1.0, "kpc"), units=galactic
    )
    _default_nucleus: ClassVar[HernquistPotential] = HernquistPotential(
        m_tot=Quantity(1.71e9, "Msun"), r_s=Quantity(70, "pc"), units=galactic
    )

    def __init__(
        self,
        *,
        disk: MiyamotoNagaiPotential | Mapping[str, Any] | None = None,
        halo: NFWPotential | Mapping[str, Any] | None = None,
        bulge: HernquistPotential | Mapping[str, Any] | None = None,
        nucleus: HernquistPotential | Mapping[str, Any] | None = None,
        units: Any = galactic,
        constants: Any = default_constants,
    ) -> None:
        units_ = unitsystem(units) if units is not None else galactic

        super().__init__(
            disk=_parse_potential(self._default_disk, disk, units_),
            halo=_parse_potential(self._default_halo, halo, units_),
            bulge=_parse_potential(self._default_bulge, bulge, units_),
            nucleus=_parse_potential(self._default_nucleus, nucleus, units_),
            units=units_,
            constants=constants,
        )


@final
class MilkyWayPotential2022(AbstractCompositePotential):
    """Milky Way mass model.

    A mass-model for the Milky Way consisting of a spherical nucleus and bulge,
    a 3-component sum of Miyamoto-Nagai disks to represent an exponential disk,
    and a spherical NFW dark matter halo.

    The disk model is fit to the Eilers et al. 2019 rotation curve for the
    radial dependence, and the shape of the phase-space spiral in the solar
    neighborhood is used to set the vertical structure in Darragh-Ford et al.
    2023.

    Other parameters are fixed by fitting to a compilation of recent mass
    measurements of the Milky Way, from 10 pc to ~150 kpc.

    Parameters
    ----------
    units : `~unxt.AbstractUnitSystem` (optional)
        Set of non-reducible units.
    disk : dict (optional)
        Parameters to be passed to the
        :class:`~galax.potential.MiyamotoNagaiPotential`.
    bulge : dict (optional)
        Parameters to be passed to the
        :class:`~galax.potential.HernquistPotential`.
    halo : dict (optional)
        Parameters to be passed to the :class:`~galax.potential.NFWPotential`.
    nucleus : dict (optional)
        Parameters to be passed to the
        :class:`~galax.potential.HernquistPotential`.

    Note: in subclassing, order of arguments must match order of potential
    components added at bottom of init.
    """

    _data: dict[str, AbstractBasePotential] = eqx.field(init=False)
    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(init=True, static=True, converter=unitsystem)
    constants: ImmutableMap[str, Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    _default_disk: ClassVar[MN3Sech2Potential] = MN3Sech2Potential(
        m_tot=Quantity(4.7717e10, "Msun"),
        h_R=Quantity(2.6, "kpc"),
        h_z=Quantity(0.3, "kpc"),
        units=galactic,
        positive_density=True,
    )
    _default_halo: ClassVar[NFWPotential] = NFWPotential(
        m=Quantity(5.5427e11, "Msun"), r_s=Quantity(15.626, "kpc"), units=galactic
    )
    _default_bulge: ClassVar[HernquistPotential] = HernquistPotential(
        m_tot=Quantity(5e9, "Msun"), r_s=Quantity(1.0, "kpc"), units=galactic
    )
    _default_nucleus: ClassVar[HernquistPotential] = HernquistPotential(
        m_tot=Quantity(1.8142e9, "Msun"), r_s=Quantity(68.8867, "pc"), units=galactic
    )

    def __init__(
        self,
        *,
        disk: MN3Sech2Potential | Mapping[str, Any] | None = None,
        halo: NFWPotential | Mapping[str, Any] | None = None,
        bulge: HernquistPotential | Mapping[str, Any] | None = None,
        nucleus: HernquistPotential | Mapping[str, Any] | None = None,
        units: Any = galactic,
        constants: Any = default_constants,
    ) -> None:
        units_ = unitsystem(units) if units is not None else galactic

        super().__init__(
            disk=_parse_potential(self._default_disk, disk, units_),
            halo=_parse_potential(self._default_halo, halo, units_),
            bulge=_parse_potential(self._default_bulge, bulge, units_),
            nucleus=_parse_potential(self._default_nucleus, nucleus, units_),
            units=units_,
            constants=constants,
        )
