"""galax: Galactic Dynamix in Jax."""

__all__ = ["MilkyWayPotential"]


from dataclasses import KW_ONLY
from typing import Any, final

import astropy.units as u
import equinox as eqx
from astropy.units import Quantity

from galax.units import UnitSystem, dimensionless, galactic

from .base import AbstractPotentialBase
from .builtin import HernquistPotential, MiyamotoNagaiPotential, NFWPotential
from .composite import AbstractCompositePotential
from .utils import converter_to_usys

_default_disk = {"m": 6.8e10 * u.Msun, "a": 3.0 * u.kpc, "b": 0.28 * u.kpc}
_default_halo = {"m": 5.4e11 * u.Msun, "r_s": 15.62 * u.kpc}
_default_bulge = {"m": 5e9 * u.Msun, "c": 1.0 * u.kpc}
_default_nucleus = {"m": 1.71e9 * u.Msun, "c": 0.07 * u.kpc}


def _munge(value: dict[str, Quantity], units: UnitSystem) -> Any:
    if units == dimensionless:
        return {k: v.value for k, v in value.items()}
    return value


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

    def __init__(
        self,
        *,
        units: Any = galactic,
        disk: dict[str, Any] | None = None,
        halo: dict[str, Any] | None = None,
        bulge: dict[str, Any] | None = None,
        nucleus: dict[str, Any] | None = None,
    ) -> None:
        units_ = converter_to_usys(units) if units is not None else galactic
        super().__init__(
            disk=MiyamotoNagaiPotential(
                units=units_, **_munge(_default_disk, units_) | (disk or {})
            ),
            halo=NFWPotential(
                units=units_, **_munge(_default_halo, units_) | (halo or {})
            ),
            bulge=HernquistPotential(
                units=units_, **_munge(_default_bulge, units_) | (bulge or {})
            ),
            nucleus=HernquistPotential(
                units=units_, **_munge(_default_nucleus, units_) | (nucleus or {})
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
