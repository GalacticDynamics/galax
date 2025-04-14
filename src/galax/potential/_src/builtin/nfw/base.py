"""galax: Galactic Dynamix in Jax."""

__all__ = [
    # class
    "NFWPotential",
    # functions
    "rho0_of_m",
    "m_of_rho0",
    "density",
    "enclosed_mass",
    "potential",
]

from dataclasses import KW_ONLY
from functools import partial
from typing import final

import equinox as eqx
import jax
import numpy as np

import quaxed.numpy as jnp
import unxt as u
from xmmutablemap import ImmutableMap

import galax._custom_types as gt
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.utils import r_spherical


@final
class NFWPotential(AbstractSinglePotential):
    r"""NFW Potential.

    The NFW profile is one of the most commonly used model profiles for dark
    matter halos.

    The density profile is given by:

    $$ \rho(r) = \frac{ \rho_0 }{\frac{r}{r_s} (1 + \frac{r}{r_s})^2} $$

    where :math:`\rho_0` is the central density and :math:`r_s` is the scale
    radius. For similarity with the other potentials, we define the
    characteristic mass

    $$ m = 4 \pi \rho_0 r_s^3 $$

    Since the integrated mass of the NFW profile from 0 to :math:`r` is

    $$ M(<r) = \frac{m}{\ln(1 + x) - \frac{x}{1 + x}} $$ $$

    where $ x = r / r_s$ is the dimensionless radius.

    Solving Poisson's equation gives the gravitational potential

    $$ \Phi(r) = -\frac{G m}{r_s} \frac{r_s}{r} \log(1 + \frac{r}{r_s}) $$

    Which has the expected behavior of diverging at the center and going to zero
    at infinity.

    """

    m: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass", doc="Mass parameter. This is NOT the total mass."
    )

    r_s: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="Scale radius of the potential."
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit)
    def _potential(  # TODO: inputs w/ units
        self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /
    ) -> gt.BBtSz0:
        r"""Potential energy.

        .. math::

            \Phi(r) = -\frac{G M}{r_s} \frac{r_s}{r} \log(1 + \frac{r}{r_s})
        """
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])
        t = u.Quantity.from_(t, self.units["time"])

        params = {
            "G": self.constants["G"].value,
            "m": self.m(t, ustrip=self.units["mass"]),
            "r_s": self.r_s(t, ustrip=self.units["length"]),
        }
        return potential(params, r)

    @partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BtFloatSz0:
        r"""Density.

        .. math::

            v_{h2} = -\frac{G M}{r_s}
            \rho_0 = \frac{v_{h2}}{4 \pi G r_s^2}
            \rho(r) = \frac{\rho_0}{u (1 + u)^2}

        """
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])
        t = u.Quantity.from_(t, self.units["time"])

        params = {
            "m": self.m(t, ustrip=self.units["mass"]),
            "r_s": self.r_s(t, ustrip=self.units["length"]),
        }
        return density(params, r)

    # ===========================================
    # Constructors

    @staticmethod
    def _vc_rs_rref_to_m(
        v_c: u.Quantity["velocity"],
        r_s: u.Quantity["length"],
        r_ref: u.Quantity["length"],
        *,
        G: u.AbstractQuantity,
    ) -> u.Quantity["mass"]:
        uu = r_ref / r_s
        vs2 = v_c**2 / uu / (jnp.log(1 + uu) / uu**2 - 1 / (uu * (1 + uu)))
        return r_s * vs2 / G

    @classmethod
    def from_circular_velocity(
        cls,
        v_c: u.Quantity["velocity"],
        r_s: u.Quantity["length"],
        r_ref: u.Quantity["length"] | None = None,
        *,
        units: u.AbstractUnitSystem | str = "galactic",
        constants: ImmutableMap[str, u.AbstractQuantity] = default_constants,
    ) -> "NFWPotential":
        r"""Create an NFW potential from the circular velocity at a given radius.

        Parameters
        ----------
        v_c
            Circular velocity (at the specified reference radius).
        r_s
            Scale radius.
        r_ref (optional)
            The reference radius for the circular velocity. If None, the scale radius is
            used.

        Returns
        -------
        NFWPotential
            NFW potential instance with the given circular velocity and scale radius.
        """
        r_ref = r_s if r_ref is None else r_ref
        usys = u.unitsystem(units)
        m = cls._vc_rs_rref_to_m(v_c, r_s, r_ref, G=constants["G"])
        return cls(m=m, r_s=r_s, units=usys)

    @classmethod
    def from_M200_c(
        cls,
        M200: u.Quantity["mass"],
        c: u.Quantity["dimensionless"],
        rho_c: u.Quantity["mass density"] | None = None,
        *,
        units: u.AbstractUnitSystem | str,
    ) -> "NFWPotential":
        """Create an NFW potential from a virial mass and concentration.

        Parameters
        ----------
        M200
            Virial mass, or mass at 200 times the critical density, ``rho_c``.
        c
            NFW halo concentration.
        rho_c (optional)
            Critical density at z=0. If not specified, uses the default astropy
            cosmology to obtain this, `~astropy.cosmology.default_cosmology`.
        """
        usys = u.unitsystem(units)
        if rho_c is None:
            from astropy.cosmology import default_cosmology

            cosmo = default_cosmology.get()
            rho_c = (3 * cosmo.H(0.0) ** 2 / (8 * np.pi * default_constants["G"])).to(
                usys["mass density"]
            )
            rho_c = u.Quantity(rho_c.value, usys["mass density"])

        r_vir = jnp.cbrt(M200 / (200 * rho_c) / (4.0 / 3 * jnp.pi))
        r_s = r_vir / c

        A_NFW = jnp.log(1 + c) - c / (1 + c)
        m = M200 / A_NFW

        return cls(m=m, r_s=r_s, units=usys)


# =============================================================


# -----------------------------------------------


@partial(jax.jit)
def rho0_of_m(p: gt.Params, /) -> gt.Sz0:
    r"""Central density for the NFW model.

    The NFW profile is parametrized by a characteristic mass $m$ and a scale
    radius $r_s$. The central density is given by

    $$ \rho_0 = \frac{m}{4 \pi r_s^3}. $$

    Examples
    --------
    >>> import jax.numpy as jnp

    A quick sanity check:

    >>> rho0_of_m({"m": 1.0, "r_s": 1.0}) - 1 / (4 * jnp.pi)
    Array(0., dtype=float64, weak_type=True)

    """
    return p["m"] / (4 * jnp.pi * p["r_s"] ** 3)


@partial(jax.jit)
def m_of_rho0(p: gt.Params, /) -> gt.Sz0:
    r"""Characteristic mass for the NFW model.

    The NFW profile is parametrized by a characteristic mass $m$ and a scale
    radius $r_s$. The characteristic mass is given by

    $$ m = 4 \pi \rho_0 r_s^3. $$

    Examples
    --------
    >>> import jax.numpy as jnp

    A quick sanity check:
    >>> m_of_rho0({"rho0": 1.0, "r_s": 1.0}) - 4 * jnp.pi
    Array(0., dtype=float64, weak_type=True)

    """
    return 4 * jnp.pi * p["rho0"] * p["r_s"] ** 3


# -----------------------------------------------


@partial(jax.jit)
def density(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""Density profile for the NFW model.

    $$ \rho(r) = \frac{ \rho_0 }{\frac{r}{r_s} (1 + \frac{r}{r_s})^2} $$

    where $\rho_0$ is the central density and $r_s$ is the scale radius. We
    actually parametrize not with $\rho_0$ but by a characteristic mass $m$,
    defined as

    $$ m = 4 \pi \rho_0 r_s^3. $$

    """
    x = r / p["r_s"]
    rho0: gt.Sz0 = rho0_of_m(p)
    return rho0 / (x * (1 + x) ** 2)


@partial(jax.jit)
def enclosed_mass(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""Enclosed mass for the NFW model.

    $$ M(<r) = \frac{m}{\ln(1 + x) - \frac{x}{1 + x}} $$

    where $x = r / r_s$ is the dimensionless radius and $m$ is the
    characteristic mass.

    """
    x = r / p["r_s"]
    m = p["m"]
    return m * (jnp.log(1 + x) - x / (1 + x))


@partial(jax.jit)
def potential(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""Potential for the NFW model.

    $$ \Phi(r) = -\frac{G m}{r_s} \frac{r_s}{r} \log(1 + \frac{r}{r_s}) $$

    where $m$ is the characteristic mass and $r_s$ is the scale radius.

    """
    r_s = p["r_s"]
    x = r / r_s
    phi0 = -p["G"] * p["m"] / r_s
    return phi0 * jnp.log(1 + x) / x
