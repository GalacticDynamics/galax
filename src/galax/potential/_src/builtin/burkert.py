"""galax: Galactic Dynamix in Jax."""

__all__ = [
    # class
    "BurkertPotential",
    # functions
    "rho0",
    "density",
    "potential",
]

import functools as ft
from dataclasses import KW_ONLY
from typing import Any, Final, final

import equinox as eqx
import jax

import quaxed.numpy as jnp
import unxt as u
from xmmutablemap import ImmutableMap

import galax._custom_types as gt
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.utils import r_spherical

# -------------------------------------------------------------------


BURKERT_CONST: Final = 3 * jnp.log(jnp.asarray(2.0)) - 0.5 * jnp.pi


@final
class BurkertPotential(AbstractSinglePotential):
    r"""Burkert Potential.

    https://ui.adsabs.harvard.edu/abs/1995ApJ...447L..25B/abstract,
    https://iopscience.iop.org/article/10.1086/309140/fulltext/50172.text.html.

    The mass parameter sets the core mass:

    .. math::

        M_0 = \\pi \rho_0 r_s^3 (3 \\log(2) - \\pi / 2)

    """

    m: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass",
        doc="Core mass of the potential (i.e. the mass within r_s).",
    )

    r_s: AbstractParameter = ParameterField(dimensions="length", doc="Scale radius")  # type: ignore[assignment]

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])
        t = u.Quantity.from_(t, self.units["time"])

        # Compute potential
        params = {
            "G": self.constants["G"].value,
            "m": self.m(t, ustrip=self.units["mass"]),
            "r_s": self.r_s(t, ustrip=self.units["length"]),
        }
        return potential(params, r)

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BtFloatSz0:
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])
        t = u.Quantity.from_(t, self.units["time"])

        params = {
            "m": self.m(t, ustrip=self.units["mass"]),
            "r_s": self.r_s(t, ustrip=self.units["length"]),
        }
        return density(params, r)

    @ft.partial(jax.jit)
    def _mass(self, xyz: gt.BBtQuSz3, /, t: gt.BtQuSz0 | gt.QuSz0) -> gt.BtFloatQuSz0:
        t = u.Quantity.from_(t, self.units["time"])
        x = jnp.linalg.vector_norm(xyz, axis=-1) / self.r_s(t)
        return (
            self.m(t)
            / BURKERT_CONST
            * (-2 * jnp.atan(x) + 2 * jnp.log(1 + x) + jnp.log(1 + x**2))
        )

    # -------------------------------------------------------------------

    def rho0(self, t: gt.BtQuSz0 | gt.QuSz0) -> gt.BtFloatQuSz0:
        r"""Central density of the potential.

        $$ m0 = \pi \rho_0 r_s^3 (3 \log(2) - \pi / 2) $$
        """
        return rho0(self.m(t), self.r_s(t))

    # -------------------------------------------------------------------
    # Constructors

    @classmethod
    def from_central_density(
        cls, rho_0: u.Quantity, r_s: u.Quantity, **kwargs: Any
    ) -> "BurkertPotential":
        r"""Create a Burkert potential from the central density.

        Parameters
        ----------
        rho_0 : :class:`~unxt.Quantity`[mass density]
            Central density.
        r_s : :class:`~unxt.Quantity`[length]
            Scale radius.

        Returns
        -------
        :class:`~galax.potential.BurkertPotential`
            Burkert potential.

        Examples
        --------
        >>> import unxt as u
        >>> import galax.potential as gp

        >>> rho_0 = u.Quantity(1e6, "Msun / kpc3")
        >>> r_s = u.Quantity(1, "kpc")
        >>> pot = gp.BurkertPotential.from_central_density(rho_0, r_s, units="galactic")
        >>> pot
        BurkertPotential(
            units=LTMAUnitSystem( length=Unit("kpc"), ...),
            constants=ImmutableMap({'G': ...}),
            m=ConstantParameter(...),
            r_s=ConstantParameter(...)
        )

        """
        m = jnp.pi * rho_0 * r_s**3 * BURKERT_CONST
        return cls(m=m, r_s=r_s, **kwargs)


# ===================================================================


@ft.partial(jax.jit)
def rho0(m: gt.QuSz0, r_s: gt.QuSz0, /) -> gt.Sz0:
    r"""Central density of the potential."""
    return m / (BURKERT_CONST * jnp.pi * r_s**3)


@ft.partial(jax.jit)
def density(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Density profile for the Burkert potential.

    $$ \rho(r) = \frac{m}{\pi (3 \log(2) - \pi / 2)}
    \frac{1}{(r + r_s)(r^2 + r_s^2)} $$

    where $m$ is the characteristic mass and $r_s$ is the scale radius.
    """
    factor = p["m"] / (jnp.pi * BURKERT_CONST)
    return factor / ((r + p["r_s"]) * (r**2 + p["r_s"] ** 2))


@ft.partial(jax.jit)
def potential(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Potential for the Burkert potential.

    $$
    \Phi(r) = -\frac{G m}{r_s (3 \log(2) - \pi / 2)}
        \left[ \pi - 2 (1 + r/r_s) \tan^{-1}(r/r_s)
        + 2 (1 + r/r_s) \log(1 + r/r_s)
        - (1 - r/r_s) \log(1 + (r/r_s)^2)
        \right]
    $$

    where $m$ is the characteristic mass and $r_s$ is the scale radius.
    """
    # Compute potential
    r_s = p["r_s"]
    x = r / r_s
    xinv = 1 / x
    prefactor = p["G"] * p["m"] / (r_s * BURKERT_CONST)
    return -prefactor * (
        jnp.pi
        - 2 * (1 + xinv) * jnp.atan(x)
        + 2 * (1 + xinv) * jnp.log(1 + x)
        - (1 - xinv) * jnp.log(1 + x**2)
    )
