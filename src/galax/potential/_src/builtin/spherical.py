"""galax: Galactic Dynamix in Jax."""

__all__ = [
    "BurkertPotential",
    "PowerLawCutoffPotential",
]

import functools as ft
from dataclasses import KW_ONLY
from typing import Any, Final, final

import equinox as eqx
import jax

import quaxed.numpy as jnp
import quaxed.scipy.special as qsp
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
    """Burkert Potential.

    https://ui.adsabs.harvard.edu/abs/1995ApJ...447L..25B/abstract,
    https://iopscience.iop.org/article/10.1086/309140/fulltext/50172.text.html.

    """

    m: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass",
        doc=r"""Characteristic mass of the potential.

    $$ m0 = \pi \rho_0 r_s^3 (3 \log(2) - \pi / 2) $$

    """,
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
        # Compute parameters
        m = self.m(t, ustrip=self.units["mass"])
        r_s = self.r_s(t, ustrip=self.units["length"])

        # Compute potential
        x = r / r_s
        xinv = 1 / x
        prefactor = self.constants["G"].value * m / (r_s * BURKERT_CONST)
        return -prefactor * (
            jnp.pi
            - 2 * (1 + xinv) * jnp.atan(x)
            + 2 * (1 + xinv) * jnp.log(1 + x)
            - (1 - xinv) * jnp.log(1 + x**2)
        )

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BtFloatSz0:
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])
        t = u.Quantity.from_(t, self.units["time"])
        # Compute parameters
        m = self.m(t, ustrip=self.units["mass"])
        r_s = self.r_s(t, ustrip=self.units["length"])

        return m / (jnp.pi * BURKERT_CONST) / ((r + r_s) * (r**2 + r_s**2))

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

        .. math::

            m0 = \pi \rho_0 r_s^3 (3 \log(2) - \pi / 2)
        """
        return self.m(t) / (jnp.pi * self.r_s(t) ** 3 * BURKERT_CONST)

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
            m=ConstantParameter( ... ),
            r_s=ConstantParameter( ... )
        )

        """
        m = jnp.pi * rho_0 * r_s**3 * BURKERT_CONST
        return cls(m=m, r_s=r_s, **kwargs)


# -------------------------------------------------------------------


@ft.partial(jax.jit)
def _safe_gamma_inc(a: u.Quantity, x: u.Quantity) -> u.Quantity:  # TODO: types
    return qsp.gammainc(a, x) * qsp.gamma(a)


@final
class PowerLawCutoffPotential(AbstractSinglePotential):
    r"""A spherical power-law density profile with an exponential cutoff.

    .. math::

        \rho(r) = \frac{G M}{2\pi \Gamma((3-\alpha)/2) r_c^3} \left(\frac{r_c}{r}\right)^\alpha \exp{-(r / r_c)^2}

    Parameters
    ----------
    m_tot : :class:`~unxt.Quantity`[mass]
        Total mass.
    alpha : :class:`~unxt.Quantity`[dimensionless]
        Power law index. Must satisfy: ``0 <= alpha < 3``.
    r_c : :class:`~unxt.Quantity`[length]
        Cutoff radius.
    """  # noqa: E501

    m_tot: AbstractParameter = ParameterField(dimensions="mass", doc="Total mass.")  # type: ignore[assignment]
    """Total mass of the potential."""

    alpha: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        doc="Power law index. Must satisfy: ``0 <= alpha < 3``",
    )

    r_c: AbstractParameter = ParameterField(dimensions="length", doc="Cutoff radius.")  # type: ignore[assignment]

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQuSz3, t: gt.BBtQuSz0, /) -> gt.BtSz0:
        # Parse inputs
        ul = self.units["length"]
        r = r_spherical(xyz, ul)
        t = u.Quantity.from_(t, self.units["time"])
        # Compute parameters
        m_tot = self.m_tot(t, ustrip=self.units["mass"])
        alpha = self.alpha(t, ustrip=self.units["dimensionless"])
        r_c = self.r_c(t, ustrip=ul)

        a = alpha / 2
        s2 = (r / r_c) ** 2
        GM = self.constants["G"].value * m_tot

        return GM * (
            (a - 1.5) * _safe_gamma_inc(1.5 - a, s2) / (r * qsp.gamma(2.5 - a))
            + _safe_gamma_inc(1 - a, s2) / (r_c * qsp.gamma(1.5 - a))
        )
