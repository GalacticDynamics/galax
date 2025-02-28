"""Disk-like potentials."""

__all__ = [
    "KuzminPotential",
    "MiyamotoNagaiPotential",
    "MN3ExponentialPotential",
    "MN3Sech2Potential",
]

from dataclasses import KW_ONLY
from functools import partial
from typing import Final, final

import equinox as eqx
import jax
from jaxtyping import Array

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue
from unxt.unitsystems import dimensionless
from xmmutablemap import ImmutableMap

import galax._custom_types as gt
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.core import AbstractParameter
from galax.potential._src.params.field import ParameterField


@final
class KuzminPotential(AbstractSinglePotential):
    r"""Kuzmin Potential.

    .. math::

        \Phi(x, t) = -\frac{G M(t)}{\sqrt{R^2 + (a(t) + |z|)^2}}

    See https://galaxiesbook.org/chapters/II-01.-Flattened-Mass-Distributions.html#Razor-thin-disk:-The-Kuzmin-model

    """

    m_tot: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    """Total mass of the potential."""

    r_s: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    """Scale length of the 'disk'."""

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit, inline=True)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Quantity.from_(t, self.units["time"])

        # Compute parameters
        m_tot = self.m_tot(t, ustrip=self.units["mass"])
        r_s = self.r_s(t, ustrip=self.units["length"])

        R2 = xyz[..., 0] ** 2 + xyz[..., 1] ** 2
        z = xyz[..., 2]
        return (
            -self.constants["G"].value * m_tot / jnp.sqrt(R2 + (jnp.abs(z) + r_s) ** 2)
        )


# -------------------------------------------------------------------


@final
class MiyamotoNagaiPotential(AbstractSinglePotential):
    """Miyamoto-Nagai Potential."""

    m_tot: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    """Total mass of the potential."""

    # TODO: rename
    a: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    """Scale length in the major-axis (x-y) plane."""

    b: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    """Scale length in the minor-axis (x-y) plane."""

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit, inline=True)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Quantity.from_(t, self.units["time"])

        # Compute parameters
        ul = self.units["length"]
        m_tot = self.m_tot(t, ustrip=self.units["mass"])
        a, b = self.a(t, ustrip=ul), self.b(t, ustrip=ul)

        R2 = xyz[..., 0] ** 2 + xyz[..., 1] ** 2
        zp2 = (jnp.sqrt(xyz[..., 2] ** 2 + b**2) + a) ** 2
        return -self.constants["G"].value * m_tot / jnp.sqrt(R2 + zp2)


# -------------------------------------------------------------------


_mn3_K_pos_dens: Final = jnp.array(  # noqa: N816
    [
        [0.0036, -0.0330, 0.1117, -0.1335, 0.1749],
        [-0.0131, 0.1090, -0.3035, 0.2921, -5.7976],
        [-0.0048, 0.0454, -0.1425, 0.1012, 6.7120],
        [-0.0158, 0.0993, -0.2070, -0.7089, 0.6445],
        [-0.0319, 0.1514, -0.1279, -0.9325, 2.6836],
        [-0.0326, 0.1816, -0.2943, -0.6329, 2.3193],
    ]
)
_mn3_K_neg_dens: Final = jnp.array(  # noqa: N816
    [
        [-0.0090, 0.0640, -0.1653, 0.1164, 1.9487],
        [0.0173, -0.0903, 0.0877, 0.2029, -1.3077],
        [-0.0051, 0.0287, -0.0361, -0.0544, 0.2242],
        [-0.0358, 0.2610, -0.6987, -0.1193, 2.0074],
        [-0.0830, 0.4992, -0.7967, -1.2966, 4.4441],
        [-0.0247, 0.1718, -0.4124, -0.5944, 0.7333],
    ]
)
_mn3_b_coeffs_exp: Final = jnp.array([-0.269, 1.08, 1.092])
_mn3_b_coeffs_sech2: Final = jnp.array([-0.033, 0.262, 0.659])


class AbstractMN3Potential(AbstractSinglePotential):
    """A base class for sums of three Miyamoto-Nagai disk potentials."""

    m_tot: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    """Total mass of the potential."""

    h_R: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment] # noqa: N815
    """Radial (exponential) scale length."""

    h_z: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    """
    If ``sech2_z=True``, this is the scale height in a sech^2 vertical profile. If
    ``sech2_z=False``, this is an exponential scale height.
    """

    positive_density: bool = eqx.field(default=False, static=True)
    """
    If ``True``, the density will be positive everywhere, but is only a good
    approximation of the exponential density within about 5 disk scale lengths. If
    ``False``, the density will be negative in some regions, but is a better
    approximation out to about 7 or 8 disk scale lengths.
    """

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    def _get_mn_components(
        self, t: gt.BBtQorVSz0, /
    ) -> tuple[MiyamotoNagaiPotential, MiyamotoNagaiPotential, MiyamotoNagaiPotential]:
        hR = self.h_R(t)
        hzR = (self.h_z(t) / hR).ustrip(dimensionless)
        K = _mn3_K_pos_dens if self.positive_density else _mn3_K_neg_dens

        # get b / h_R with fitting functions:
        b_hR = self._b_coeffs @ jnp.array([hzR**3, hzR**2, hzR])

        x = jnp.vander(b_hR[None], N=5)[0]
        param_vec = K @ x

        # use fitting function to get the Miyamoto-Nagai component parameters
        mn_ms = param_vec[:3] * self.m_tot(t)
        mn_as = param_vec[3:] * hR
        mn_b = b_hR * hR
        return (
            MiyamotoNagaiPotential(
                m_tot=mn_ms[0], a=mn_as[0], b=mn_b, units=self.units
            ),
            MiyamotoNagaiPotential(
                m_tot=mn_ms[1], a=mn_as[1], b=mn_b, units=self.units
            ),
            MiyamotoNagaiPotential(
                m_tot=mn_ms[2], a=mn_as[2], b=mn_b, units=self.units
            ),
        )

    @partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.ustrip(AllowValue, self.units["time"], t)
        return jnp.sum(
            jnp.asarray([mn._potential(xyz, t) for mn in self._get_mn_components(t)]),  # noqa: SLF001
            axis=0,
        )

    @partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtFloatSz0:
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        densities = jnp.asarray(
            [mn._density(xyz, t) for mn in self._get_mn_components(t)]  # noqa: SLF001
        )
        return jnp.sum(densities, axis=0)


@final
class MN3ExponentialPotential(AbstractMN3Potential):
    """A sum of three Miyamoto-Nagai disk potentials.

    A sum of three Miyamoto-Nagai disk potentials that approximate the potential
    generated by an exponential (radial) disk with an exponential vertical profile (i.e.
    a double exponential disk).

    This model is taken from `Smith et al. (2015)
    <https://ui.adsabs.harvard.edu/abs/2015MNRAS.448.2934S/abstract>`_ : if you use this
    potential class, please also cite that work.

    As described in the above reference, this approximation has two options: (1)
    with the ``positive_density=True`` argument set, this density will be
    positive everywhere, but is only a good approximation of the exponential
    density within about 5 disk scale lengths, and (2) with
    ``positive_density=False``, this density will be negative in some regions,
    but is a better approximation out to about 7 or 8 disk scale lengths.
    """

    @property
    def _b_coeffs(self) -> Array:
        return _mn3_b_coeffs_exp


@final
class MN3Sech2Potential(AbstractMN3Potential):
    """A sum of three Miyamoto-Nagai disk potentials.

    A sum of three Miyamoto-Nagai disk potentials that approximate the potential
    generated by an exponential (radial) disk with a sech^2 vertical profile.

    This model is taken from `Smith et al. (2015)
    <https://ui.adsabs.harvard.edu/abs/2015MNRAS.448.2934S/abstract>`_ : if you use this
    potential class, please also cite that work.

    As described in the above reference, this approximation has two options: (1)
    with the ``positive_density=True`` argument set, this density will be
    positive everywhere, but is only a good approximation of the exponential
    density within about 5 disk scale lengths, and (2) with
    ``positive_density=False``, this density will be negative in some regions,
    but is a better approximation out to about 7 or 8 disk scale lengths.
    """

    @property
    def _b_coeffs(self) -> Array:
        return _mn3_b_coeffs_sech2
