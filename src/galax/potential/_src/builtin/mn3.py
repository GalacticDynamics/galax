"""Disk-like potentials."""

__all__ = [
    "MN3ExponentialPotential",
    "MN3Sech2Potential",
]

import abc
import functools as ft
from dataclasses import KW_ONLY
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
from . import miyamotonagai as mn
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField

MN3_K_POS_DENS: Final = jnp.array(
    [
        [0.0036, -0.0330, 0.1117, -0.1335, 0.1749],
        [-0.0131, 0.1090, -0.3035, 0.2921, -5.7976],
        [-0.0048, 0.0454, -0.1425, 0.1012, 6.7120],
        [-0.0158, 0.0993, -0.2070, -0.7089, 0.6445],
        [-0.0319, 0.1514, -0.1279, -0.9325, 2.6836],
        [-0.0326, 0.1816, -0.2943, -0.6329, 2.3193],
    ]
)
MN3_K_NEG_DENS: Final = jnp.array(
    [
        [-0.0090, 0.0640, -0.1653, 0.1164, 1.9487],
        [0.0173, -0.0903, 0.0877, 0.2029, -1.3077],
        [-0.0051, 0.0287, -0.0361, -0.0544, 0.2242],
        [-0.0358, 0.2610, -0.6987, -0.1193, 2.0074],
        [-0.0830, 0.4992, -0.7967, -1.2966, 4.4441],
        [-0.0247, 0.1718, -0.4124, -0.5944, 0.7333],
    ]
)
MN3_B_COEFFS_EXP: Final = jnp.array([-0.269, 1.08, 1.092])
MN3_B_COEFFS_SECH2: Final = jnp.array([-0.033, 0.262, 0.659])


class AbstractMN3Potential(AbstractSinglePotential):
    """A base class for sums of three Miyamoto-Nagai disk potentials."""

    m_tot: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass", doc="Total mass of the potential."
    )

    h_R: AbstractParameter = ParameterField(  # type: ignore[assignment] # noqa: N815
        dimensions="length", doc="Radial (exponential) scale length."
    )

    h_z: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="The vertical scale height."
    )

    positive_density: bool = eqx.field(default=False, static=True)
    """
    If ``True``, the density will be positive everywhere, but is only a good
    approximation of the exponential density within about 5 disk scale lengths. If
    ``False``, the density will be negative in some regions, but is a better
    approximation out to about 7 or 8 disk scale lengths.
    """

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @property
    @abc.abstractmethod
    def _b_coeffs(self) -> Array:
        """The coefficients to use for computing b / h_R."""
        raise NotImplementedError  # pragma: no cover

    def _get_mn_components(
        self, t: gt.BBtQorVSz0, /
    ) -> tuple[
        mn.MiyamotoNagaiPotential, mn.MiyamotoNagaiPotential, mn.MiyamotoNagaiPotential
    ]:
        hR = self.h_R(t)
        hzR = (self.h_z(t) / hR).ustrip(dimensionless)
        K = MN3_K_POS_DENS if self.positive_density else MN3_K_NEG_DENS

        # get b / h_R with fitting functions:
        b_hR = self._b_coeffs @ jnp.array([hzR**3, hzR**2, hzR])

        x = jnp.vander(b_hR[None], N=5)[0]
        param_vec = K @ x

        # use fitting function to get the Miyamoto-Nagai component parameters
        mn_ms = param_vec[:3] * self.m_tot(t)
        mn_as = param_vec[3:] * hR
        mn_b = b_hR * hR
        return (
            mn.MiyamotoNagaiPotential(
                m_tot=mn_ms[0], a=mn_as[0], b=mn_b, units=self.units
            ),
            mn.MiyamotoNagaiPotential(
                m_tot=mn_ms[1], a=mn_as[1], b=mn_b, units=self.units
            ),
            mn.MiyamotoNagaiPotential(
                m_tot=mn_ms[2], a=mn_as[2], b=mn_b, units=self.units
            ),
        )

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.ustrip(AllowValue, self.units["time"], t)
        # TODO: swap to use mn.potential when have an easy way to construct
        # paarams dict.
        return jnp.sum(
            jnp.asarray([mn._potential(xyz, t) for mn in self._get_mn_components(t)]),  # noqa: SLF001
            axis=0,
        )

    @ft.partial(jax.jit)
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
        return MN3_B_COEFFS_EXP


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
        return MN3_B_COEFFS_SECH2
