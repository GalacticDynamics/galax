"""galdynamix: Galactic Dynamix in Jax"""
# ruff: noqa: F403

from __future__ import annotations

from typing import Any

__all__ = [
    "MiyamotoNagaiDisk",
    "BarPotential",
    "Isochrone",
    "NFWPotential",
    "SubHaloPopulation",
]

from dataclasses import KW_ONLY

import equinox as eqx
import jax
import jax.numpy as xp
import jax.typing as jt
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

from galdynamix.potential._potential.base import AbstractPotential
from galdynamix.potential._potential.param import AbstractParameter, ParameterField
from galdynamix.units import galactic
from galdynamix.utils import partial_jit
from galdynamix.utils.dataclasses import field

# -------------------------------------------------------------------


class MiyamotoNagaiDisk(AbstractPotential):
    m: AbstractParameter = ParameterField(physical_type="mass")  # type: ignore[assignment]
    a: AbstractParameter = ParameterField(physical_type="length")  # type: ignore[assignment]
    b: AbstractParameter = ParameterField(physical_type="length")  # type: ignore[assignment]

    @partial_jit()
    def potential_energy(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        R2 = q[0] ** 2 + q[1] ** 2
        return (
            -self._G
            * self.m(t)
            / xp.sqrt(R2 + xp.square(xp.sqrt(q[2] ** 2 + self.b(t) ** 2) + self.a(t)))
        )


# -------------------------------------------------------------------


class BarPotential(AbstractPotential):
    """
    Rotating bar potentil, with hard-coded rotation.
    Eq 8a in https://articles.adsabs.harvard.edu/pdf/1992ApJ...397...44L
    Rz according to https://en.wikipedia.org/wiki/Rotation_matrix
    """

    m: AbstractParameter = ParameterField(physical_type="mass")  # type: ignore[assignment]
    a: AbstractParameter = ParameterField(physical_type="length")  # type: ignore[assignment]
    b: AbstractParameter = ParameterField(physical_type="length")  # type: ignore[assignment]
    c: AbstractParameter = ParameterField(physical_type="length")  # type: ignore[assignment]
    Omega: AbstractParameter = ParameterField(physical_type="frequency")  # type: ignore[assignment]

    @partial_jit()
    def potential_energy(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        ## First take the simulation frame coordinates and rotate them by Omega*t
        ang = -self.Omega(t) * t
        Rot_mat = xp.array(
            [
                [xp.cos(ang), -xp.sin(ang), 0],
                [xp.sin(ang), xp.cos(ang), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        # Rot_inv = xp.linalg.inv(Rot_mat)
        q_corot = xp.matmul(Rot_mat, q)

        a = self.a(t)
        b = self.b(t)
        c = self.c(t)
        T_plus = xp.sqrt(
            (a + q_corot[0]) ** 2
            + q_corot[1] ** 2
            + (b + xp.sqrt(c**2 + q_corot[2] ** 2)) ** 2
        )
        T_minus = xp.sqrt(
            (a - q_corot[0]) ** 2
            + q_corot[1] ** 2
            + (b + xp.sqrt(c**2 + q_corot[2] ** 2)) ** 2
        )

        # potential in a corotating frame
        return (self._G * self.m(t) / (2.0 * a)) * xp.log(
            (q_corot[0] - a + T_minus) / (q_corot[0] + a + T_plus)
        )


# -------------------------------------------------------------------


class Isochrone(AbstractPotential):
    m: AbstractParameter = ParameterField(physical_type="mass")  # type: ignore[assignment]
    a: AbstractParameter = ParameterField(physical_type="length")  # type: ignore[assignment]

    @partial_jit()
    def potential_energy(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        r = xp.linalg.norm(q, axis=0)
        a = self.a(t)
        return -self._G * self.m(t) / (a + xp.sqrt(r**2 + a**2))


# -------------------------------------------------------------------


class NFWPotential(AbstractPotential):
    """NFW Potential."""

    m: AbstractParameter = ParameterField(physical_type="mass")  # type: ignore[assignment]
    r_s: AbstractParameter = ParameterField(physical_type="length")  # type: ignore[assignment]
    _: KW_ONLY
    softening_length: jt.Array = field(
        default=0.001, static=True, physical_type="length"
    )

    @partial_jit()
    def potential_energy(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        v_h2 = -self._G * self.m(t) / self.r_s(t)
        m = xp.sqrt(
            q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + self.softening_length
        ) / self.r_s(t)
        return v_h2 * xp.log(1.0 + m) / m


# -------------------------------------------------------------------


@jax.jit  # type: ignore[misc]
def get_splines(x_eval: jt.Array, x: jt.Array, y: jt.Array) -> Any:
    return InterpolatedUnivariateSpline(x, y, k=3)(x_eval)


@jax.jit  # type: ignore[misc]
def single_subhalo_potential(
    params: dict[str, jt.Array], q: jt.Array, /, t: jt.Array
) -> jt.Array:
    """
    Potential for a single subhalo
    TODO: custom unit specification/subhalo potential specficiation.
    Currently supports units kpc, Myr, Msun, rad.
    """
    pot_single = Isochrone(m=params["m"], a=params["a"], units=galactic)
    return pot_single.potential_energy(q, t)


class SubHaloPopulation(AbstractPotential):
    """
    m has length n_subhalo
    a has length n_subhalo
    tq_subhalo_arr has shape t_orbit x n_subhalo x 3
    t_orbit is the array of times the subhalos are integrated over
    """

    m: AbstractParameter = ParameterField(physical_type="mass")  # type: ignore[assignment]
    a: AbstractParameter = ParameterField(physical_type="length")  # type: ignore[assignment]
    tq_subhalo_arr: jt.Array = eqx.field(converter=xp.asarray)
    t_orbit: jt.Array = eqx.field(converter=xp.asarray)

    @partial_jit()
    def potential_energy(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        x_at_t_eval = get_splines(
            t, self.t_orbit, self.tq_subhalo_arr[:, :, 0]
        )  # expect n_subhalo x-positions
        y_at_t_eval = get_splines(
            t, self.t_orbit, self.tq_subhalo_arr[:, :, 1]
        )  # expect n_subhalo y-positions
        z_at_t_eval = get_splines(
            t, self.t_orbit, self.tq_subhalo_arr[:, :, 2]
        )  # expect n_subhalo z-positions

        subhalo_locations = xp.vstack(
            [x_at_t_eval, y_at_t_eval, z_at_t_eval]
        ).T  # n_subhalo x 3: the position of all subhalos at time t

        delta_position = q - subhalo_locations  # n_subhalo x 3
        # sum over potential due to all subhalos in the field by vmapping over m, a, and delta_position
        ##dct = {'m': self.m, 'a': self.a,}
        return xp.sum(
            jax.vmap(
                single_subhalo_potential,
                in_axes=(({"m": 0, "a": 0}, 0, None)),
            )({"m": self.m(t), "a": self.a(t)}, delta_position, t)
        )
