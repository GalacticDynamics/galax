from __future__ import annotations

__all__ = ["SubHaloPopulation"]


from typing import Any

import astropy.units as u
import jax
import jax.numpy as xp
import jax.typing as jt
from gala.units import UnitSystem
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

from galdynamix.potential._potential.base import PotentialBase
from galdynamix.utils import jit_method

from .isochrone import Isochrone

usys = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian)


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
    pot_single = Isochrone(m=params["m"], a=params["a"], units=usys)
    return pot_single.energy(q, t)


class SubHaloPopulation(PotentialBase):
    """
    m has length n_subhalo
    a has length n_subhalo
    tq_subhalo_arr has shape t_orbit x n_subhalo x 3
    t_orbit is the array of times the subhalos are integrated over
    """

    m: jt.Array
    a: jt.Array
    tq_subhalo_arr: jt.Array
    t_orbit: jt.Array

    @jit_method()
    def energy(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
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
            )({"m": self.m, "a": self.a}, delta_position, t)
        )
