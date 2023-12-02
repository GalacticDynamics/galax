"""galdynamix: Galactic Dynamix in Jax."""

from __future__ import annotations

__all__ = [
    "SubHaloPopulation",
]

from typing import Any

import equinox as eqx
import jax
import jax.numpy as xp
import jax.typing as jt
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

from galdynamix.potential._potential.base import AbstractPotential
from galdynamix.potential._potential.builtin import Isochrone
from galdynamix.potential._potential.param import AbstractParameter, ParameterField
from galdynamix.units import galactic
from galdynamix.utils import partial_jit

# -------------------------------------------------------------------


@jax.jit  # type: ignore[misc]
def get_splines(x_eval: jt.Array, x: jt.Array, y: jt.Array) -> Any:
    return InterpolatedUnivariateSpline(x, y, k=3)(x_eval)


@jax.jit  # type: ignore[misc]
def single_subhalo_potential(
    params: dict[str, jt.Array], q: jt.Array, /, t: jt.Array
) -> jt.Array:
    """Potential for a single subhalo.

    TODO: custom unit specification/subhalo potential specficiation.
    Currently supports units kpc, Myr, Msun, rad.
    """
    pot_single = Isochrone(m=params["m"], a=params["a"], units=galactic)
    return pot_single.potential_energy(q, t)


class SubHaloPopulation(AbstractPotential):
    """m has length n_subhalo.

    a has length n_subhalo
    tq_subhalo_arr has shape t_orbit x n_subhalo x 3
    t_orbit is the array of times the subhalos are integrated over
    """

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    a: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    tq_subhalo_arr: jt.Array = eqx.field(converter=xp.asarray)
    t_orbit: jt.Array = eqx.field(converter=xp.asarray)

    @partial_jit()
    def potential_energy(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        # expect n_subhalo x-positions
        x_at_t_eval = get_splines(t, self.t_orbit, self.tq_subhalo_arr[:, :, 0])
        # expect n_subhalo y-positions
        y_at_t_eval = get_splines(t, self.t_orbit, self.tq_subhalo_arr[:, :, 1])
        # expect n_subhalo z-positions
        z_at_t_eval = get_splines(t, self.t_orbit, self.tq_subhalo_arr[:, :, 2])

        # n_subhalo x 3: the position of all subhalos at time t
        subhalo_locations = xp.vstack([x_at_t_eval, y_at_t_eval, z_at_t_eval]).T

        delta_position = q - subhalo_locations  # n_subhalo x 3
        # sum over potential due to all subhalos in the field by vmapping over
        # m, a, and delta_position
        return xp.sum(
            jax.vmap(
                single_subhalo_potential,
                in_axes=(({"m": 0, "a": 0}, 0, None)),
            )({"m": self.m(t), "a": self.a(t)}, delta_position, t),
        )
