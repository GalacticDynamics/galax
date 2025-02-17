"""Sample release times.

This is private API.

"""

__all__ = ["ReleaseTimeSampler"]

from functools import partial
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

import quaxed.numpy as jnp
import unxt as u

from .fields import AbstractMassRateField


class ReleaseTimeSampler(eqx.Module):  # type: ignore[misc]
    """Release time sampler.

    This requires a dense mass history solution.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd
    >>> import galax.dynamics.cluster as gdc

    >>> dMdt_fn = gdc.Baumgardt1998MassLossRate()
    >>> pot = gp.MilkyWayPotential2022()
    >>> w0 = gc.PhaseSpaceCoordinate(q=u.Quantity([15, 0, 0], "kpc"),
    ...     p=u.Quantity([0, 100, 0], "km/s"), t=u.Quantity(0, "Gyr"))
    >>> orbit = gd.compute_orbit(pot, w0, u.Quantity([0, 2], "Gyr"), dense=True)

    >>> params = {"orbit": orbit, "m_avg": u.Quantity(3, "Msun"),
    ...           "xi0": 0.001, "alpha": 14.9, "r_hm": u.Quantity(1, "pc")}

    >>> M0 = u.Quantity(1e4, "Msun")
    >>> dMdt_fn(0, M0, params)  # [Msun/Myr]
    Array(-1.06101689, dtype=float64)

    >>> t0, t1 = u.Quantity([0, 2], "Gyr")
    >>> mass_solver = gdc.MassSolver()
    >>> mass_history = mass_solver.solve(dMdt_fn, M0, t0, t1, args=params,
    ...      dense=True, vectorize_interpolation=True)
    >>> mass_history
    Solution( t0=f64[], t1=f64[], ts=f64[1], ys=f64[1],
      interpolation=VectorizedDenseInterpolation( ... ),
      ...
    )

    >>> mass_history.evaluate(0.5)  # [Msun]
    Array(9999.46950028, dtype=float64)

    >>> sampler = gdc.ReleaseTimeSampler(dm_dt=dMdt_fn, m_of_t=mass_history,
    ...                                  units=pot.units)

    >>> key = jr.PRNGKey(0)
    >>> n_stars = 5
    >>> release_times = sampler.sample(key, t0, t1, n_stars=n_stars, mass_params=params)
    >>> release_times.round(2)
    Quantity['time'](Array([0.44, 0.63, 0.68, 0.82, 1.17], dtype=float64), unit='Gyr')

    """

    #: The mass field to use for the release time sampling.
    dm_dt: AbstractMassRateField

    #: Solution m(t) given dm_dt
    m_of_t: dfx.Solution

    #: The unit system
    units: u.AbstractUnitSystem = eqx.field(static=True)

    def __check_init__(self) -> None:
        # TODO: enable discrete times sampler
        if self.m_of_t.interpolation is None:
            msg = "Mass history must be dense."
            raise ValueError(msg)

    @partial(jax.jit, static_argnames=("n_stars",))
    def sample(
        self,
        key: PRNGKeyArray,
        /,
        t0: u.AbstractQuantity,
        t1: u.AbstractQuantity,
        *,
        n_stars: int,
        mass_params: dict[str, Any],
    ) -> Float[Array, "{n_stars}"]:
        """Sample release times.

        Parameters
        ----------
        key : PRNGKeyArray
            Random key for sampling. This key is consumed, so you should split
            it before passing to this function.
        t0, t1 : u.AbstractQuantity
            Initial and final times of the mass history.
        n_stars : int
            Number of stars to sample.
        mass_params : dict[str, Any]
            Parameters for the mass loss rate function.

        Returns
        -------
        Float[Array, (n_stars,)]
            Sorted sampled release times from `t0` and `t1`.

        """
        # TODO: get the times from the dense interpolation. But have to figure
        # out the infs. Then don't need to make this ts array
        ts = jnp.linspace(t0, t1, 2**13)  # TODO: select number
        tsv = ts.ustrip(self.units["time"])
        m_of_t = self.m_of_t.evaluate(tsv)
        escape_rate = -self.dm_dt(ts, m_of_t, mass_params)  # type: ignore[arg-type]
        # TODO: check that escape_rate is positive

        # Normalize to form a probability distribution
        escape_cdf = jnp.cumsum(escape_rate)
        escape_cdf = escape_cdf / escape_cdf[-1]  # Normalize to [0,1]

        # Sample N uniform random numbers and invert the CDF
        U = jr.uniform(key, shape=(n_stars,))
        release_times: Float[Array, "{n_stars}"]
        release_times = jnp.sort(jnp.interp(U, escape_cdf, ts))
        return release_times
