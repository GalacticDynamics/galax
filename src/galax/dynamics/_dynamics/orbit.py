"""galax: Galactic Dynamix in Jax."""

__all__ = ["Orbit", "integrate_orbit"]

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
from astropy.units import Quantity

from galax.integrate import Integrator
from galax.potential._potential.base import AbstractPotentialBase
from galax.typing import BatchFloatScalar, BatchVec6, BroadBatchVec3, VecTime
from galax.utils._shape import batched_shape
from galax.utils.dataclasses import converter_float_array

from .base import AbstractPhaseSpacePosition
from .utils import getitem_vectime_index

if TYPE_CHECKING:
    from typing import Self

##############################################################################


@final
class Orbit(AbstractPhaseSpacePosition):
    """Represents an orbit.

    Represents an orbit: positions and velocities (conjugate momenta) as a
    function of time.

    """

    q: BroadBatchVec3 = eqx.field(converter=converter_float_array)
    """Positions (x, y, z)."""

    p: BroadBatchVec3 = eqx.field(converter=converter_float_array)
    r"""Conjugate momenta ($v_x$, $v_y$, $v_z$)."""

    # TODO: consider how this should be vectorized
    t: VecTime = eqx.field(converter=converter_float_array)
    """Array of times corresponding to the positions."""

    potential: AbstractPotentialBase
    """Potential in which the orbit was integrated."""

    # ==========================================================================
    # Array properties

    @property
    def _shape_tuple(self) -> tuple[tuple[int, ...], tuple[int, int, int]]:
        """Batch, component shape."""
        qbatch, qshape = batched_shape(self.q, expect_ndim=1)
        pbatch, pshape = batched_shape(self.p, expect_ndim=1)
        tbatch, _ = batched_shape(self.t, expect_ndim=1)
        batch_shape = jnp.broadcast_shapes(qbatch, pbatch, tbatch)
        array_shape = qshape + pshape + (1,)
        return batch_shape, array_shape

    def __getitem__(self, index: Any) -> "Self":
        """Return a new object with the given slice applied."""
        # Compute subindex
        subindex = getitem_vectime_index(index, self.t)
        # Apply slice
        return replace(self, q=self.q[index], p=self.p[index], t=self.t[subindex])

    # ==========================================================================
    # Dynamical quantities

    @partial(jax.jit)
    def potential_energy(
        self, potential: AbstractPotentialBase | None = None, /
    ) -> BatchFloatScalar:
        r"""Return the specific potential energy.

        .. math::

            E_\Phi = \Phi(\boldsymbol{q})

        Parameters
        ----------
        potential : `galax.potential.AbstractPotentialBase`
            The potential object to compute the energy from.

        Returns
        -------
        E : Array[float, (*batch,)]
            The specific potential energy.
        """
        if potential is None:
            return self.potential.potential_energy(self.q, t=self.t)
        return potential.potential_energy(self.q, t=self.t)

    @partial(jax.jit)
    def energy(
        self, potential: "AbstractPotentialBase | None" = None, /
    ) -> BatchFloatScalar:
        r"""Return the specific total energy.

        .. math::

            E_K = \frac{1}{2} \\, |\boldsymbol{v}|^2
            E_\Phi = \Phi(\boldsymbol{q})
            E = E_K + E_\Phi

        Returns
        -------
        E : Array[float, (*batch,)]
            The kinetic energy.
        """
        return self.kinetic_energy() + self.potential_energy(potential)


##############################################################################


@partial(jax.jit, static_argnames=("integrator",))
def integrate_orbit(
    pot: AbstractPotentialBase,
    w0: BatchVec6,
    t: VecTime | Quantity,
    *,
    integrator: Integrator,
) -> Orbit:
    """Integrate an orbit in potential.

    Parameters
    ----------
    pot : :class:`~galax.potential.AbstractPotentialBase`
        The potential in which to integrate the orbit.
    w0 : Array[float, (*batch, 6)]
        Initial position and velocity.
    t: Array[float, (time,)]
        Array of times at which to compute the orbit. The first element
        should be the initial time and the last element should be the final
        time and the array should be monotonically moving from the first to
        final time.  See the Examples section for options when constructing
        this argument.

        .. warning::

            This is NOT the timesteps to use for integration, which are
            controlled by the `integrator`; the default integrator
            :class:`~galax.integrator.DiffraxIntegrator` uses adaptive
            timesteps.

    integrator : :class:`~galax.integrate.Integrator`, keyword-only
        Integrator to use.

    Returns
    -------
    orbit : :class:`~galax.dynamics.Orbit`
        The integrated orbit evaluated at the given times.

    Examples
    --------
    We start by integrating a single orbit in the potential of a point mass.
    A few standard imports are needed:

    >>> import astropy.units as u
    >>> import jax.experimental.array_api as xp  # preferred over `jax.numpy`
    >>> import galax.potential as gp
    >>> from galax.units import galactic

    We can then create the point-mass' potential, with galactic units:

    >>> potential = gp.KeplerPotential(m=1e12 * u.Msun, units=galactic)

    We can then integrate an initial phase-space position in this potential
    to get an orbit:

    >>> xv0 = xp.asarray([10., 0., 0., 0., 0.1, 0.])  # (x, v) galactic units
    >>> ts = xp.linspace(0., 1000, 4)  # (1 Gyr, 4 steps)
    >>> orbit = potential.integrate_orbit(xv0, ts)
    >>> orbit
    Orbit(
        q=f64[4,3], p=f64[4,3], t=f64[4], potential=KeplerPotential(...)
    )

    Note how there are 4 points in the orbit, corresponding to the 4 steps.
    Changing the number of steps is easy:

    >>> ts = xp.linspace(0., 1000, 10)  # (1 Gyr, 4 steps)
    >>> orbit = potential.integrate_orbit(xv0, ts)
    >>> orbit
    Orbit(
        q=f64[10,3], p=f64[10,3], t=f64[10], potential=KeplerPotential(...)
    )

    We can also integrate a batch of orbits at once:

    >>> xv0 = xp.asarray([[10., 0., 0., 0., 0.1, 0.], [10., 0., 0., 0., 0.2, 0.]])
    >>> orbit = potential.integrate_orbit(xv0, ts)
    >>> orbit
    Orbit(
        q=f64[2,10,3], p=f64[2,10,3], t=f64[10], potential=KeplerPotential(...)
    )
    """
    # Reboot the integrator to avoid stateful issues
    integrator = replace(integrator)

    # Integrate the orbit
    ws = integrator(pot._integrator_F, w0, t)  # noqa: SLF001
    # TODO: ꜛ reduce repeat dimensions of `time`.

    # Construct the orbit object
    return Orbit(q=ws[..., 0:3], p=ws[..., 3:6], t=t, potential=pot)
