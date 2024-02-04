"""galax: Galactic Dynamix in Jax."""

__all__ = ["Orbit", "integrate_orbit"]

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any, final, overload

import equinox as eqx
import jax
import jax.numpy as jnp
from astropy.units import Quantity

from galax.coordinates import (
    AbstractPhaseSpaceTimePosition,
    PhaseSpacePosition,
    PhaseSpaceTimePosition,
)
from galax.coordinates._utils import Shaped, getitem_vec1time_index
from galax.potential._potential.base import AbstractPotentialBase
from galax.typing import BatchFloatScalar, BatchVec6, BroadBatchVecTime3, Vec1, VecTime
from galax.utils._shape import batched_shape
from galax.utils.dataclasses import converter_float_array

from .integrate import DiffraxIntegrator, Integrator

if TYPE_CHECKING:
    from typing import Self

##############################################################################


@final
class Orbit(AbstractPhaseSpaceTimePosition):
    """Represents an orbit.

    An orbit is a set of ositions and velocities (conjugate momenta) as a
    function of time resulting from the integration of the equations of motion
    in a given potential.
    """

    q: BroadBatchVecTime3 = eqx.field(converter=converter_float_array)
    """Positions (x, y, z)."""

    p: BroadBatchVecTime3 = eqx.field(converter=converter_float_array)
    r"""Conjugate momenta ($v_x$, $v_y$, $v_z$)."""

    # TODO: consider how this should be vectorized
    t: VecTime | Vec1 = eqx.field(converter=converter_float_array)
    """Array of times corresponding to the positions."""

    potential: AbstractPotentialBase
    """Potential in which the orbit was integrated."""

    def __post_init__(self) -> None:
        """Post-initialization."""
        # Need to ensure t shape is correct. Can be Vec0.
        if self.t.ndim == 0:
            object.__setattr__(self, "t", self.t[None])

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

    @overload
    def __getitem__(self, index: int) -> PhaseSpaceTimePosition:
        ...

    @overload
    def __getitem__(self, index: slice | Shaped | tuple[Any, ...]) -> "Self":
        ...

    def __getitem__(self, index: Any) -> "Self | PhaseSpaceTimePosition":
        """Return a new object with the given slice applied."""
        # TODO: return an OrbitSnapshot (or similar) instead of PhaseSpaceTimePosition?
        if isinstance(index, int):
            return PhaseSpaceTimePosition(
                q=self.q[index], p=self.p[index], t=self.t[index]
            )

        if isinstance(index, Shaped):
            msg = "Shaped indexing not yet implemented."
            raise NotImplementedError(msg)

        # Compute subindex
        subindex = getitem_vec1time_index(index, self.t)
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


# TODO: enable setting the default integrator
_default_integrator: Integrator = DiffraxIntegrator()


@partial(jax.jit, static_argnames=("integrator",))
def integrate_orbit(
    pot: AbstractPotentialBase,
    w0: PhaseSpacePosition | BatchVec6,
    t: VecTime | Quantity,
    *,
    integrator: Integrator | None = None,
) -> Orbit:
    """Integrate an orbit in a potential, from position `w0` at time ``t[0]``.

    Parameters
    ----------
    pot : :class:`~galax.potential.AbstractPotentialBase`
        The potential in which to integrate the orbit.
    w0 : PhaseSpacePosition | Array[float, (*batch, 6)]
        The phase-space position (includes velocity) from which to integrate.

        - :class:`~galax.coordinates.PhaseSpacePosition`[float, (*batch,)]:
            The phase-space position. `w0` will be integrated from ``t[0]`` to
            ``t[1]`` assuming that `w0` is defined at ``t[0]``, returning the
            orbit calculated at `t`.
        - Array[float, (*batch, 6)]:
            A :class:`~galax.coordinates.PhaseSpacePosition` will be
            constructed, interpreting the array as the  'q', 'p' (each
            Array[float, (*batch, 3)]) arguments, with 't' set to ``t[0]``.
    t: Array[float, (time,)]
        Array of times at which to compute the orbit. The first element should
        be the initial time and the last element should be the final time and
        the array should be monotonically moving from the first to final time.
        See the Examples section for options when constructing this argument.

        .. warning::

            This is NOT the timesteps to use for integration, which are
            controlled by the `integrator`; the default integrator
            :class:`~galax.integrator.DiffraxIntegrator` uses adaptive
            timesteps.

    integrator : :class:`~galax.integrate.Integrator`, keyword-only
        Integrator to use.  If `None`, the default integrator
        :class:`~galax.integrator.DiffraxIntegrator` is used.

    Returns
    -------
    orbit : :class:`~galax.dynamics.Orbit`
        The integrated orbit evaluated at the given times.

    Examples
    --------
    We start by integrating a single orbit in the potential of a point mass.  A
    few standard imports are needed:

    >>> import astropy.units as u
    >>> import jax.experimental.array_api as xp  # preferred over `jax.numpy`
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> from galax.units import galactic

    We can then create the point-mass' potential, with galactic units:

    >>> potential = gp.KeplerPotential(m=1e12 * u.Msun, units=galactic)

    We can then integrate an initial phase-space position in this potential to
    get an orbit:

    >>> w0 = gc.PhaseSpacePosition(xp.asarray([10., 0., 0.]),  # (x, v) [galactic]
    ...                            xp.asarray([0., 0.1, 0.]))
    >>> ts = xp.linspace(0., 1000, 4)  # (1 Gyr, 4 steps)
    >>> orbit = potential.integrate_orbit(w0, ts)
    >>> orbit
    Orbit(
        q=f64[4,3], p=f64[4,3], t=f64[4], potential=KeplerPotential(...)
    )

    Note how there are 4 points in the orbit, corresponding to the 4 requested
    return times. Changing the number of times is easy:

    >>> ts = xp.linspace(0., 1000, 10)  # (1 Gyr, 10 steps)
    >>> orbit = potential.integrate_orbit(w0, ts)
    >>> orbit
    Orbit(
        q=f64[10,3], p=f64[10,3], t=f64[10], potential=KeplerPotential(...)
    )

    We can also integrate a batch of orbits at once:

    >>> w0 = gc.PhaseSpacePosition(xp.asarray([[10., 0., 0.], [10., 0., 0.]]),
    ...                            xp.asarray([[0., 0.1, 0.], [0., 0.2, 0.]]))
    >>> orbit = potential.integrate_orbit(w0, ts)
    >>> orbit
    Orbit(
        q=f64[2,10,3], p=f64[2,10,3], t=f64[10], potential=KeplerPotential(...)
    )
    """
    # Parse w0
    qp0 = w0.w() if isinstance(w0, PhaseSpacePosition) else w0

    # Determine the integrator
    # Reboot the integrator to avoid stateful issues
    integrator = replace(integrator) if integrator is not None else _default_integrator

    # Integrate the orbit
    # TODO: ꜛ reduce repeat dimensions of `time`.
    # TODO: push parsing w0 to the integrator-level
    ws = integrator(pot._integrator_F, qp0, t)  # noqa: SLF001

    # Construct the orbit object
    return Orbit(q=ws[..., 0:3], p=ws[..., 3:6], t=t, potential=pot)
