"""galax: Galactic Dynamix in Jax."""

__all__ = ["Orbit", "integrate_orbit", "evaluate_orbit"]

import warnings
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any, final, overload

import equinox as eqx
import jax
import jax.experimental.array_api as xp
import jax.numpy as jnp
import jaxtyping as jt
from astropy.units import Quantity

from galax.coordinates import (
    AbstractPhaseSpaceTimePosition,
    PhaseSpacePosition,
    PhaseSpaceTimePosition,
)
from galax.coordinates._utils import Shaped, getitem_vec1time_index
from galax.potential._potential.base import AbstractPotentialBase
from galax.typing import (
    BatchFloatScalar,
    BatchVec6,
    BroadBatchFloatScalar,
    BroadBatchVecTime3,
    FloatScalar,
    Vec1,
    VecTime,
)
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
    w0: PhaseSpacePosition | PhaseSpaceTimePosition | BatchVec6,
    t: VecTime | Quantity,
    *,
    integrator: Integrator | None = None,
) -> Orbit:
    """Integrate an orbit in a potential, from position `w0` at time ``t[0]``.

    Parameters
    ----------
    pot : :class:`~galax.potential.AbstractPotentialBase`
        The potential in which to integrate the orbit.
    w0 : PhaseSpacePosition | Array[float, (*batch,6)]
        The phase-space position (includes velocity) from which to integrate.

        - :class:`~galax.coordinates.PhaseSpacePosition`[float, (*batch,)]:
            The phase-space position. `w0` will be integrated from ``t[0]`` to
            ``t[1]`` assuming that `w0` is defined at ``t[0]``, returning the
            orbit calculated at `t`.
        - :class:`~galax.coordinates.PhaseSpaceTimePosition`:
            The phase-space position, including a time. The time will be ignored
            and the orbit will be integrated from ``t[0]`` to ``t[1]``,
            returning the orbit calculated at `t`. Note: this will raise a
            warning.
        - Array[float, (*batch, 6)]:
            A :class:`~galax.coordinates.PhaseSpacePosition` will be
            constructed, interpreting the array as the  'q', 'p' (each
            Array[float, (*batch, 3)]) arguments, with 't' set to ``t[0]``.
    t: Array[float, (time,)]
        Array of times at which to compute the orbit. The first element should
        be the initial time and the last element should be the final time and
        the array should be monotonically moving from the first to final time.
        See the Examples section for options when constructing this argument.

        .. note::

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

    Warns
    -----
    UserWarning
        If `w0` is a :class:`~galax.coordinates.PhaseSpaceTimePosition`, a
        warning is raised to indicate that the time is ignored.

    See Also
    --------
    evaluate_orbit
        A higher-level function that computes an orbit. The main difference is
        that `evaluate_orbit` allows for the phase-space position to also include
        a time, which allows for the phase-space position to be defined at a
        different time than the initial time of the integration.

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

    >>> w0 = gc.PhaseSpacePosition(q=[10., 0., 0.], p=[0., 0.1, 0.])
    >>> ts = xp.linspace(0., 1000, 4)  # (1 Gyr, 4 steps)
    >>> orbit = integrate_orbit(potential, w0, ts)
    >>> orbit
    Orbit(
        q=f64[4,3], p=f64[4,3], t=f64[4], potential=KeplerPotential(...)
    )

    Note how there are 4 points in the orbit, corresponding to the 4 requested
    return times. Changing the number of times is easy:

    >>> ts = xp.linspace(0., 1000, 10)  # (1 Gyr, 10 steps)
    >>> orbit = integrate_orbit(potential, w0, ts)
    >>> orbit
    Orbit(
        q=f64[10,3], p=f64[10,3], t=f64[10], potential=KeplerPotential(...)
    )

    We can also integrate a batch of orbits at once:

    >>> w0 = gc.PhaseSpacePosition(q=[[10., 0., 0.], [10., 0., 0.]],
    ...                            p=[[0., 0.1, 0.], [0., 0.2, 0.]])
    >>> orbit = integrate_orbit(potential, w0, ts)
    >>> orbit
    Orbit(
        q=f64[2,10,3], p=f64[2,10,3], t=f64[10], potential=KeplerPotential(...)
    )
    """
    # Parse w0
    if isinstance(w0, PhaseSpaceTimePosition):
        warnings.warn(
            "The time in the input phase-space position is ignored when "
            "integrating the orbit.",
            UserWarning,
            stacklevel=2,
        )
        qp0 = w0.w()
    elif isinstance(w0, PhaseSpacePosition):
        qp0 = w0.w()
    else:
        qp0 = w0

    # Determine the integrator
    # Reboot the integrator to avoid stateful issues
    integrator = replace(integrator) if integrator is not None else _default_integrator

    # Integrate the orbit
    # TODO: ꜛ reduce repeat dimensions of `time`.
    # TODO: push parsing w0 to the integrator-level
    ws = integrator(pot._integrator_F, qp0, t)  # noqa: SLF001

    # Construct the orbit object
    return Orbit(q=ws[..., 0:3], p=ws[..., 3:6], t=t, potential=pot)


_select_w0 = jnp.vectorize(jax.lax.select, signature="(),(6),(6)->(6)")


def _psp2t(
    pspt: BroadBatchFloatScalar, t0: FloatScalar
) -> jt.Shaped[FloatScalar, "*#batch 2"]:
    """Start at PSP time end at t0 for integration from t0."""
    return xp.stack((pspt, xp.full_like(pspt, t0)), axis=-1)


@partial(jax.jit, static_argnames=("integrator",))
def evaluate_orbit(
    pot: AbstractPotentialBase,
    w0: PhaseSpacePosition | PhaseSpaceTimePosition | BatchVec6,
    t: VecTime | Quantity,
    *,
    integrator: Integrator | None = None,
) -> Orbit:
    """Compute an orbit in a potential.

    This method is similar to :meth:`~galax.dynamics.integrate_orbit`, but can
    behave differently when ``w0`` is a
    :class:`~galax.coordinates.PhaseSpacePositionTime`.
    :class:`~galax.coordinates.PhaseSpacePositionTime` includes a time in
    addition to the position (and velocity) information, enabling the orbit to
    be evaluated over a time range that is different from the initial time of
    the position.

    Parameters
    ----------
    pot : :class:`~galax.potential.AbstractPotentialBase`
        The potential in which to integrate the orbit.
    w0 : PhaseSpaceTimePosition | PhaseSpacePosition | Array[float, (*batch, 6)]
        The phase-space position (includes velocity and time) from which to
        integrate. Integration includes the time of the initial position, so be
        sure to set the initial time to the desired value. See the `t` argument
        for more details.

        - :class:`~galax.dynamics.PhaseSpaceTimePosition`[float, (*batch,)]:
            The full phase-space position, including position, velocity, and
            time. `w0` will be integrated from ``w0.t`` to ``t[0]``, then
            integrated from ``t[0]`` to ``t[1]``, returning the orbit calculated
            at `t`.
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

        .. note::

            This is NOT the timesteps to use for integration, which are
            controlled by the `integrator`; the default integrator
            :class:`~galax.integrator.DiffraxIntegrator` uses adaptive
            timesteps.

    integrator : :class:`~galax.integrate.Integrator`, keyword-only
        Integrator to use.  If `None`, the default integrator
        :class:`~galax.integrator.DiffraxIntegrator` is used.  This integrator
        is used twice: once to integrate from `w0.t` to `t[0]` and then from
        `t[0]` to `t[1]`.

    Returns
    -------
    orbit : :class:`~galax.dynamics.Orbit`
        The integrated orbit evaluated at the given times.

    See Also
    --------
    integrate_orbit
        A lower-level function that integrates an orbit.

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

    >>> w0 = gc.PhaseSpaceTimePosition(q=[10., 0., 0.], p=[0., 0.1, 0.], t=-100)
    >>> ts = xp.linspace(0., 1000, 4)  # (1 Gyr, 4 steps)
    >>> orbit = evaluate_orbit(potential, w0, ts)
    >>> orbit
    Orbit(
        q=f64[4,3], p=f64[4,3], t=f64[4], potential=KeplerPotential(...)
    )

    Note how there are 4 points in the orbit, corresponding to the 4 requested
    return times. These are the times at which the orbit is evaluated, not the
    times at which the orbit is integrated. The phase-space position `w0` is
    defined at `t=-100`, but the orbit is integrated from `t=0` to `t=1000`.
    Changing the number of times is easy:

    >>> ts = xp.linspace(0., 1000, 10)  # (1 Gyr, 10 steps)
    >>> orbit = evaluate_orbit(potential, w0, ts)
    >>> orbit
    Orbit(
        q=f64[10,3], p=f64[10,3], t=f64[10], potential=KeplerPotential(...)
    )

    We can also integrate a batch of orbits at once:

    >>> w0 = gc.PhaseSpaceTimePosition(q=[[10., 0., 0.], [10., 0., 0.]],
    ...                                p=[[0., 0.1, 0.], [0., 0.2, 0.]],
    ...                                t=[-100, -150])
    >>> orbit = evaluate_orbit(potential, w0, ts)
    >>> orbit
    Orbit(
        q=f64[2,10,3], p=f64[2,10,3], t=f64[10], potential=KeplerPotential(...)
    )

    :class:`~galax.dynamics.PhaseSpaceTimePosition` has a ``t`` argument for the
    time at which the position is given. As noted earlier, this can be used to
    integrate from a different time than the initial time of the position:

    >>> w0 = gc.PhaseSpaceTimePosition(q=[10., 0., 0.], p=[0., 0.1, 0.], t=0)
    >>> ts = xp.linspace(300, 1000, 8)  # (0.3 to 1 Gyr, 10 steps)
    >>> orbit = evaluate_orbit(potential, w0, ts)
    >>> orbit.q[0]  # doctest: +SKIP
    Array([ 9.779, -0.3102,  0.        ], dtype=float64)

    Note that IS NOT the same as ``w0``. ``w0`` is integrated from ``t=0`` to
    ``t=300`` and then from ``t=300`` to ``t=1000``.

    .. note::

        If you want to reproduce :mod:`gala`'s behavior, you can use
        :class:`~galax.dynamics.PhaseSpacePosition` which does not have a time
        and will assume ``w0`` is defined at `t`[0].
    """
    # Determine the integrator
    # Reboot the integrator to avoid stateful issues
    integrator = replace(integrator) if integrator is not None else _default_integrator

    # Parse w0
    if isinstance(w0, PhaseSpaceTimePosition):
        pspt0 = w0
    elif isinstance(w0, PhaseSpacePosition):
        pspt0 = PhaseSpaceTimePosition(q=w0.q, p=w0.p, t=t[0])
    else:
        pspt0 = PhaseSpaceTimePosition(q=w0[..., 0:3], p=w0[..., 3:6], t=t[0])

    # Need to integrate `w0.t` to `t[0]`.
    # The integral int_a_a is not well defined (can be inf) so we need to
    # handle this case separately.
    # TODO: it may be better to handle this in the integrator itself (by
    # passing either `wt` instead of `w` or the initial time as a separate
    # argument).
    # fmt: off
    w0_ = pspt0.w()
    qp0 = _select_w0(
        pspt0.t == t[0],  # don't integrate if already at the desired time
        w0_,  #               [batch, final t, positions (w/out time)] ⬇
        integrator(pot._integrator_F, w0_, _psp2t(pspt0.t, t[0]))[..., -1, :-1],  # noqa: SLF001
    )
    # fmt: on

    # Integrate the orbit
    w = integrator(pot._integrator_F, qp0, t)  # noqa: SLF001
    # TODO: ꜛ reduce repeat dimensions of `time`.

    # Construct the orbit object
    return Orbit(q=w[..., 0:3], p=w[..., 3:6], t=t, potential=pot)
