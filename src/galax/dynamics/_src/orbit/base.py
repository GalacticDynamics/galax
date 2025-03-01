"""Orbit objects."""

__all__ = ["AbstractOrbit"]

from dataclasses import KW_ONLY, replace
from functools import partial
from typing import Any, ClassVar

import equinox as eqx
import jax
from jaxtyping import Array, Bool, Int
from numpy import ndarray
from plum import dispatch

import coordinax as cx
import quaxed.numpy as jnp

import galax._custom_types as gt
import galax.coordinates as gc
import galax.potential as gp
from .plot_helper import PlotOrbitDescriptor, ProxyAbstractOrbit
from galax.utils._shape import batched_shape, vector_batched_shape


class AbstractOrbit(gc.AbstractBasicPhaseSpaceCoordinate):
    """ABC for Orbit."""

    #: Positions
    q: eqx.AbstractVar[cx.vecs.AbstractPos3D]

    #: Velocities
    p: eqx.AbstractVar[cx.vecs.AbstractVel3D]

    #: Times
    t: eqx.AbstractVar[gt.QuSzTime | gt.QuSz1]

    _: KW_ONLY

    #: The reference frame of the phase-space coordinate.
    frame: eqx.AbstractVar[gc.frames.SimulationFrame]  # TODO: support frames

    #: The potential in which the orbit was integrated.
    potential: eqx.AbstractVar[gp.AbstractPotential]

    #: The interpolant of the orbit.
    interpolant: eqx.AbstractVar[gc.PhaseSpaceObjectInterpolant | None]

    _GETITEM_DYNAMIC_FILTER_SPEC: ClassVar = (True, True, True, False, False, False)
    _GETITEM_TIME_FILTER_SPEC: ClassVar = (False, False, True, False, False, False)

    # -------------------------------------------------------------------------

    plot: ClassVar = PlotOrbitDescriptor()
    """Plot the orbit."""

    # ==========================================================================
    # Interpolation

    def __call__(self, t: gt.BtFloatQuSz0) -> "AbstractOrbit":
        """Evaluate the interpolation."""
        interpolant = eqx.error_if(
            self.interpolant,
            self.interpolant is None,
            f"{self.__class__.__name__} was not integrated with interpolation.",
        )
        qp = interpolant(t)
        return replace(
            self,
            q=qp.q,
            p=qp.p,
            t=qp.t,
            potential=self.potential,
            interpolant=None,
            frame=self.frame,
        )

    # ==========================================================================
    # Array properties

    @property
    def _shape_tuple(self) -> tuple[gt.Shape, gc.ComponentShapeTuple]:
        """Batch, component shape."""
        qbatch, qshape = vector_batched_shape(self.q)
        pbatch, pshape = vector_batched_shape(self.p)
        tbatch, _ = batched_shape(self.t, expect_ndim=1)
        batch_shape = jnp.broadcast_shapes(qbatch, pbatch, tbatch)
        return batch_shape, gc.ComponentShapeTuple(q=qshape, p=pshape, t=1)

    # ==========================================================================
    # Dynamical quantities

    @partial(jax.jit, inline=True)
    def potential_energy(
        self, potential: gp.AbstractPotential | None = None, /
    ) -> gt.BtFloatQuSz0:
        r"""Return the specific potential energy.

        .. math::

            E_\Phi = \Phi(\boldsymbol{q})

        Parameters
        ----------
        potential : `galax.potential.AbstractPotential` | None
            The potential object to compute the energy from. If `None`
            (default), use the potential attribute of the orbit.

        Returns
        -------
        E : Array[float, (*batch,)]
            The specific potential energy.
        """
        if potential is None:
            return self.potential.potential(self.q, t=self.t)
        return potential.potential(self.q, t=self.t)

    @partial(jax.jit, inline=True)
    def total_energy(
        self, potential: gp.AbstractPotential | None = None, /
    ) -> gt.BtFloatQuSz0:
        r"""Return the specific total energy.

        .. math::

            E_K = \frac{1}{2} \\, |\boldsymbol{v}|^2
            E_\Phi = \Phi(\boldsymbol{q})
            E = E_K + E_\Phi

        Parameters
        ----------
        potential : `galax.potential.AbstractPotential` | None
            The potential object to compute the energy from. If `None`
            (default), use the potential attribute of the orbit.

        Returns
        -------
        E : Array[float, (*batch,)]
            The kinetic energy.
        """
        return self.kinetic_energy() + self.potential_energy(potential)


ProxyAbstractOrbit.deliver(AbstractOrbit)


#####################################################################

# =========================================================
# `__getitem__`


@dispatch
def _psc_getitem_time_index(orbit: AbstractOrbit, index: tuple[Any, ...], /) -> Any:
    """Return the time index slicer. Default is to return as-is.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")
    >>> w0 = gc.PhaseSpaceCoordinate(
    ...     q=u.Quantity([8., 0., 0.], "kpc"),
    ...     p=u.Quantity([0., 230, 0.], "km/s"),
    ...     t=u.Quantity(0, "Myr"))
    >>> ts = u.Quantity(jnp.linspace(0, 1, 10), "Gyr")
    >>> orbit = gd.evaluate_orbit(pot, w0, ts)

    >>> orbit[()] is orbit
    True

    >>> orbit[(slice(None),)]
    Orbit(
        q=CartesianPos3D(
        x=Quantity[...](value=f64[10], unit=Unit("kpc")),
        ... ),
        p=CartesianVel3D(
        x=Quantity[...]( value=f64[10], unit=Unit("kpc / Myr") ),
        ... ),
        t=Quantity['time'](Array(..., dtype=float64), unit='Myr'),
        frame=SimulationFrame(),
        potential=KeplerPotential( ... ),
        interpolant=None
    )

    """
    # Handle the time index, subselecting the time component of the index
    # if the time component is a vector.
    return index[-1] if (orbit.t.ndim == 1 and len(index) == orbit.ndim) else index


@dispatch
def _psc_getitem_time_index(orbit: AbstractOrbit, index: slice, /) -> Any:
    """Slice the orbit.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")
    >>> w0 = gc.PhaseSpaceCoordinate(
    ...     q=u.Quantity([8., 0., 0.], "kpc"),
    ...     p=u.Quantity([0., 230, 0.], "km/s"),
    ...     t=u.Quantity(0, "Myr"))
    >>> ts = u.Quantity(jnp.linspace(0, 1, 11), "Gyr")
    >>> orbit = gd.evaluate_orbit(pot, w0, ts)

    >>> orbit[0:2]
    Orbit(
        q=CartesianPos3D(
        x=Quantity[...](value=f64[2], unit=Unit("kpc")),
        ...
        ),
        p=CartesianVel3D(
        x=Quantity[...]( value=f64[2], unit=Unit("kpc / Myr") ),
        ...
        ),
        t=Quantity['time'](Array([  0., 100.], dtype=float64), unit='Myr'),
        frame=SimulationFrame(),
        potential=KeplerPotential( ... ),
        interpolant=None
    )

    """
    # The index only applies to the time component if the slice reaches
    # the last axis, which is the time axis. Otherwise, the slice applies
    # to all components.
    return index if orbit.ndim == 1 else Ellipsis


@dispatch
def _psc_getitem_time_index(
    orbit: AbstractOrbit, index: Int[Array, "..."] | Bool[Array, "..."] | ndarray, /
) -> Any:
    """Get the orbit at specific indices.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot=gp.KeplerPotential(m_tot=1e12, units="galactic")
    >>> orbit = gd.Orbit(
    ...     q=u.Quantity([[0, 1, 2]], "kpc"), p=u.Quantity([[4, 5, 6]], "km/s"),
    ...     t=u.Quantity(0, "Gyr"), potential=pot,
    ...     frame=gc.frames.simulation_frame)
    >>> print(orbit)
    Orbit(
        q=<CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [[0 1 2]]>,
        p=<CartesianVel3D (x[km / s], y[km / s], z[km / s])
            [[4 5 6]]>,
        t=Quantity['time'](Array(0, dtype=int64, ...), unit='Gyr'),
        ...)
    >>> orbit.ndim
    1

    When index.ndim < orbit.ndim:

    >>> print(orbit[jnp.array(0)])
    Orbit(
        q=<CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [0 1 2]>,
        p=<CartesianVel3D (x[km / s], y[km / s], z[km / s])
            [4 5 6]>,
        t=Quantity['time'](Array(0, dtype=int64, ...), unit='Gyr'),
        ...)

    Otherwise:

    TODO: something broken when index.ndim >= orbit.ndim.

    """
    return Ellipsis if index.ndim < orbit.ndim else index


@gc.AbstractPhaseSpaceObject.__getitem__.dispatch  # type: ignore[attr-defined,misc]
def getitem(self: AbstractOrbit, index: int) -> gc.PhaseSpaceCoordinate:
    """Get the orbit at a specific time.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")
    >>> w0 = gc.PhaseSpaceCoordinate(
    ...     q=u.Quantity([8., 0., 0.], "kpc"),
    ...     p=u.Quantity([0., 230, 0.], "km/s"),
    ...     t=u.Quantity(0, "Myr"))
    >>> ts = u.Quantity([0., 1.], "Gyr")
    >>> orbit = gd.evaluate_orbit(pot, w0, ts)

    >>> orbit[0]
    PhaseSpaceCoordinate(
        q=CartesianPos3D( ... ),
        p=CartesianVel3D( ... ),
        t=Quantity['time'](Array(0., dtype=float64), unit='Myr'),
        frame=SimulationFrame()
    )
    >>> orbit[0].t
    Quantity['time'](Array(0., dtype=float64), unit='Myr')

    """
    return gc.PhaseSpaceCoordinate(q=self.q[index], p=self.p[index], t=self.t[index])
