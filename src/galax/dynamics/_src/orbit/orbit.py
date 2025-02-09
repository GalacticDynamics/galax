"""Orbit objects."""

__all__ = ["Orbit"]

from dataclasses import KW_ONLY, replace
from functools import partial
from typing import Any, ClassVar

import equinox as eqx
import jax
from jaxtyping import Array, Bool, Int
from numpy import ndarray

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.potential as gp
import galax.typing as gt
from .plot_helper import PlotOrbitDescriptor, ProxyOrbit
from galax.typing import BtFloatQuSz0, QuSz1, QuSzTime
from galax.utils._shape import batched_shape, vector_batched_shape


class Orbit(gc.AbstractBasicPhaseSpaceCoordinate):
    """Represents an orbit.

    An orbit is a set of positions and velocities (conjugate momenta) as a
    function of time resulting from the integration of the equations of motion
    in a given potential.

    Examples
    --------
    We can create an orbit by integrating a point mass in a Kepler
    potential:

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.dynamics as gd
    >>> import galax.potential as gp

    >>> potential = gp.KeplerPotential(m_tot=1e12, units="galactic")
    >>> w0 = gc.PhaseSpaceCoordinate(
    ...     q=u.Quantity([8., 0., 0.], "kpc"),
    ...     p=u.Quantity([0., 230, 0.], "km/s"),
    ...     t=u.Quantity(0, "Myr"))
    >>> ts = u.Quantity(jnp.linspace(0., 1., 10), "Gyr")

    >>> orbit = gd.evaluate_orbit(potential, w0, ts)
    >>> orbit
    Orbit(
      q=CartesianPos3D( ... ),
      p=CartesianVel3D( ... ),
      t=Quantity['time'](Array(..., dtype=float64), unit='Myr'),
      frame=SimulationFrame(),
      potential=KeplerPotential( ... ),
      interpolant=None
    )

    >>> orbit = gd.evaluate_orbit(potential, w0, ts, dense=True)
    >>> orbit
    Orbit(
      q=CartesianPos3D( ... ),
      p=CartesianVel3D( ... ),
      t=Quantity['time'](Array(..., dtype=float64), unit='Myr'),
      frame=SimulationFrame(),
      potential=KeplerPotential( ... ),
      interpolant=Interpolant( ... )
    )

    >>> orbit(u.Quantity(0.5, "Gyr"))
    Orbit(
      q=CartesianPos3D( ... ),
      p=CartesianVel3D( ... ),
      t=Quantity['time'](Array([0.5], dtype=float64, ...), unit='Gyr'),
      frame=SimulationFrame(),
      potential=KeplerPotential( ... ),
      interpolant=None
    )

    """

    q: cx.vecs.AbstractPos3D = eqx.field(converter=cx.vector)
    """Positions (x, y, z)."""

    p: cx.vecs.AbstractVel3D = eqx.field(converter=cx.vector)
    r"""Conjugate momenta ($v_x$, $v_y$, $v_z$)."""

    # TODO: consider how this should be vectorized
    t: QuSzTime | QuSz1 = eqx.field(converter=u.Quantity["time"].from_)
    """Array of times corresponding to the positions."""

    _: KW_ONLY

    frame: gc.frames.SimulationFrame  # TODO: support frames
    """The reference frame of the phase-space position."""

    potential: gp.AbstractPotential
    """Potential in which the orbit was integrated."""

    interpolant: gc.PhaseSpaceObjectInterpolant | None = None
    """The interpolation function."""

    def __post_init__(self) -> None:
        """Post-initialization."""
        # Need to ensure t shape is correct. Can be initialized as Vec0.
        if self.t.ndim == 0:
            object.__setattr__(self, "t", jnp.atleast_1d(self.t))

    # -------------------------------------------------------------------------

    plot: ClassVar = PlotOrbitDescriptor()
    """Plot the orbit."""

    # TODO: figure out public API. This is used by `evaluate_orbit`
    @classmethod
    def _from_psp(
        cls,
        w: gc.PhaseSpaceCoordinate,
        t: QuSzTime,
        potential: gp.AbstractPotential,
    ) -> "Orbit":
        """Create an orbit from a phase-space position."""
        return Orbit(
            q=w.q,
            p=w.p,
            t=t,
            frame=w.frame,
            potential=potential,
            interpolant=getattr(w, "interpolant", None),
        )

    # ==========================================================================
    # Interpolation

    def __call__(self, t: BtFloatQuSz0) -> "Orbit":
        """Call the interpolation."""
        interpolant = eqx.error_if(
            self.interpolant,
            self.interpolant is None,
            "Orbit was not integrated with interpolation.",
        )
        qp = interpolant(t)
        return Orbit(
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

    # -------------------------------------------------------------------------
    # Getitem

    @gc.AbstractPhaseSpaceObject.__getitem__.dispatch
    def __getitem__(self: "Orbit", index: tuple[Any, ...]) -> "Orbit":
        """Get a multi-index selection of the orbit.

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
        # Empty selection w[()] should return the same object
        if len(index) == 0:
            return self

        # Handle the time index, subselecting the time component of the index
        # if the time component is a vector.
        tindex = index[-1] if (self.t.ndim == 1 and len(index) == self.ndim) else index

        return replace(self, q=self.q[index], p=self.p[index], t=self.t[tindex])

    @gc.AbstractPhaseSpaceObject.__getitem__.dispatch
    def __getitem__(self: "Orbit", index: slice) -> "Orbit":
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
        tindex = index if self.ndim == 1 else Ellipsis

        return replace(self, q=self.q[index], p=self.p[index], t=self.t[tindex])

    @gc.AbstractPhaseSpaceObject.__getitem__.dispatch
    def __getitem__(self: "Orbit", index: int) -> gc.PhaseSpaceCoordinate:
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
        return gc.PhaseSpaceCoordinate(
            q=self.q[index], p=self.p[index], t=self.t[index]
        )

    @gc.AbstractPhaseSpaceObject.__getitem__.dispatch
    def __getitem__(
        self: "Orbit", index: Int[Array, "..."] | Bool[Array, "..."] | ndarray
    ) -> "Orbit":
        """Get the orbit at specific indices."""
        tindex = Ellipsis if index.ndim < self.ndim else index
        return replace(self, q=self.q[index], p=self.p[index], t=self.t[tindex])

    # ==========================================================================
    # Dynamical quantities

    @partial(jax.jit, inline=True)
    def potential_energy(
        self, potential: gp.AbstractPotential | None = None, /
    ) -> BtFloatQuSz0:
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
        self, potential: "gp.AbstractPotential | None" = None, /
    ) -> BtFloatQuSz0:
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


ProxyOrbit.deliver(Orbit)
