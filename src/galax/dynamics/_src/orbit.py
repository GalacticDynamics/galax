"""Orbit objects."""

__all__ = ["Orbit"]

from dataclasses import KW_ONLY, replace
from functools import partial
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int
from numpy import ndarray

import coordinax as cx
from unxt import Quantity

import galax.coordinates as gc
import galax.potential as gp
import galax.typing as gt
from .orbit_plot import PlotOrbitDescriptor, ProxyOrbit
from galax.typing import BatchFloatQScalar, QVec1, QVecTime
from galax.utils._shape import batched_shape, vector_batched_shape


class Orbit(gc.AbstractPhaseSpacePosition):
    """Represents an orbit.

    An orbit is a set of positions and velocities (conjugate momenta) as a
    function of time resulting from the integration of the equations of motion
    in a given potential.

    Examples
    --------
    We can create an orbit by integrating a point mass in a Kepler
    potential:

    >>> import jax.numpy as jnp
    >>> from unxt import Quantity
    >>> import galax.coordinates as gc
    >>> import galax.dynamics as gd
    >>> import galax.potential as gp

    >>> potential = gp.KeplerPotential(m_tot=1e12, units="galactic")
    >>> w0 = gc.PhaseSpacePosition(
    ...     q=Quantity([8., 0., 0.], "kpc"),
    ...     p=Quantity([0., 230, 0.], "km/s"),
    ...     t=Quantity(0, "Myr"))
    >>> ts = Quantity(jnp.linspace(0., 1., 10), "Gyr")

    >>> orbit = gd.evaluate_orbit(potential, w0, ts)
    >>> orbit
    Orbit(
      q=CartesianPosition3D( ... ),
      p=CartesianVelocity3D( ... ),
      t=Quantity[...](value=f64[10], unit=Unit("Myr")),
      potential=KeplerPotential( ... ),
      interpolant=None
    )

    >>> orbit = gd.evaluate_orbit(potential, w0, ts, interpolated=True)
    >>> orbit
    Orbit(
      q=CartesianPosition3D( ... ),
      p=CartesianVelocity3D( ... ),
      t=Quantity[...](value=f64[10], unit=Unit("Myr")),
      potential=KeplerPotential( ... ),
      interpolant=Interpolant( ... )
    )

    >>> orbit(Quantity(0.5, "Gyr"))
    Orbit(
      q=CartesianPosition3D( ... ),
      p=CartesianVelocity3D( ... ),
      t=Quantity[...](value=f64[1], unit=Unit("Gyr")),
      potential=KeplerPotential( ... ),
      interpolant=None
    )

    """

    q: cx.AbstractPosition3D = eqx.field(converter=cx.AbstractPosition3D.constructor)
    """Positions (x, y, z)."""

    p: cx.AbstractVelocity3D = eqx.field(converter=cx.AbstractVelocity3D.constructor)
    r"""Conjugate momenta ($v_x$, $v_y$, $v_z$)."""

    # TODO: consider how this should be vectorized
    t: QVecTime | QVec1 = eqx.field(converter=Quantity["time"].constructor)
    """Array of times corresponding to the positions."""

    _: KW_ONLY

    potential: gp.AbstractPotentialBase
    """Potential in which the orbit was integrated."""

    interpolant: gc.PhaseSpacePositionInterpolant | None = None
    """The interpolation function."""

    def __post_init__(self) -> None:
        """Post-initialization."""
        # Need to ensure t shape is correct. Can be initialized as Vec0.
        if self.t.ndim == 0:
            object.__setattr__(self, "t", self.t[None])

    # -------------------------------------------------------------------------

    plot: ClassVar = PlotOrbitDescriptor()
    """Plot the orbit."""

    # ==========================================================================
    # Interpolation

    def __call__(self, t: BatchFloatQScalar) -> "Orbit":
        """Call the interpolation."""
        interpolant = eqx.error_if(
            self.interpolant,
            self.interpolant is None,
            "Orbit was not integrated with interpolation.",
        )
        qp = interpolant(t)
        return Orbit(q=qp.q, p=qp.p, t=qp.t, potential=self.potential, interpolant=None)

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

    @gc.AbstractBasePhaseSpacePosition.__getitem__.dispatch
    def __getitem__(self: "Orbit", index: tuple[Any, ...]) -> "Orbit":
        """Get a multi-index selection of the orbit.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> from unxt import Quantity
        >>> import galax.coordinates as gc
        >>> import galax.potential as gp
        >>> import galax.dynamics as gd

        >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")
        >>> w0 = gc.PhaseSpacePosition(
        ...     q=Quantity([8., 0., 0.], "kpc"),
        ...     p=Quantity([0., 230, 0.], "km/s"),
        ...     t=Quantity(0, "Myr"))
        >>> ts = Quantity(jnp.linspace(0, 1, 10), "Gyr")
        >>> orbit = gd.evaluate_orbit(pot, w0, ts)

        >>> orbit[()] is orbit
        True

        >>> orbit[(slice(None),)]
        Orbit(
          q=CartesianPosition3D(
            x=Quantity[...](value=f64[10], unit=Unit("kpc")),
            ... ),
          p=CartesianVelocity3D(
            d_x=Quantity[...]( value=f64[10], unit=Unit("kpc / Myr") ),
            ... ),
          t=Quantity[PhysicalType('time')](value=f64[10], unit=Unit("Myr")),
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

    @gc.AbstractBasePhaseSpacePosition.__getitem__.dispatch
    def __getitem__(self: "Orbit", index: slice) -> "Orbit":
        """Slice the orbit.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> from unxt import Quantity
        >>> import galax.coordinates as gc
        >>> import galax.potential as gp
        >>> import galax.dynamics as gd

        >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")
        >>> w0 = gc.PhaseSpacePosition(
        ...     q=Quantity([8., 0., 0.], "kpc"),
        ...     p=Quantity([0., 230, 0.], "km/s"),
        ...     t=Quantity(0, "Myr"))
        >>> ts = Quantity(jnp.linspace(0, 1, 10), "Gyr")
        >>> orbit = gd.evaluate_orbit(pot, w0, ts)

        >>> orbit[0:2]
        Orbit(
          q=CartesianPosition3D(
            x=Quantity[...](value=f64[2], unit=Unit("kpc")),
            ...
          ),
          p=CartesianVelocity3D(
            d_x=Quantity[...]( value=f64[2], unit=Unit("kpc / Myr") ),
            ...
          ),
          t=Quantity[...](value=f64[2], unit=Unit("Myr")),
          potential=KeplerPotential( ... ),
          interpolant=None
        )

        """
        # The index only applies to the time component if the slice reaches
        # the last axis, which is the time axis. Otherwise, the slice applies
        # to all components.
        tindex = index if self.ndim == 1 else Ellipsis

        return replace(self, q=self.q[index], p=self.p[index], t=self.t[tindex])

    @gc.AbstractBasePhaseSpacePosition.__getitem__.dispatch
    def __getitem__(self: "Orbit", index: int) -> gc.PhaseSpacePosition:
        """Get the orbit at a specific time.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import galax.coordinates as gc
        >>> import galax.potential as gp
        >>> import galax.dynamics as gd

        >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")
        >>> w0 = gc.PhaseSpacePosition(
        ...     q=Quantity([8., 0., 0.], "kpc"),
        ...     p=Quantity([0., 230, 0.], "km/s"),
        ...     t=Quantity(0, "Myr"))
        >>> ts = Quantity([0., 1.], "Gyr")
        >>> orbit = gd.evaluate_orbit(pot, w0, ts)

        >>> orbit[0]
        PhaseSpacePosition(
          q=CartesianPosition3D( ... ),
          p=CartesianVelocity3D( ... ),
          t=Quantity[...](value=f64[], unit=Unit("Myr"))
        )
        >>> orbit[0].t
        Quantity['time'](Array(0., dtype=float64), unit='Myr')

        """
        return gc.PhaseSpacePosition(q=self.q[index], p=self.p[index], t=self.t[index])

    @gc.AbstractBasePhaseSpacePosition.__getitem__.dispatch
    def __getitem__(
        self: "Orbit", index: Int[Array, "..."] | Bool[Array, "..."] | ndarray
    ) -> "Orbit":
        """Get the orbit at specific indices."""
        match index.ndim:
            case 0:  # is this possible?
                msg = "Invalid index."
                raise IndexError(msg)
            case _:
                tindex = index

        return replace(self, q=self.q[index], p=self.p[index], t=self.t[tindex])

    # ==========================================================================
    # Dynamical quantities

    @partial(jax.jit, inline=True)
    def potential_energy(
        self, potential: gp.AbstractPotentialBase | None = None, /
    ) -> BatchFloatQScalar:
        r"""Return the specific potential energy.

        .. math::

            E_\Phi = \Phi(\boldsymbol{q})

        Parameters
        ----------
        potential : `galax.potential.AbstractPotentialBase` | None
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
        self, potential: "gp.AbstractPotentialBase | None" = None, /
    ) -> BatchFloatQScalar:
        r"""Return the specific total energy.

        .. math::

            E_K = \frac{1}{2} \\, |\boldsymbol{v}|^2
            E_\Phi = \Phi(\boldsymbol{q})
            E = E_K + E_\Phi

        Parameters
        ----------
        potential : `galax.potential.AbstractPotentialBase` | None
            The potential object to compute the energy from. If `None`
            (default), use the potential attribute of the orbit.

        Returns
        -------
        E : Array[float, (*batch,)]
            The kinetic energy.
        """
        return self.kinetic_energy() + self.potential_energy(potential)


ProxyOrbit.deliver(Orbit)
