"""Orbit objects."""

__all__ = ["Orbit"]

from dataclasses import KW_ONLY, replace
from functools import partial
from typing import TYPE_CHECKING, Any, overload

import equinox as eqx
import jax
import jax.numpy as jnp

import coordinax as cx
from unxt import Quantity
from xmmutablemap import ImmutableMap

import galax.coordinates as gc
import galax.typing as gt
from galax.coordinates._psp.interp import PhaseSpacePositionInterpolant
from galax.coordinates._psp.utils import HasShape, getitem_vec1time_index
from galax.potential import AbstractPotentialBase
from galax.typing import BatchFloatQScalar, QVec1, QVecTime
from galax.utils._shape import batched_shape, vector_batched_shape

if TYPE_CHECKING:
    from typing import Self


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
      interpolant=DiffraxInterpolant( ... )
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

    potential: AbstractPotentialBase
    """Potential in which the orbit was integrated."""

    interpolant: PhaseSpacePositionInterpolant | None = None
    """The interpolation function."""

    meta: ImmutableMap[Any] = eqx.field(
        default_factory=dict,
        converter=ImmutableMap,
        static=True,
        repr=False,
        compare=False,
    )
    """Metadata about the orbit."""

    def __post_init__(self) -> None:
        """Post-initialization."""
        # Need to ensure t shape is correct. Can be Vec0.
        if self.t.ndim == 0:
            object.__setattr__(self, "t", self.t[None])

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

    @overload
    def __getitem__(self, index: int) -> gc.PhaseSpacePosition: ...

    @overload
    def __getitem__(self, index: slice | HasShape | tuple[Any, ...]) -> "Self": ...

    def __getitem__(self, index: Any) -> "Self | gc.PhaseSpacePosition":
        """Return a new object with the given slice applied."""
        # TODO: return an OrbitSnapshot (or similar) instead of PhaseSpacePosition?
        if isinstance(index, int):
            return gc.PhaseSpacePosition(
                q=self.q[index], p=self.p[index], t=self.t[index]
            )

        if isinstance(index, HasShape):
            msg = "Shaped indexing not yet implemented."
            raise NotImplementedError(msg)

        # Compute subindex
        subindex = getitem_vec1time_index(index, self.t)
        # Apply slice
        return replace(self, q=self.q[index], p=self.p[index], t=self.t[subindex])

    # ==========================================================================
    # Dynamical quantities

    @partial(jax.jit, inline=True)
    def potential_energy(
        self, potential: AbstractPotentialBase | None = None, /
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
        self, potential: "AbstractPotentialBase | None" = None, /
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
