__all__ = ["Interpolant"]

from typing import final
from typing_extensions import override

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

import coordinax as cx
import quaxed.numpy as xp
import unxt as u
from unxt.quantity import UncheckedQuantity as FastQ

import galax.coordinates as gc
import galax.typing as gt
from galax.dynamics._src.diffeq.interp import AbstractVectorizedDenseInterpolation


@final
class Interpolant(AbstractVectorizedDenseInterpolation):
    """Wrapper for `diffrax.DenseInterpolation`.

    This satisfies the `galax.coordinates.PhaseSpacePositionInterpolant`
    Protocol.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> from unxt.unitsystems import galactic
    >>> import galax.coordinates as gc
    >>> import galax.dynamics as gd
    >>> import galax.potential as gp

    We define initial conditions and a potential:

    >>> w0 = gc.PhaseSpacePosition(q=u.Quantity([10, 0, 0], "kpc"),
    ...                            p=u.Quantity([0, 200, 0], "km/s"))
    >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
    ...                             r_s=u.Quantity(5, "kpc"), units="galactic")

    We can integrate the phase-space position for 1 Gyr, getting the final
    position.  The integrator accepts any function for the equations of motion.
    Here we will reproduce what happens with orbit integrations.

    >>> integrator = gd.integrate.Integrator()
    >>> t0, t1 = u.Quantity(0, "Gyr"), u.Quantity(1, "Gyr")
    >>> w = integrator(gd.fields.HamiltonianField(pot), w0, t0, t1, units=galactic,
    ...                interpolated=True)
    >>> type(w)
    <class 'galax.coordinates...InterpolatedPhaseSpacePosition'>

    >>> isinstance(w.interpolant, gc.PhaseSpacePositionInterpolant)
    True

    """

    #: Dense interpolation with flattened batch dimensions.
    scalar_interpolation: dfx.DenseInterpolation

    #: The batch shape of the interpolation without vectorization over the
    #: solver that produced this interpolation. E.g.
    batch_shape: gt.Shape

    #: The shape of the solution.
    y0_shape: PyTree[gt.Shape, "Y"]

    #: The unit system of the solution. This is used to convert the time input
    #: to the interpolant and the phase-space position output.
    units: u.AbstractUnitSystem = eqx.field(static=True, converter=u.unitsystem)

    def __init__(
        self, interp: dfx.DenseInterpolation, /, *, units: u.AbstractUnitSystem
    ) -> None:
        # Set the units
        self.units = self.__dataclass_fields__["units"].metadata["converter"](units)

        # # Store the batch shape
        bshape = interp.t0_if_trivial.shape
        bshape = eqx.error_if(
            bshape,
            bshape != interp.t0_if_trivial.shape,
            "batch_shape must match the shape of the ts_size of the interpolation",
        )
        self.batch_shape = bshape
        self.y0_shape = jax.tree.map(
            lambda x: x.shape[self.batch_ndim :], interp.y0_if_trivial
        )

        # Flatten the batch shape of the interpolation
        self.scalar_interpolation = jax.tree.map(
            lambda x: x.reshape(-1, *x.shape[self.batch_ndim :]),
            interp,
            is_leaf=eqx.is_array,
        )

    @override
    def evaluate(
        self,
        t0: u.Quantity["time"],
        t1: u.Quantity["time"] | None = None,
        left: bool = False,
    ) -> gc.PhaseSpacePosition:
        """Evaluate the interpolation."""
        ys = super().evaluate(
            t0.ustrip(self.units["time"]),
            t1 if t1 is None else t1.ustrip(self.units["time"]),
            left=left,
        )

        # Reshape (T, *batch) to (*batch, T)
        if t0.ndim != 0:
            ys = jax.tree.map(lambda x: xp.moveaxis(x, 0, -2), ys)

        # Construct and return the result
        return gc.PhaseSpacePosition(
            q=FastQ(ys[0], self.units["length"]),
            p=FastQ(ys[1], self.units["speed"]),
            t=t0,
        )


# TODO: support interpolation
@gc.AbstractOnePhaseSpacePosition.from_.dispatch  # type: ignore[misc,attr-defined]
def from_(
    cls: type[gc.InterpolatedPhaseSpacePosition],
    soln: dfx.Solution,
    *,
    frame: cx.frames.AbstractReferenceFrame,  # not dispatched on, but required
    units: u.AbstractUnitSystem,  # not dispatched on, but required
    interpolant: Interpolant,  # not dispatched on, but required
    unbatch_time: bool = False,
) -> gc.AbstractOnePhaseSpacePosition:
    """Convert a solution to a phase-space position."""
    # Reshape (T, *batch) to (*batch, T)
    t = soln.ts  # already in the correct shape
    q = jnp.moveaxis(soln.ys[0], 0, -2)
    p = jnp.moveaxis(soln.ys[1], 0, -2)

    # Reshape (*batch,T=1,6) to (*batch,6) if t is a scalar
    if unbatch_time and t.shape[-1] == 1:
        t = t[..., -1]
        q = q[..., -1, :]
        p = p[..., -1, :]

    # Convert the solution to a phase-space position
    return cls(
        q=cx.CartesianPos3D.from_(q, units["length"]),
        p=cx.CartesianVel3D.from_(p, units["speed"]),
        t=FastQ(soln.ts, units["time"]),
        frame=frame,
        interpolant=interpolant,
    )
