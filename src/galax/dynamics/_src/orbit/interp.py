"""galax: Galactic Dynamix in Jax."""

__all__ = ["PhaseSpaceInterpolation"]

from collections.abc import Callable
from functools import partial
from typing import Any, cast

import diffrax as dfx
import equinox as eqx
import jax
from jaxtyping import Array, Bool, Int, PyTree, Real

import diffraxtra as dfxtra
import quaxed.numpy as jnp
import unxt as u
from diffraxtra.interp import (
    BatchedIntScalar,
    BatchedRealScalar,
    BatchedRealTimes,
    RealScalarLike,
    VecDenseInfos,
)
from unxt.quantity import BareQuantity as FastQ

import galax.coordinates as gc
import galax.typing as gt


@partial(jax.jit)
def within_bounds(
    t: Real[Array, "N"], t_lower: Real[Array, ""], t_upper: Real[Array, ""]
) -> Bool[Array, "N"]:
    return jnp.logical_and(jnp.greater_equal(t, t_lower), jnp.less_equal(t, t_upper))


# TODO: move this to galax.coordinates?
# TODO: address mypy complaints about subclassing
# AbstractVectorizedDenseInterpolation
class PhaseSpaceInterpolation(eqx.Module):  # type: ignore[misc]
    """Evaluate phase-space interpolations."""

    #: The vectorized interpolation object.
    interp: dfxtra.VectorizedDenseInterpolation = eqx.field(
        converter=dfxtra.VectorizedDenseInterpolation.from_
    )

    #: The unit system for the interpolation.
    units: u.AbstractUnitSystem = eqx.field(static=True, converter=u.unitsystem)

    @eqx.filter_jit  # type: ignore[misc]
    def evaluate(self, ts: Any) -> gc.PhaseSpaceCoordinate:
        usys = self.units
        t = FastQ.from_(ts, usys["time"])

        # TODO: is there a way to push this into the interpolant?
        tval = u.ustrip(usys["time"], t)
        tval = eqx.error_if(
            tval,
            jnp.logical_not(jnp.all(within_bounds(tval, self.t0, self.t1))),
            "Time out of bounds.",
        )

        # Evaluate the interpolation
        ys = self.interp.evaluate(tval)
        # Reshape (T, *batch) to (*batch, T)
        if jnp.ndim(ts) != 0:
            ys = jax.tree.map(lambda x: jnp.moveaxis(x, 0, -2), ys)

        q = FastQ(ys[0], usys["length"])
        p = FastQ(ys[1], usys["speed"])

        # Return as a phase-space position
        return gc.PhaseSpaceCoordinate(q=q, p=p, t=t)

    # =====================================================
    # diffraxtra.AbstractVectorizedDenseInterpolation API

    @property
    def scalar_interpolation(self) -> dfx.DenseInterpolation:
        """Return the scalar interpolation for the phase-space position."""
        return cast(dfx.DenseInterpolation, self.interp.scalar_interpolation)

    @property
    def batch_shape(self) -> gt.Shape:
        """Return the batch shape of the interpolation."""
        return cast(gt.Shape, self.interp.batch_shape)

    @property
    def y0_shape(self) -> gt.Shape:
        """Return the shape of the initial value."""
        return cast(gt.Shape, self.interp.y0_shape)

    @property
    def batch_ndim(self) -> int:
        """Return the number of batch dimensions."""
        return cast(int, self.interp.batch_ndim)

    def __call__(self, *args: Any, **kwds: Any) -> gc.PhaseSpaceCoordinate:
        return cast(gc.PhaseSpaceCoordinate, self.evaluate(*args, **kwds))

    @property
    def t0(self) -> BatchedRealScalar:
        """The start time of the interpolation."""
        return self.interp.t0

    @property
    def t1(self) -> BatchedRealScalar:
        """The end time of the interpolation."""
        return self.interp.t1

    @property
    def ts(self) -> BatchedRealTimes:
        """The times of the interpolation."""
        return self.interp.ts

    @property
    def ts_size(self) -> Int[Array, "..."]:  # TODO: shape
        """The number of times in the interpolation."""
        return self.interp.ts_size

    @property
    def infos(self) -> VecDenseInfos:
        """The infos of the interpolation."""
        return self.interp.infos

    @property
    def interpolation_cls(self) -> Callable[..., dfx.AbstractLocalInterpolation]:
        """The interpolation class of the interpolation."""
        return cast(
            "Callable[..., dfx.AbstractLocalInterpolation]",
            self.interp.interpolation_cls,
        )

    @property
    def direction(self) -> BatchedIntScalar:
        """Direction vector."""
        return self.interp.direction

    @property
    def t0_if_trivial(self) -> BatchedRealScalar:
        """The start time of the interpolation if scalar input."""
        return self.interp.t0_if_trivial

    @property  # TODO: get the shape correct
    def y0_if_trivial(self) -> PyTree[RealScalarLike, "Y"]:  # type: ignore[name-defined]
        """The start value of the interpolation if scalar input."""
        return self.interp.y0_if_trivial
