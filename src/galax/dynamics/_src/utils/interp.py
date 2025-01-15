"""vectorized wrapper around `diffrax.DenseInterpolation`.

This is private API.

"""

__all__ = ["VectorizedDenseInterpolation"]

from collections.abc import Callable
from functools import partial
from typing import TypeAlias, cast, final
from typing_extensions import override

import diffrax as dfx
import equinox as eqx
import jax
from jaxtyping import Array, ArrayLike, Float, Int, PyTree, Real, Shaped

import quaxed.numpy as jnp

RealTimes = Real[Array, "#times"]
IntScalarLike: TypeAlias = Int[ArrayLike, ""]
FloatScalarLike: TypeAlias = Float[ArrayLike, ""]
RealScalarLike: TypeAlias = FloatScalarLike | IntScalarLike
DenseInfos: TypeAlias = dict[str, PyTree[Shaped[Array, "times-1 ..."]]]
Shape: TypeAlias = tuple[int, ...]


@final
class VectorizedDenseInterpolation(dfx.AbstractPath):  # type: ignore[misc]
    """Vectorized wrapper around a `diffrax.DenseInterpolation`.

    This also works on non-batched interpolations.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import diffrax as dfx

    We'll start with a non-batched interpolation:

    >>> vector_field = lambda t, y, args: -y
    >>> term = dfx.ODETerm(vector_field)
    >>> solver = dfx.Dopri5()
    >>> ts = jnp.array([0.0, 1, 2, 3])
    >>> saveat = dfx.SaveAt(ts=ts, dense=True)
    >>> stepsize_controller = dfx.PIDController(rtol=1e-5, atol=1e-5)

    >>> sol = dfx.diffeqsolve(
    ...     term, solver, t0=0, t1=3, dt0=0.1, y0=1, saveat=saveat,
    ...     stepsize_controller=stepsize_controller)
    >>> interp = VectorizedDenseInterpolation(sol.interpolation)

    >>> interp.evaluate(ts[-1])  # scalar evaluation
    Array(0.04978961, dtype=float64)

    >>> interp.evaluate(ts)  # It works on arrays!
    Array([1. , 0.36788338, 0.13533922, 0.04978961], dtype=float64)

    >>> interp.evaluate(ts, ts[0])  # t1 - t0 mixed scalar and array
    Array([0. , 0.63211662, 0.86466078, 0.95021039], dtype=float64)

    As a convenience, we can also apply the `VectorizedDenseInterpolation` to
    the solution:

    >>> sol = VectorizedDenseInterpolation.apply_to_solution(sol)
    >>> isinstance(sol, dfx.Solution)
    True
    >>> isinstance(sol.interpolation, VectorizedDenseInterpolation)
    True

    Now we'll batch the interpolation:

    >>> @jax.vmap
    ... def solve(y0):
    ...     sol = dfx.diffeqsolve(
    ...         term, solver, t0=0, t1=3, dt0=0.1, y0=y0, saveat=saveat,
    ...         stepsize_controller=stepsize_controller)
    ...     return sol
    >>> sol = solve(jnp.array([1, 2, 3]))
    >>> interp = VectorizedDenseInterpolation(sol.interpolation)

    >>> interp.evaluate(ts[-1]).round(3)  # scalar eval of batched interp
    Array([0.05 , 0.1  , 0.149], dtype=float64)

    >>> interp.evaluate(ts).round(3)  # array eval of batched interp
    Array([[1.   , 0.368, 0.135, 0.05 ],
           [2.   , 0.736, 0.271, 0.1  ],
           [3.   , 1.104, 0.406, 0.149]], dtype=float64)

    >>> interp.evaluate(ts, ts[0]).round(3)  # mixed scalar and array eval
    Array([[0.   , 0.632, 0.865, 0.95 ],
           [0.   , 1.264, 1.729, 1.9  ],
           [0.   , 1.896, 2.594, 2.851]], dtype=float64)

    Let's inspect the API.

    >>> interp.scalar_interpolation
    DenseInterpolation(
      ts=f64[3,4097],
      ts_size=weak_i64[3],
      infos={'k': f64[3,4096,7], 'y0': f64[3,4096], 'y1': f64[3,4096]},
      interpolation_cls=<class 'diffrax._solver.dopri5._Dopri5Interpolation'>,
      direction=weak_i64[3],
      t0_if_trivial=f64[3],
      y0_if_trivial=f64[3]
    )

    >>> interp.batch_shape
    (3,)

    >>> interp.t0
    Array([0., 0., 0.], dtype=float64)

    >>> interp.t1
    Array([3., 3., 3.], dtype=float64)

    >>> interp.ts.shape
    (3, 4097)

    >>> interp.ts_size
    Array([8, 9, 9], dtype=int64, weak_type=True)

    >>> jax.tree.map(lambda x: x.shape, interp.infos)
    {'k': (3, 4096, 7), 'y0': (3, 4096), 'y1': (3, 4096)}

    >>> interp.interpolation_cls
    <class 'diffrax..._Dopri5Interpolation'>

    >>> interp.direction
    Array([1, 1, 1], dtype=int64, weak_type=True)

    >>> interp.t0_if_trivial
    Array([0., 0., 0.], dtype=float64)

    >>> interp.y0_if_trivial
    Array([1., 2., 3.], dtype=float64)

    """

    #: Dense interpolation with flattened batch dimensions.
    scalar_interpolation: dfx.DenseInterpolation

    #: The batch shape of the interpolation without vectorization over the
    #: solver that produced this interpolation. E.g.
    batch_shape: Shape

    #: The shape of the solution.
    y0_shape: PyTree[Shape, "Y"]

    def __init__(self, interp: dfx.DenseInterpolation, /) -> None:
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

    @property
    def batch_ndim(self) -> int:
        """The number of batch dimensions."""
        return len(self.batch_shape)

    @override
    @eqx.filter_jit  # type: ignore[misc]
    def evaluate(
        self,
        t0: Real[Array, "time"],
        t1: Real[Array, "time"] | None = None,
        left: bool = True,
    ) -> PyTree[Shaped[Array, "?*shape"], "Y"]:
        """Evaluate the interpolation at any point in the region of integration.

        Args:
            t0: The point to evaluate the solution at.
            t1: If passed, then the increment from `t0` to `t1` is returned.
                (``=evaluate(t1) - evaluate(t0)``)
            left: When evaluating at a jump in the solution, whether to return
                the left-limit or the right-limit at that point.

        Return:
            The solution at the given time.
            Shape (*batch, [len(t0)], *y0.shape)

        """
        # If t1, then return the difference
        if t1 is not None:
            t0, t1 = jnp.broadcast_arrays(t0, t1)
            return self.evaluate(t1, left=left) - self.evaluate(t0, left=left)

        # Evaluate the scalar interpolation
        # TODO: enable t0, t1 to be N-D arrays
        t0ndim = jnp.ndim(t0)  # store shape for unpacking
        t0 = jnp.atleast_1d(t0).astype(float)
        ys = jax.vmap(  # vmap over the batch dimension of the interpolator
            lambda interp: jax.vmap(partial(interp.evaluate, left=left))(t0)
        )(self.scalar_interpolation)

        # Reshape to remove the time batch dimension if the time was scalar.
        # Since the interpolation is flattened, this is always the 1st index.
        if t0ndim == 0:
            ys = jax.tree.map(lambda x: x[:, 0], ys)

        # Reshape the 0th dimension back to the original batch shape.
        ys = jax.tree.map(lambda x: x.reshape(*self.batch_shape, *x.shape[1:]), ys)

        return ys  # noqa: RET504

    # =======================
    # DenseInterpolation API
    # modified to have batch dimensions.

    @property
    def t0(self) -> Real[Array, "{self.batch_shape}"]:
        """The start time of the interpolation."""
        flatt0 = jax.vmap(lambda x: x.t0)(self.scalar_interpolation)
        return flatt0.reshape(*self.batch_shape)

    @property
    def t1(self) -> RealScalarLike:
        """The end time of the interpolation."""
        flatt1 = jax.vmap(lambda x: x.t1)(self.scalar_interpolation)
        return flatt1.reshape(*self.batch_shape)

    @property
    def ts(self) -> Real[Array, "times"]:
        """The times of the interpolation."""
        return self.scalar_interpolation.ts

    @property
    def ts_size(self) -> IntScalarLike:
        """The number of times in the interpolation."""
        return self.scalar_interpolation.ts_size

    @property
    def infos(self) -> DenseInfos:
        """The infos of the interpolation."""
        return cast(DenseInfos, self.scalar_interpolation.infos)

    @property
    def interpolation_cls(self) -> Callable[..., dfx.AbstractLocalInterpolation]:
        """The interpolation class of the interpolation."""
        return cast(
            Callable[..., dfx.AbstractLocalInterpolation],
            self.scalar_interpolation.interpolation_cls,
        )

    @property
    def direction(self) -> IntScalarLike:
        """Direction vector."""
        return self.scalar_interpolation.direction

    @property
    def t0_if_trivial(self) -> RealScalarLike:
        """The start time of the interpolation if scalar input."""
        return self.scalar_interpolation.t0_if_trivial

    @property
    def y0_if_trivial(self) -> PyTree[RealScalarLike, "Y"]:
        """The start value of the interpolation if scalar input."""
        return self.scalar_interpolation.y0_if_trivial

    # =======================
    # Convenience methods

    @classmethod
    def apply_to_solution(cls, soln: dfx.Solution, /) -> dfx.Solution:
        """Make a `diffrax.Solution` interpolation vectorized.

        This does an out-of-place transformation, wrapping the interpolation
        in a `VectorizedDenseInterpolation`.

        """
        if soln.interpolation is None:
            return soln

        return eqx.tree_at(
            lambda tree: tree.interpolation, soln, cls(soln.interpolation)
        )
