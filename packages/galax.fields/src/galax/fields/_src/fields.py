"""Fields."""

__all__ = ["AbstractField"]

import functools as ft
from dataclasses import KW_ONLY

from collections.abc import Callable
from jaxtyping import ArrayLike, PyTree
from typing import Any

import diffrax as dfx
import equinox as eqx
from plum import dispatch

import diffraxtra as dfxtra
import quaxed.numpy as jnp
import unxt as u


class AbstractField(eqx.Module):
    """Abstract base class for fields.

    Methods
    -------
    - `__call__` : evaluates the field.
    - `terms` : returns the `diffrax.AbstractTerm` wrapper of the `jax.jit`-ed
      ``__call__`` for integration with `diffrax.diffeqsolve`. `terms` takes as
      input the `diffrax.AbstractSolver` object (or something that wraps it,
      like a `diffraxtra.DiffEqSolver`), to determine the correct
      `diffrax.AbstractTerm` and its `jaxtyping.PyTree` structure.

    Examples
    --------
    >>> import diffrax as dfx
    >>> import quaxed.numpy as jnp
    >>> import galax.fields as gf

    Define a field from a function, here exponential decay (``dy/dt = -k y``)
    with ``k = 0.003`` per Myr:

    >>> field = gf.CustomField(lambda t, y, args: -0.003 * y)

    Evaluate the field at a given time and state:

    >>> field(0.0, jnp.array([1.0]))
    Array([-0.003], dtype=float64)

    The state can be any PyTree, for example a harmonic oscillator with state
    ``(x, v)``:

    >>> osc = gf.CustomField(lambda t, y, args: (y[1], -0.001 * y[0]))
    >>> osc(0.0, (jnp.array(1.0), jnp.array(0.0)))
    (Array(0., dtype=float64, weak_type=True),
     Array(-0.001, dtype=float64, weak_type=True))

    For integration with `diffrax.diffeqsolve` the ``terms`` method returns the
    correctly-structured `diffrax.AbstractTerm` `jaxtyping.PyTree`. The term is
    a wrapper around the ``__call__`` method, which is `jax.jit`-ed for
    performance.

    >>> field.terms(dfx.Dopri8())
    ODETerm(...)

    For fields on orbits see `galax.dynamics.HamiltonianField`.

    """

    __call__: eqx.AbstractClassVar[Callable[..., Any]]

    _: KW_ONLY
    #: unit system of the field.
    units: eqx.AbstractVar[u.AbstractUnitSystem]

    @dispatch.abstract
    def terms(self, solver: Any, /) -> PyTree[dfx.AbstractTerm]:
        """Return the `diffrax.AbstractTerm` `jaxtyping.PyTree` for integration."""
        raise NotImplementedError  # pragma: no cover

    # TODO: consider the frame information, like in `galax.dynamics`'s `parse_to_t_y`
    @dispatch.abstract
    def parse_inputs(self, *args: Any, **kwargs: Any) -> Any:
        """Parse inputs for the field.

        Dispatches to this method should at least support the following:

        - `parse_inputs(self, t: Array, y0: PyTree[Array], /, *, ustrip: bool) -> tuple[Array, PyTree[Array]]`
        - `parse_inputs(self, t: Quantity, y0: PyTree[Quantity], /, *, ustrip: bool) -> tuple[Array, PyTree[Array]]`

        Where the output types are suitable for use with `diffrax`.

        """  # noqa: E501
        raise NotImplementedError


# ==================================================================
# Terms


@AbstractField.terms.dispatch
def terms(self: AbstractField, _: dfx.AbstractSolver, /) -> dfx.ODETerm:
    """Return diffeq terms, redispatching to the solver.

    This is the default implementation, which wraps the field's ``__call__``
    method in a `equinox.filter_jit`-ed function and returns an
    `diffrax.ODETerm`.

    Examples
    --------
    >>> import diffrax as dfx
    >>> import galax.fields as gf

    >>> field = gf.CustomField(lambda t, y, args: -0.003 * y)

    >>> field.terms(dfx.Dopri8())
    ODETerm(...)

    """
    return dfx.ODETerm(eqx.filter_jit(self.__call__))


@AbstractField.terms.dispatch
def terms(
    self: AbstractField, wrapper: dfxtra.AbstractDiffEqSolver, /
) -> PyTree[dfx.AbstractTerm]:
    """Return diffeq terms, redispatching to the solver.

    Examples
    --------
    >>> import diffrax as dfx
    >>> import diffraxtra as dfxtra
    >>> import galax.fields as gf

    >>> solver = dfxtra.DiffEqSolver(dfx.Dopri8())

    >>> field = gf.CustomField(lambda t, y, args: -0.003 * y)

    >>> field.terms(solver)
    ODETerm(...)

    """
    return self.terms(wrapper.solver)


# ==================================================================
# Compatibility with `diffraxtra.AbstractDiffEqSolver`


@dfxtra.AbstractDiffEqSolver.__call__.dispatch(precedence=1)  # type: ignore[misc]
@ft.partial(eqx.filter_jit)
def call(
    self: dfxtra.AbstractDiffEqSolver,
    field: AbstractField,
    /,
    t0: Any,
    t1: Any,
    dt0: Any | None,
    y0: PyTree[ArrayLike],
    args: PyTree[Any] = None,
    **kwargs: Any,
) -> dfx.Solution:
    """`diffraxtra.AbstractDiffEqSolver` supports `galax.fields.AbstractField`.

    Re-dispatches to `diffraxtra.AbstractDiffEqSolver` after determining the
    terms structure.

    A given ``dt0`` sets the size of the first step, and ``t0 -> t1`` sets its
    direction. `diffrax.diffeqsolve` needs the sign of ``dt0`` to match
    ``t1 - t0``, so without this, one fixed-step solver could not integrate both
    backward and forward, as `galax.dynamics.evaluate_orbit` does when the
    orbit's start time lies inside the requested times.

    A `unxt.Quantity` ``dt0`` is converted to the field's time unit; a plain
    number is taken to be in that unit already.

    """
    terms = field.terms(self.solver)
    if dt0 is not None:
        if isinstance(dt0, u.AbstractQuantity):
            dt0 = u.ustrip(field.units["time"], dt0)
        dt0 = jnp.where(t1 < t0, -1, 1) * jnp.abs(dt0)
    return self(terms, t0, t1, dt0, y0, args, **kwargs)
