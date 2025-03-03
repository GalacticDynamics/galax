"""Fields."""

__all__ = ["AbstractField"]

from collections.abc import Callable
from dataclasses import KW_ONLY
from functools import partial
from typing import Any

import diffrax as dfx
import equinox as eqx
from jaxtyping import ArrayLike, PyTree
from plum import dispatch

import diffraxtra as dfxtra
import unxt as u


class AbstractField(eqx.Module, strict=True):  # type: ignore[misc,call-arg]
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
    >>> import unxt as u
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    Define a Hamiltonian field:

    >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    Evaluate the field at a given coordinate:

    >>> t = u.Quantity(0, "Gyr")
    >>> x = u.Quantity([8., 0, 0], "kpc")
    >>> v = u.Quantity([0, 220, 0], "km/s")

    >>> field(t, x, v)
    (Array([0.        , 0.22499668, 0.        ], dtype=float64),
     Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    This can also be done with `jax.Array` directly, but care must be taken to
    ensure the units are correct. In this case ``x`` is in the right units, but
    ``t``, ``v`` are not. We use `unxt.ustrip` to correctly convert and remove
    the units:

    >>> field(t.ustrip("Myr"), x.value, v.ustrip("kpc/Myr"))
    (Array([0.        , 0.22499668, 0.        ], dtype=float64),
     Array([-0.0702891, -0.       , -0.       ], dtype=float64))

    Field evaluation is very flexible and can work with a large variety of
    inputs. For more information, see the
    `galax.dynamics.fields.HamiltonianField` class.

    For integration with `diffrax.diffeqsolve` the ``terms`` method returns the
    correctly-structured `diffrax.AbstractTerm` `jaxtyping.PyTree`. The term is
    a wrapper around the ``__call__`` method, which is `jax.jit`-ed for
    performance.

    >>> field.terms(dfx.Dopri8())
    ODETerm(...)

    >>> field.terms(dfx.SemiImplicitEuler())
    (ODETerm(...), ODETerm(...))

    """

    __call__: eqx.AbstractClassVar[Callable[..., Any]]

    _: KW_ONLY
    #: unit system of the field.
    units: eqx.AbstractVar[u.AbstractUnitSystem]

    @dispatch.abstract
    def terms(self, solver: Any, /) -> PyTree[dfx.AbstractTerm]:
        """Return the `diffrax.AbstractTerm` `jaxtyping.PyTree` for integration."""
        raise NotImplementedError  # pragma: no cover

    # TODO: consider the frame information, like in `parse_to_t_y`
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
def terms(self: AbstractField, _: dfx.AbstractSolver, /) -> dfx.AbstractTerm:
    """Return diffeq terms, redispatching to the solver.

    This is the default implementation, which wraps the field's ``__call__``
    method in a `equinox.filter_jit`-ed function and returns an
    `diffrax.ODETerm`.

    Examples
    --------
    >>> import diffrax as dfx
    >>> import unxt as u
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> field.terms(dfx.Dopri8())
    ODETerm(...)

    >>> field = gd.cluster.ZeroMassRate(units="galactic")
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
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> solver = gd.DiffEqSolver(dfx.Dopri8())

    >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> field.terms(solver)
    ODETerm(...)

    """
    return self.terms(wrapper.solver)


# ==================================================================
# Compatibility with `diffraxtra.AbstractDiffEqSolver`


@dfxtra.AbstractDiffEqSolver.__call__.dispatch(precedence=1)  # type: ignore[misc]
@partial(eqx.filter_jit)
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
    """`diffraxtra.AbstractDiffEqSolver` supports `galax.dynamics.AbstractField`.

    Re-dispatches to `diffraxtra.AbstractDiffEqSolver` after determining the
    terms structure.

    """
    terms = field.terms(self.solver)
    return self(terms, t0, t1, dt0, y0, args, **kwargs)
