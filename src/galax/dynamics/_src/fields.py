"""Fields."""

__all__ = ["AbstractField"]

from collections.abc import Callable
from typing import Any

import diffrax as dfx
import equinox as eqx
from jaxtyping import PyTree
from plum import dispatch

import diffraxtra as dfxtra


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
    ODETerm(vector_field=<wrapped function __call__>)

    >>> field.terms(dfx.SemiImplicitEuler())
    (ODETerm(...), ODETerm(...))

    """

    __call__: eqx.AbstractClassVar[Callable[..., Any]]

    @dispatch.abstract
    def terms(self, solver: Any, /) -> PyTree[dfx.AbstractTerm]:
        """Return the `diffrax.AbstractTerm` `jaxtyping.PyTree` for integration."""
        raise NotImplementedError  # pragma: no cover


# ==================================================================


@AbstractField.terms.dispatch  # type: ignore[misc]
def terms(
    self: AbstractField, wrapper: dfxtra.AbstractDiffEqSolver, /
) -> PyTree[dfx.AbstractTerm]:
    """Return diffeq terms, redispatching to the solver.

    Examples
    --------
    >>> import diffrax as dfx
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> solver = gd.solve.DiffEqSolver(dfx.Dopri8())

    >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> field.terms(solver)
    ODETerm(vector_field=<wrapped function __call__>)

    """
    return self.terms(wrapper.solver)
