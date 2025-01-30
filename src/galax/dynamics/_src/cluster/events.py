"""``galax`` dynamics."""

__all__ = ["MassBelowThreshold"]

from functools import partial
from typing import Any, TypeAlias

import equinox as eqx
from jaxtyping import Array
from plum import dispatch

import unxt as u
from unxt.quantity import AbstractQuantity

Args: TypeAlias = dict[str, Any]


class MassBelowThreshold(eqx.Module):  # type: ignore[misc]
    """Event to stop integration when the mass falls below a threshold.

    Instances can be used as the ``cond_fn`` argument of `diffrax.Event`. Since
    this returns a scalar (not a `bool`) the solve the solve will terminate on
    the step when the mass is below the threshold.

    With `diffrax.Event` this can be combined with a root-finder to find the
    exact time when the mass is below the threshold, rather than the step
    after.

    Example
    -------
    >>> import unxt as u
    >>> from galax.dynamics.cluster import MassBelowThreshold

    >>> cond_fn = MassBelowThreshold(u.Quantity(0.0, "Msun"))
    >>> args = {"units": u.unitsystems.galactic}

    >>> cond_fn(0.0, u.Quantity(1.0, "Msun"), args)
    Array(1., dtype=float64, weak_type=True)

    >>> cond_fn(0.0, u.Quantity(0.0, "Msun"), args)
    Array(0., dtype=float64, weak_type=True)

    TODO: example using it as a with `diffrax.Event`.

    """

    #: Threshold mass at which to stop integration.
    threshold: AbstractQuantity

    @partial(eqx.filter_jit)
    def __call__(
        self: "MassBelowThreshold", t: Any, y: Any, args: Args, **kw: Any
    ) -> Any:
        """Evaluate the event condition.

        Parameters
        ----------
        t : Any
            Current time.
        y : Any
            Current state.
        args : dict[str, Any]
            Additional arguments. Must contain:
            - "units": `unxt.AbstractUnitSystem`

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import diffrax as dfx
        >>> import unxt as u
        >>> import galax.dynamics as gd

        >>> event = dfx.Event(gd.cluster.MassBelowThreshold(u.Quantity(0.0, "Msun")))
        >>> mass_solver = gd.cluster.MassSolver(event=event)

        >>> mass_field = lambda t, Mc, args: -2e5 / (t + 1)
        >>> Mc0 = u.Quantity(1e6, "Msun")
        >>> t0, t1 = u.Quantity(0, "Gyr"), u.Quantity(1, "Gyr")
        >>> saveat = jnp.linspace(t0, t1, 10)
        >>> mass_soln = mass_solver.solve(mass_field, Mc0, t0, t1, saveat=saveat)
        >>> mass_soln.ys
        Array([1000000. , 56101.91157639, inf, inf, ...], dtype=float64)

        """
        return self.evaluate(t, y, args=args, **kw)

    @dispatch.abstract
    def evaluate(
        self: "MassBelowThreshold", t: Any, y: Any, /, *, args: Args, **kw: Any
    ) -> Any:
        """Evaluate the event condition."""
        raise NotImplementedError  # pragma: no cover


# ============================================================


@MassBelowThreshold.evaluate.dispatch
def evaluate(
    self: MassBelowThreshold, _: Any, y: Array, /, *, args: Args, **__: Any
) -> Array:
    return y - self.threshold.ustrip(args["units"])


@MassBelowThreshold.evaluate.dispatch
def evaluate(
    self: MassBelowThreshold, _: Any, y: AbstractQuantity, /, *, args: Args, **__: Any
) -> Array:
    return u.ustrip(args["units"], y - self.threshold)
