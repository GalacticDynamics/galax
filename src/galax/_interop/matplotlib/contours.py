__all__ = ["plot_potential_contours"]


from typing import Any

import matplotlib.pyplot as plt
from astropy.utils import isiterable
from matplotlib.axes import Axes
from matplotlib.cm import Blues
from matplotlib.figure import Figure
from plum import dispatch

import quaxed.numpy as qnp
from unxt import Quantity

from galax.potential._potential.base import AbstractPotentialBase
from galax.potential._potential.plot import MatplotlibBackend

# ============================================================================
# Plot potential contours


@dispatch  # type: ignore[misc]
def plot_potential_contours(
    pot: AbstractPotentialBase,
    _: type[MatplotlibBackend] = MatplotlibBackend,
    /,
    *,
    grid: tuple[Quantity | int, ...],
    t: Quantity["time"] = Quantity(0.0, "Myr"),  # noqa: B008
    filled: bool = True,
    ax: Any | None = None,
    labels: tuple[str, ...] | None = None,
    subplots_kw: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Figure:
    """Plot contours of the potential.

    Parameters
    ----------
    pot : :class:`~galax.potential.base.AbstractPotentialBase`
        Potential object to plot.
    backend : type[:class:`~galax.potential.plot.MatplotlibBackend`]
        The Matplotlib plotting backend.

    grid : tuple[Any, ...]
        Coordinate grids or slice value for each dimension. Should be a
        tuple of 1D arrays or numbers.
    t : Quantity["time"], optional
        The time to evaluate at.

    filled : bool, optional keyword-only
        Use :func:`~matplotlib.pyplot.contourf` instead of
        :func:`~matplotlib.pyplot.contour`. Default is ``True``.
    ax : `~matplotlib.Axes`, optional keyword-only
        Axes object to plot on. If not specified, a new figure and axes will
        be created.
    labels : tuple[str, ...] or None, optional keyword-only
        List of axis labels.
    subplots_kw : dict[str, Any], optional keyword-only
        kwargs passed to matplotlib's subplots() function if an axes object
        is not specified.
    **kwargs : Any, optional keyword-only
        kwargs passed to either :func:`~matplotlib.pyplot.contourf` or
        :func:`~matplotlib.pyplot.plot`.

    Returns
    -------
    :class:`~matplotlib.Figure`
    """
    # Process figure and axes
    if ax is None:  # make new figure
        fig, ax = plt.subplots(1, 1, **(subplots_kw or {}))
    else:  # use existing figure
        fig = ax.figure

    # Figure out which elements are iterable, which are numeric
    _grids: list[tuple[int, Any]] = []
    _slices: list[tuple[int, Any]] = []
    for i, g in enumerate(grid):
        if isiterable(g):
            _grids.append((i, g))
        else:
            _slices.append((i, g))

    # Prepare kwargs for plotting sub-functions
    kwargs = {
        "ax": ax,
        "grids": _grids,
        "slices": _slices,
        "labels": labels,
        "kwargs": kwargs,
    }

    match ndim := len(_grids):
        case 1:
            _plot_potential_countours_1d(pot, t, **kwargs)
        case 2:
            _plot_potential_countours_2d(pot, t, filled=filled, **kwargs)
        # TODO: implement 3D contours
        case _:
            msg = (
                f"ndim = {ndim}: you can only make contours on a 1D or 2D grid."
                " For other dimensions, you have to specify values to slice."
            )
            raise ValueError(msg)

    return fig


def _plot_potential_countours_1d(
    pot: AbstractPotentialBase,
    t: Quantity["time"],
    *,
    ax: Axes,
    grids: list[Any],
    slices: list[Any],
    labels: tuple[str, ...] | None,
    kwargs: dict[str, Any],
) -> None:
    x1 = grids[0][1].to_value(pot.units["length"])

    # Create q array
    q = qnp.zeros((len(x1), len(grids) + len(slices)))
    q = q.at[:, grids[0][0]].set(x1)
    for ii, slc in slices:
        q = q.at[:, ii].set(slc)
    q = Quantity(q, pot.units["length"])

    # Evaluate potential
    Z = pot.potential(q, t)

    # Plot potential
    ax.plot(  # TODO: solve matplotlib-Quantity issue
        x1,
        Z.to_units_value(pot.units["specific energy"]),
        **kwargs,
    )

    if labels is not None:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(f"potential [{pot.units['specific energy']}]")


def _plot_potential_countours_2d(
    pot: AbstractPotentialBase,
    t: Quantity["time"],
    *,
    ax: Axes,
    grids: list[Any],
    slices: list[Any],
    labels: tuple[str, str] | None,
    filled: bool,
    kwargs: dict[str, Any],
) -> None:
    # Create meshgrid
    # TODO: don't take to_value when Quantity.at is implemented
    x1, x2 = qnp.meshgrid(
        grids[0][1].to_value(pot.units["length"]),
        grids[1][1].to_value(pot.units["length"]),
    )
    shape = x1.shape

    # Create q array
    # TODO: use Quantity.at when it's implemented
    q = qnp.zeros((x1.size, len(grids) + len(slices)))
    q = q.at[:, grids[0][0]].set(qnp.ravel(x1))
    q = q.at[:, grids[1][0]].set(qnp.ravel(x2))
    for ii, slc in slices:
        q = q.at[:, ii].set(slc)
    q = Quantity(q, pot.units["length"])

    # Evaluate potential
    Z = pot.potential(q, t)

    # Plot contours
    kwargs.setdefault("cmap", Blues)  # better default colormap
    plot_func = ax.contourf if filled else ax.contour
    plot_func(  # TODO: solve matplotlib-Quantity issue
        x1,
        x2,
        Z.reshape(shape).to_value(pot.units["specific energy"]),
        **kwargs,
    )

    if labels is not None:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
