__all__ = ["plot_contours"]


from typing import Any

import numpy as np
from astropy.utils import isiterable

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import Blues
from matplotlib.figure import Figure

from galax.potential._potential.base import AbstractPotentialBase

# ============================================================================
# Plot contours


def _plot_countours_1d(
    pot: AbstractPotentialBase,
    t: float,
    *,
    ax: Axes,
    grids: list[Any],
    slices: list[Any],
    labels: tuple[str, ...] | None,
    kwargs: dict[str, Any],
) -> None:
    x1 = grids[0][1]
    r = np.zeros((len(grids) + len(slices), len(x1)))
    r[grids[0][0]] = x1

    for ii, slc in slices:
        r[ii] = slc

    Z = pot.potential_energy(r * pot.units["length"], t=t)
    ax.plot(x1, Z, **kwargs)

    if labels is not None:
        ax.set_xlabel(labels[0])
        ax.set_ylabel("potential")


def _plot_countours_2d(
    pot: AbstractPotentialBase,
    t: float,
    *,
    ax: Axes,
    grids: list[Any],
    slices: list[Any],
    labels: tuple[str, ...] | None,
    filled: bool,
    kwargs: dict[str, Any],
) -> None:
    x1, x2 = np.meshgrid(grids[0][1], grids[1][1])
    shp = x1.shape
    x1, x2 = x1.ravel(), x2.ravel()

    r = np.zeros((len(grids) + len(slices), len(x1)))
    r[grids[0][0]] = x1
    r[grids[1][0]] = x2

    for ii, slc in slices:
        r[ii] = slc

    Z = pot.potential_energy(r * pot.units["length"], t=t)

    # make default colormap not suck
    cmap = kwargs.setdefault("cmap", Blues)
    if filled:
        ax.contourf(x1.reshape(shp), x2.reshape(shp), Z.reshape(shp), **kwargs)
    else:
        ax.contour(x1.reshape(shp), x2.reshape(shp), Z.reshape(shp), **kwargs)

    if labels is not None:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])


def plot_contours(
    pot: AbstractPotentialBase,
    /,
    grid: tuple[Any, ...],
    t: float = 0.0,
    *,
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
    grid : tuple[Any, ...]
        Coordinate grids or slice value for each dimension. Should be a
        tuple of 1D arrays or numbers.
    t : quantity-like, optional
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
    # Make figure
    if ax is None:
        fig, ax = plt.subplots(1, 1, **(subplots_kw or {}))
    else:
        fig = ax.figure

    # Figure out which elements are iterable, which are numeric
    _grids = []
    _slices = []
    for i, g in enumerate(grid):
        if isiterable(g):
            _grids.append((i, g))
        else:
            _slices.append((i, g))

    # Figure out the dimensionality
    ndim = len(_grids)

    if ndim == 0 or ndim > 2:
        msg = (
            f"ndim = {ndim}: you can only make contours on a 1D or 2D grid."
            " For other dimensions, you have to specify values to slice."
        )
        raise ValueError(msg)

    kwargs = {
        "ax": ax,
        "grids": _grids,
        "slices": _slices,
        "labels": labels,
        "kwargs": kwargs,
    }

    if ndim == 1:
        _plot_countours_1d(pot, t, **kwargs)

    else:
        _plot_countours_2d(pot, t, **kwargs, filled=filled)

    return fig


# ============================================================================
