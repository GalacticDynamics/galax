__all__ = ["plot_density_contours", "plot_potential_contours"]


from typing import Any

import matplotlib.pyplot as plt
from astropy.utils import isiterable
from matplotlib.axes import Axes
from matplotlib.cm import Blues
from matplotlib.figure import Figure
from plum import dispatch

import quaxed.numpy as jnp
import unxt as u
from plotting_backends import MatplotlibBackend

from galax.potential._src.base import AbstractBasePotential


def _get_figure(
    ax: Axes | None, subplots_kw: dict[str, Any] | None
) -> tuple[Figure, Axes]:
    # Process figure and axes
    if ax is None:  # make new figure
        fig, ax = plt.subplots(1, 1, **(subplots_kw or {}))
    else:  # use existing figure
        fig = ax.figure

    return fig, ax


def _parse_grid(
    grid: tuple[u.Quantity | int, ...],
) -> tuple[list[tuple[int, Any]], list[tuple[int, Any]]]:
    _grids: list[tuple[int, Any]] = []
    _slices: list[tuple[int, Any]] = []
    for i, g in enumerate(grid):
        if isiterable(g):
            _grids.append((i, g))
        else:
            _slices.append((i, g))
    return _grids, _slices


# ============================================================================
# Plot potential contours


@dispatch  # type: ignore[misc]
def plot_potential_contours(
    pot: AbstractBasePotential,
    _: type[MatplotlibBackend] = MatplotlibBackend,
    /,
    *,
    grid: tuple[u.Quantity | int, ...],
    t: u.Quantity["time"] = u.Quantity(0.0, "Myr"),  # noqa: B008
    filled: bool = True,
    ax: Any | None = None,
    labels: tuple[str, ...] | None = None,
    subplots_kw: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Figure:
    """Plot contours of the potential.

    Parameters
    ----------
    pot : :class:`~galax.potential.base.AbstractBasePotential`
        Potential object to plot.
    backend : type[:class:`~galax.potential.plot.MatplotlibBackend`]
        The Matplotlib plotting backend.

    grid : tuple[Any, ...]
        Coordinate grids or slice value for each dimension. Should be a
        tuple of 1D arrays or numbers.
    t : u.Quantity["time"], optional
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
    fig, ax = _get_figure(ax, subplots_kw)

    # Figure out which elements are iterable, which are numeric
    _grids, _slices = _parse_grid(grid)

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
    pot: AbstractBasePotential,
    t: u.Quantity["time"],
    *,
    ax: Axes,
    grids: list[Any],
    slices: list[Any],
    labels: tuple[str, ...] | None,
    kwargs: dict[str, Any],
) -> None:
    x1 = u.uconvert(pot.units["length"], grids[0][1])

    # Create q array
    q = jnp.zeros((len(x1), len(grids) + len(slices)))
    q = q.at[:, grids[0][0]].set(x1)
    for ii, slc in slices:
        q = q.at[:, ii].set(slc)
    q = u.Quantity(q, pot.units["length"])

    # Evaluate potential
    Z = pot.potential(q, t)

    # Plot potential
    ax.plot(x1, u.ustrip(pot.units["specific energy"], Z), **kwargs)

    if labels is not None:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(f"potential [{pot.units['specific energy']}]")


def _plot_potential_countours_2d(
    pot: AbstractBasePotential,
    t: u.Quantity["time"],
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
    x1, x2 = jnp.meshgrid(
        u.ustrip(pot.units["length"], grids[0][1]),
        u.ustrip(pot.units["length"], grids[1][1]),
    )
    shape = x1.shape

    # Create q array
    # TODO: use Quantity.at when it's implemented
    q = jnp.zeros((x1.size, len(grids) + len(slices)))
    q = q.at[:, grids[0][0]].set(jnp.ravel(x1))
    q = q.at[:, grids[1][0]].set(jnp.ravel(x2))
    for ii, slc in slices:
        q = q.at[:, ii].set(slc)
    q = u.Quantity(q, pot.units["length"])

    # Evaluate potential
    Z = pot.potential(q, t)

    # Plot contours
    kwargs.setdefault("cmap", Blues)  # better default colormap
    plot_func = ax.contourf if filled else ax.contour
    plot_func(
        x1, x2, u.ustrip(pot.units["specific energy"], Z.reshape(shape)), **kwargs
    )

    if labels is not None:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])


# ============================================================================
# Plot density contours


@dispatch  # type: ignore[misc]
def plot_density_contours(
    pot: AbstractBasePotential,
    _: type[MatplotlibBackend] = MatplotlibBackend,
    /,
    *,
    grid: tuple[u.Quantity | int, ...],
    t: u.Quantity["time"] = u.Quantity(0.0, "Myr"),  # noqa: B008
    filled: bool = True,
    ax: Axes | None = None,
    labels: tuple[str, ...] | None = None,
    subplots_kw: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Figure:
    """Plot density contours.

    Computes the density on a grid (specified by the array `grid`).

    Parameters
    ----------
    grid : tuple
        Coordinate grids or slice value for each dimension. Should be a tuple of
        1D arrays or numbers.
    t : quantity-like (optional)
        The time to evaluate at.
    filled : bool (optional)
        Use :func:`~matplotlib.pyplot.contourf` instead of
        :func:`~matplotlib.pyplot.contour`. Default is ``True``.
    ax : matplotlib.Axes (optional) labels : iterable (optional)
        List of axis labels.
    subplots_kw : dict
        kwargs passed to matplotlib's subplots() function if an axes object is
        not specified.
    kwargs : dict
        kwargs passed to either contourf() or plot().

    Returns
    -------
    fig : `~matplotlib.Figure`

    """
    # Process figure and axes
    fig, ax = _get_figure(ax, subplots_kw)

    # Figure out which elements are iterable, which are numeric
    _grids, _slices = _parse_grid(grid)

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
            _plot_density_countours_1d(pot, t, **kwargs)
        case 2:
            _plot_density_countours_2d(pot, t, filled=filled, **kwargs)
        # TODO: implement 3D contours
        case _:
            msg = (
                f"ndim = {ndim}: you can only make contours on a 1D or 2D grid."
                " For other dimensions, you have to specify values to slice."
            )
            raise ValueError(msg)

    return fig


def _plot_density_countours_1d(
    pot: AbstractBasePotential,
    t: u.Quantity["time"],
    *,
    ax: Axes,
    grids: list[Any],
    slices: list[Any],
    labels: tuple[str, ...] | None,
    kwargs: dict[str, Any],
) -> None:
    x1 = u.ustrip(pot.units["length"], grids[0][1])

    # Create q array
    q = jnp.zeros((len(x1), len(grids) + len(slices)))
    q = q.at[:, grids[0][0]].set(x1)
    for ii, slc in slices:
        q = q.at[:, ii].set(slc)
    q = u.Quantity(q, pot.units["length"])

    # Evaluate density
    Z = pot.density(q, t)

    # Plot mass density
    ax.plot(x1, u.ustrip(pot.units["mass density"], Z), **kwargs)

    if labels is not None:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(f"density [{pot.units['mass density']}]")


def _plot_density_countours_2d(
    pot: AbstractBasePotential,
    t: u.Quantity["time"],
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
    x1, x2 = jnp.meshgrid(
        u.ustrip(pot.units["length"], grids[0][1]),
        u.ustrip(pot.units["length"], grids[1][1]),
    )
    shape = x1.shape

    # Create q array
    # TODO: use Quantity.at when it's implemented
    q = jnp.zeros((x1.size, len(grids) + len(slices)))
    q = q.at[:, grids[0][0]].set(jnp.ravel(x1))
    q = q.at[:, grids[1][0]].set(jnp.ravel(x2))
    for ii, slc in slices:
        q = q.at[:, ii].set(slc)
    q = u.Quantity(q, pot.units["length"])

    # Evaluate potential
    Z = pot.density(q, t)

    # Plot contours
    kwargs.setdefault("cmap", Blues)  # better default colormap
    plot_func = ax.contourf if filled else ax.contour
    plot_func(x1, x2, u.ustrip(pot.units["mass density"], Z.reshape(shape)), **kwargs)

    if labels is not None:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
