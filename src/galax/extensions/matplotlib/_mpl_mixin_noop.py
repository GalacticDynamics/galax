__all__ = ["plot_contours"]


from typing import TYPE_CHECKING, Any, Never

from galax.potential._potential.base import AbstractPotentialBase

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_contours(
    pot: AbstractPotentialBase,
    /,
    grid: tuple[Any, ...],
    t: float = 0.0,
    *,
    filled: bool = True,
    ax: "Axes | None" = None,
    labels: tuple[str, ...] | None = None,
    subplots_kw: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Never:
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
    labels : iterable, optional keyword-only
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
    msg = r"No module named 'matplotlib'"
    raise ModuleNotFoundError(msg)
