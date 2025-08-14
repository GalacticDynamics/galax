__all__ = ["plot_components"]

from typing import Any, Protocol, cast, runtime_checkable

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from plum import dispatch

from plotting_backends import MatplotlibBackend
from unxt.quantity import AbstractQuantity

import galax.dynamics as gd
from .potential import _get_figure


def _get_component(orbit: gd.Orbit, coord: str) -> AbstractQuantity:
    if hasattr(orbit.q, coord):
        out = getattr(orbit.q, coord)
    elif hasattr(orbit.p, coord):
        out = getattr(orbit.p, coord)
    elif hasattr(orbit, coord):  # allows for, e.g., orbit.t
        out = getattr(orbit, coord)
    elif coord.startswith("d_"):
        out = getattr(orbit.p, coord[2:])
    else:
        msg = f"Orbit does not have attribute {coord}"
        raise AttributeError(msg) from None

    return out


@runtime_checkable
class PlotFunctionCallable(Protocol):
    """Protocol for a plotting functions, e.g. `matplotlib.pyplot.scatter`."""

    def __call__(
        self, x: AbstractQuantity, y: AbstractQuantity, **kwargs: Any
    ) -> Axes: ...


@dispatch
def plot_components(
    orbit: gd.Orbit,
    backend: type[MatplotlibBackend] = MatplotlibBackend,  # noqa: ARG001
    /,
    *,
    x: str | None = None,
    y: str | None = None,
    plot_function: str | PlotFunctionCallable = "plot",
    vector_representation: Any = None,
    ax: Any | None = None,
    subplots_kw: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Axes | Any:
    if x is None and y is None:
        # Plot all components:
        return plot_all_components(orbit, **kwargs)

    if (x is None and y is not None) or (x is not None and y is None):
        msg = "Both x and y components must be specified, or neither."
        raise ValueError(msg)

    # Process figure and axes
    _, ax = _get_figure(ax, subplots_kw)

    if vector_representation is not None:
        orbit = orbit.vconvert(vector_representation)

    # get the x, y data
    x_data = _get_component(orbit, cast("str", x))
    y_data = _get_component(orbit, cast("str", y))

    # intercept color
    if "c" in kwargs and kwargs["c"] == "orbit.t":
        kwargs["c"] = orbit.t.value  # TODO: in right unit system

    # plot
    plot_fn = (
        plot_function
        if isinstance(plot_function, PlotFunctionCallable)
        else getattr(ax, plot_function)
    )
    _ = plot_fn(x_data, y_data, **kwargs, label="orbit")

    # labels
    ax.set_xlabel(f"{x} [{x_data.unit}]")
    ax.set_ylabel(f"{y} [{y_data.unit}]")

    return ax


def plot_all_components(
    orbit: gd.Orbit,
    /,
    *,
    axes: Any | None = None,
    **kwargs: Any,
) -> Any:  # TODO: type hint for array of axes is borked?
    subplots_kw = kwargs.pop("subplots_kw", None)
    if subplots_kw is None:
        subplots_kw = {}

    components = orbit.q.components
    xidxs, yidxs = jnp.triu_indices(len(components), 1)

    # TODO: 5 is an arbitrary number - is there a way to get this from matplotlib?
    subplots_kw.setdefault("figsize", (len(components) * 5, 5))
    if axes is None:
        _, axes = plt.subplots(1, len(xidxs), **subplots_kw)
    elif len(axes) != len(xidxs):
        msg = (
            f"Number of matplotlib axes ({len(axes)}) does not match number of "
            f"pairwise components to plot ({len(xidxs)})"
        )
        raise ValueError(msg)

    for ax, xidx, yidx in zip(axes, xidxs, yidxs, strict=True):
        plot_components(
            orbit,
            x=components[xidx],
            y=components[yidx],
            ax=ax,
            subplots_kw=None,
            **kwargs,
        )

    return axes
