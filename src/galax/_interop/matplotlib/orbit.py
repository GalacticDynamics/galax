__all__ = ["plot_components"]

from typing import Any, Protocol, runtime_checkable

from matplotlib.axes import Axes
from plum import dispatch

import unxt as u
from plotting_backends import MatplotlibBackend
from unxt.quantity import AbstractQuantity

import galax.dynamics as gd
from .potential import _get_figure


def _get_component(orbit: gd.Orbit, coord: str) -> AbstractQuantity:
    if hasattr(orbit.q, coord):
        out = getattr(orbit.q, coord)
    elif hasattr(orbit.p, coord):
        out = getattr(orbit.p, coord)
    elif coord.startswith("d_"):
        out = getattr(orbit.q, coord[2:])
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
    x: str,
    y: str,
    plot_function: str | PlotFunctionCallable = "plot",
    vector_representation: Any = None,
    ax: Any | None = None,
    subplots_kw: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Axes:
    # Process figure and axes
    _, ax = _get_figure(ax, subplots_kw)

    if vector_representation is not None:
        orbit = orbit.vconvert(vector_representation)

    # get the x, y data
    x_data = _get_component(orbit, x)
    y_data = _get_component(orbit, y)

    # intercept color
    if "c" in kwargs and kwargs["c"] == "orbit.t":
        kwargs["c"] = u.ustrip(orbit.potential.units["time"], orbit.t)

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
