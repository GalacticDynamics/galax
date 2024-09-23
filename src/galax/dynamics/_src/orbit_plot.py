"""Plotting on Potentials.

This module uses code from the `bound-class` package, which is licensed under a
BSD 3-Clause License. The original code can be found at
https:://github.com/nstarman/bound-class. See the license in the LICENSE files.
"""

__all__ = ["plot_components"]

from typing import Any

from plum import PromisedType, dispatch

from galax.utils._boundinstance import BndTo, InstanceDescriptor
from galax.utils.plot import AbstractPlottingBackend, MatplotlibBackend

ProxyOrbit = PromisedType("Orbit")


# --------------------------------------------------


class PlotOrbitDescriptor(InstanceDescriptor[BndTo]):
    """Descriptor for plotting functions."""

    def plot(
        self,
        backend: type[AbstractPlottingBackend] = MatplotlibBackend,
        **kwargs: Any,
    ) -> Any:
        """Plot specified components of the orbit.

        This calls `galax.dynamics.plot.plot_components`.

        """
        return plot_components(self.enclosing, backend, **kwargs)

    __call__ = plot

    # TODO: projection(...):


# --------------------------------------------------


@dispatch.abstract  # type: ignore[misc]
def plot_components(
    orbit: ProxyOrbit,  # type: ignore[valid-type]
    backend: type[AbstractPlottingBackend] = MatplotlibBackend,
    /,
    **kwargs: Any,
) -> Any:
    """Plot components of the orbit.

    Parameters
    ----------
    orbit : Orbit, positional-only
        The orbit for which to plot components.

    backend: type[AbstractPlottingBackend], optional positional-only
        The plotting backend to use. Default is `MatplotlibBackend`.

    **kwargs : Any, optional
        Additional keyword arguments to pass to the plotting backend.

    Returns
    -------
    Any
        The return type is determined by the plotting backend.

    """
    raise NotImplementedError  # pragma: no cover
