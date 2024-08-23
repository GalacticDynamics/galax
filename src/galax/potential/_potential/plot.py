"""Plotting on Potentials.

This module uses code from the `bound-class` package, which is licensed under a
BSD 3-Clause License. The original code can be found at
https:://github.com/nstarman/bound-class. See the license in the LICENSE files.
"""

__all__: list[str] = []

from typing import Any, final

from plum import PromisedType, dispatch

from galax.utils._boundinstance import BndTo, InstanceDescriptor

ProxyAbstractPotentialBase = PromisedType("AbstractPotentialBase")


class AbstractPlottingBackend:
    """Abstract base class for plotting backends."""


@final
class MatplotlibBackend(AbstractPlottingBackend):
    """Matplotlib plotting backend."""


# --------------------------------------------------


class PlotDescriptor(InstanceDescriptor[BndTo]):
    """Descriptor for plotting functions."""

    def potential_contours(
        self,
        backend: type[AbstractPlottingBackend] = MatplotlibBackend,
        **kwargs: Any,
    ) -> Any:
        """Plot equipotentials contours.

        This calls `galax.potential.plot.plot_potential_contours`.

        """
        return plot_potential_contours(self.enclosing, backend, **kwargs)

    def density_contours(
        self,
        backend: type[AbstractPlottingBackend] = MatplotlibBackend,
        **kwargs: Any,
    ) -> Any:
        """Plot density contours.

        This calls `galax.potential.plot.plot_density_contours`.

        """
        return plot_density_contours(self.enclosing, backend, **kwargs)


# --------------------------------------------------


@dispatch.abstract  # type: ignore[misc]
def plot_potential_contours(
    pot: ProxyAbstractPotentialBase,  # type: ignore[valid-type]  # noqa: ARG001
    backend: type[AbstractPlottingBackend] = MatplotlibBackend,  # noqa: ARG001
    /,
    **kwargs: Any,  # noqa: ARG001
) -> Any:
    """Plot equipotentials contours of the potential.

    Parameters
    ----------
    pot : AbstractPotentialBase, positional-only
        The potential for which to plot equipotential contours.

    backend: type[AbstractPlottingBackend], optional positional-only
        The plotting backend to use. Default is `MatplotlibBackend`.

    **kwargs : Any, optional
        Additional keyword arguments to pass to the plotting backend.

    Returns
    -------
    Any
        The return value equipoential contours plot. The return type is
        determined by the plotting backend.

    """
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract  # type: ignore[misc]
def plot_density_contours(
    pot: ProxyAbstractPotentialBase,  # type: ignore[valid-type]  # noqa: ARG001
    backend: type[AbstractPlottingBackend] = MatplotlibBackend,  # noqa: ARG001
    /,
    **kwargs: Any,  # noqa: ARG001
) -> Any:
    """Plot density contours of the potential.

    Parameters
    ----------
    pot : AbstractPotentialBase, positional-only
        The potential for which to plot density contours.

    backend: type[AbstractPlottingBackend], optional positional-only
        The plotting backend to use. Default is `MatplotlibBackend`.

    **kwargs : Any, optional
        Additional keyword arguments to pass to the plotting backend.

    Returns
    -------
    Any
        The return value density contours plot. The return type is determined
        by the plotting backend.

    """
    raise NotImplementedError  # pragma: no cover


# NOTE: this avoids a circular import
# isort: split
from .base import AbstractPotentialBase  # noqa: E402

ProxyAbstractPotentialBase.deliver(AbstractPotentialBase)
