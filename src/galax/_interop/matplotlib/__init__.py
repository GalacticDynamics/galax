"""Matplotlib extension for :mod:`galax`."""

__all__ = [
    # potential
    "plot_potential_contours",
    "plot_density_contours",
    # orbit
    "plot_components",
]

from .orbit import plot_components
from .potential import plot_density_contours, plot_potential_contours
