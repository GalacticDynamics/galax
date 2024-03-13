"""Matplotlib extension for :mod:`galax`."""

__all__ = ["plot_contours"]

from galax.utils._optional_deps import HAS_MATPLOTLIB

if HAS_MATPLOTLIB:
    from ._mpl_mixin import plot_contours
else:
    from ._mpl_mixin_noop import plot_contours
