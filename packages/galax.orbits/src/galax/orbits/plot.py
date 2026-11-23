""":mod:`galax.orbits.plot`.

The plotting backends (and the astropy conversions) are registered by
`import galax.dynamics`, which loads the interop plugins, so with `galax.orbits`
alone `Orbit.plot()` raises a dispatch error until `galax.dynamics` is imported.
"""

__all__ = ["plot_components"]

from ._src.plot_helper import plot_components
