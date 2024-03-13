"""Optional dependencies."""

__all__ = ["HAS_GALA"]

from importlib.util import find_spec

HAS_GALA = find_spec("gala") is not None
"""Whether gala is installed."""

HAS_MATPLOTLIB = find_spec("matplotlib.pyplot") is not None
"""Whether matplotlib is installed."""
