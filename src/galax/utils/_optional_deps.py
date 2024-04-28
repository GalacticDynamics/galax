"""Optional dependencies."""

__all__ = ["HAS_GALA", "GSL_ENABLED"]

from importlib.util import find_spec

HAS_GALA: bool = find_spec("gala") is not None

GSL_ENABLED: bool
if HAS_GALA:
    from gala._cconfig import GSL_ENABLED
else:
    GSL_ENABLED = False
