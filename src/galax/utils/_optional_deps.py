"""Optional dependencies."""

__all__ = ["HAS_GALA", "GSL_ENABLED"]

from importlib.util import find_spec

HAS_GALA: bool = find_spec("gala") is not None
if HAS_GALA:  # pragma: no cover  # TODO: remove this check
    try:
        import gala.dynamics  # noqa: F401
    except Exception:  # noqa: BLE001
        HAS_GALA = False


GSL_ENABLED: bool
if HAS_GALA:
    from gala._cconfig import GSL_ENABLED
else:
    GSL_ENABLED = False
