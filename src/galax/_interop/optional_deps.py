"""Optional dependencies."""

__all__ = ["OptDeps", "GSL_ENABLED"]


from optional_dependencies import OptionalDependencyEnum, auto
from optional_dependencies.utils import chain_checks, get_version


def _double_check_gala() -> bool:
    """Double check that gala is installed."""
    try:
        import gala.dynamics  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


class OptDeps(OptionalDependencyEnum):  # type: ignore[misc]
    """Optional dependencies for ``galax``."""

    ASTROPY = auto()
    GALA = chain_checks(get_version("gala"), _double_check_gala())
    GALPY = auto()
    MATPLOTLIB = auto()


GSL_ENABLED: bool
if OptDeps.GALA.is_installed:
    from gala._cconfig import GSL_ENABLED
else:
    GSL_ENABLED = False
