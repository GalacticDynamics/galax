"""Optional dependencies."""

__all__ = ["HAS_GALA", "GSL_ENABLED"]

import importlib.metadata
from typing import Literal

from packaging.version import Version, parse


def get_version(package_name: str) -> Version | Literal[False]:
    """Get the version of a package."""
    try:
        # Get the version string of the package
        version_str = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return False
    # Parse the version string using packaging.version.parse
    return parse(version_str)


HAS_GALA: Version | Literal[False] = get_version("gala")
if HAS_GALA:  # pragma: no cover  # gala can be installed incorrectly
    try:
        import gala.dynamics  # noqa: F401
    except Exception:  # noqa: BLE001
        HAS_GALA = False


GSL_ENABLED: bool
if HAS_GALA:
    from gala._cconfig import GSL_ENABLED
else:
    GSL_ENABLED = False
