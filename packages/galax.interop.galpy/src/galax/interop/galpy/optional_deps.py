"""Optional dependency check for the galpy interop."""

__all__ = ["OptDeps"]

from optional_dependencies import OptionalDependencyEnum, auto


class OptDeps(OptionalDependencyEnum):  # type: ignore[misc]
    """Optional dependencies for ``galax.interop.galpy``."""

    GALPY = auto()
