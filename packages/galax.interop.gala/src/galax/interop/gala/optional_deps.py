"""Optional dependency check for the gala interop."""

__all__ = ["GSL_ENABLED", "OptDeps"]

from optional_dependencies import OptionalDependencyEnum
from optional_dependencies.utils import chain_checks, get_version, is_installed


class OptDeps(OptionalDependencyEnum):  # type: ignore[misc]
    """Optional dependencies for ``galax.interop.gala``."""

    # `gala` can be importable while `gala.dynamics` is not, in a partial or
    # half-built install. Chaining the checks makes `.installed` answer `False`
    # rather than letting the inner ImportError escape at probe time.
    GALA = chain_checks(get_version("gala"), is_installed("gala.dynamics"))


GSL_ENABLED: bool
if OptDeps.GALA.installed:
    from gala._cconfig import GSL_ENABLED
else:
    GSL_ENABLED = False
