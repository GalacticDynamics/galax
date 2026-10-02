"""Version and build probes for the gala interop.

Unlike the other three interop distributions, this module earns its place.
`gala>=1.10` is a required dependency, so `OptDeps.GALA.installed` alone would
be a constant `True` -- but two things here are genuinely conditional:

- **Version gating.** `potential.py` compares `OptDeps.GALA` against
  `Version("1.8.2")` and `Version("1.11")` to pick between gala APIs.
- **`GSL_ENABLED`.** gala can be installed without `_cconfig`, so this cannot
  be inferred from the dependency pin.

`.installed` is also not quite redundant: the probe is
`chain_checks(get_version("gala"), is_installed("gala.dynamics"))`, which
catches a half-built gala that a version pin does not.
"""

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
