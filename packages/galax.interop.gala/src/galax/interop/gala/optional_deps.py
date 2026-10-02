"""Build facts about the installed gala.

`gala>=1.10` is a required dependency of this distribution, and the root test
suite skips this whole tree when the distribution is not importable, so
*whether* gala is installed is never the question here. Two things about it
still are:

- **`GSL_ENABLED`** -- gala builds optionally against GSL, and several
  conversions exist only in the GSL build. No version or dependency pin can
  express this; it has to be read from `gala._cconfig`, which some builds omit.
- **`GALA_VERSION`** -- a few conversions changed shape across gala releases
  and are gated on 1.8.2 and 1.11.

Neither needs an optional-dependency enum, which is why this distribution does
not depend on `optional-dependencies`.
"""

__all__ = ["GALA_VERSION", "GSL_ENABLED"]

from importlib.metadata import version

from packaging.version import Version

GALA_VERSION = Version(version("gala"))


def _gsl_enabled() -> bool:
    """Whether the installed gala was built against GSL.

    A function rather than a bare module-level `try`, so the no-GSL branch is
    reachable from a test. CI's gala is built with GSL, so that branch would
    otherwise never execute and would have to be excluded from coverage --
    which is the wrong trade for a configuration that really exists.
    """
    try:
        from gala._cconfig import GSL_ENABLED
    except ImportError:  # a gala built without GSL ships no `_cconfig`
        return False
    return bool(GSL_ENABLED)


GSL_ENABLED: bool = _gsl_enabled()
