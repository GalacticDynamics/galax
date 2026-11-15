""":mod:`galax.orbits`."""

__all__: list[str] = []

from .setup_package import install_import_hook

with install_import_hook("galax.orbits"):
    pass

# Cleanup
del install_import_hook
