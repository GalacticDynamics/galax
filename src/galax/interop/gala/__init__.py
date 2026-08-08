""":mod:`galax` <-> :mod:`gala` interoperability.

This package is loaded through the ``galax.*.interop`` entry-point groups, which
import the individual registration submodules directly. Importing anything here
would pull sibling submodules -- and therefore other `galax` portions -- in while
a portion is still initialising, so this module deliberately stays empty.
"""

__all__: list[str] = []
