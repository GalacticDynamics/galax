""":mod:`galax.dynamics.nbody`, re-exporting :mod:`galax.nbody`.

The names come from ``galax.nbody.__all__``, so new public names appear here
without edits.
"""

import galax.nbody as gn

__all__ = list(gn.__all__)
globals().update({name: getattr(gn, name) for name in __all__})

# Cleanup
del gn
