"""Source and target particle sets for N-body force backends."""

__all__ = ["Sources", "Targets"]

from jaxtyping import Array

import equinox as eqx


class Sources(eqx.Module):
    """Particles that exert force.

    Positions ``q`` (Ns, 3), masses ``m`` and softenings ``h`` (Ns,).
    """

    q: Array
    m: Array
    h: Array


class Targets(eqx.Module):
    """Particles that feel force: positions ``q`` (Nt, 3) and softenings ``h`` (Nt,).

    ``self_index`` (Nt,) gives each target's index among the sources, or -1 if
    the target is not a source.
    """

    q: Array
    h: Array
    self_index: Array
