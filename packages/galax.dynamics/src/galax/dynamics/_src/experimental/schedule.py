"""Release schedules for experimental spray streams."""

__all__: tuple[str, ...] = ()

from jaxtyping import Array

import equinox as eqx
import jax.numpy as jnp


class Uniform(eqx.Module):
    """Uniformly spaced releases on a fixed-step time lattice.

    The lattice runs from ``t1 - t_age`` to ``t1`` in ``n_steps`` equal steps.
    Releases sit on every ``steps_per_release``-th lattice point, starting at the
    first, so a fixed-step solve over the lattice never needs a release time
    snapped to it.

    Parameters
    ----------
    t_age : Array
        Time from the first release to ``t1`` (traced, differentiable).
    n_release : int
        Number of releases (static).
    steps_per_release : int
        Lattice steps between consecutive releases (static).
    t1 : Array
        Final time of the lattice.

    """

    t_age: Array
    n_release: int = eqx.field(static=True)
    steps_per_release: int = eqx.field(default=1, static=True)
    t1: Array = eqx.field(default_factory=lambda: jnp.asarray(0.0))

    def n_steps(self) -> int:
        """Return the number of lattice steps (one step for a single release)."""
        return (
            1 if self.n_release == 1 else self.steps_per_release * (self.n_release - 1)
        )

    def lattice(self) -> Array:
        """Lattice points ``t1 - (n_steps - k) * h``, the last exactly ``t1``."""
        n = self.n_steps()
        h = jnp.asarray(self.t_age, dtype=float) / n
        return jnp.asarray(self.t1, dtype=float) - (n - jnp.arange(n + 1)) * h

    def release_index(self) -> Array:
        """Lattice index of each release."""
        return jnp.arange(self.n_release) * self.steps_per_release

    def release_times(self) -> Array:
        """Release times, exactly lattice points."""
        return self.lattice()[self.release_index()]
