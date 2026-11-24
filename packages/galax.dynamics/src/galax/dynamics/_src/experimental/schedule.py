"""Release schedules for experimental spray streams."""

__all__: tuple[str, ...] = ()

from jaxtyping import Array, ArrayLike

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

    def __check_init__(self) -> None:
        if self.n_release < 1:
            msg = f"n_release must be at least 1, got {self.n_release}"
            raise ValueError(msg)
        if self.steps_per_release < 1:
            msg = f"steps_per_release must be at least 1, got {self.steps_per_release}"
            raise ValueError(msg)

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

    def grid(self) -> Array:
        """Return the solve's time grid, the lattice itself."""
        return self.lattice()

    def segments(self) -> tuple[Array, Array]:
        """Return the whole-stream segments ``(grids, on)``.

        One segment per stretch between releases (one for a single release):
        ``grids`` (S, L + 1) are its lattice points and ``on`` (S, n_release)
        the releases made by its start, which move in it.
        """
        n_seg = max(self.n_release - 1, 1)
        L = self.n_steps() // n_seg
        starts = jnp.arange(n_seg) * L
        grids = self.lattice()[starts[:, None] + jnp.arange(L + 1)]
        on = self.release_index()[None, :] <= starts[:, None]
        return grids, on


class Times(eqx.Module):
    """Releases at arbitrary times ``ts``, between the points of a lattice.

    The lattice has ``steps`` equal steps from ``min(ts)`` to ``t1``, computed
    as `Uniform`'s (`uniform`). A fixed-step solve steps on the merged grid of
    the lattice points and the release times, so every release is a grid point
    and no release time is snapped. A release sorts before a lattice point it
    equals, and tied releases keep the order of ``ts``, so the grid ends at the
    lattice's ``t1``. A tie gives a step of zero length: it leaves every value
    as it is, and gives the rows that travel it (a released row, from its
    release to the next grid point) the tangent of its length. Gradients at a
    tie are then the one-sided derivatives of that order: from the left for a
    release on a lattice point or at ``t1``, and for tied releases the side that
    keeps their order in ``ts``.

    Parameters
    ----------
    ts : Array
        Release times, shape ``(n,)``, in any order, each differentiable. They
        must not be after ``t1``.
    steps : int
        Number of lattice steps (static, at least 1).
    t1 : Array | None
        Final time; `None` means ``max(ts)``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import galax.dynamics as gd

    >>> sched = gd.experimental.Times(jnp.array([-120.0, -250.0]), 5, 0.0)
    >>> sched.lattice()
    Array([-250., -200., -150., -100.,  -50.,    0.], dtype=float64)
    >>> sched.grid()
    Array([-250., -250., -200., -150., -120., -100.,  -50.,    0.], dtype=float64)
    >>> sched.release_index()
    Array([4, 0], dtype=int64)

    """

    ts: Array
    steps: int = eqx.field(static=True)
    t1: Array

    def __init__(self, ts: ArrayLike, steps: int, t1: ArrayLike | None = None) -> None:
        self.ts = jnp.asarray(ts, dtype=float)
        self.steps = steps
        if self.ts.ndim != 1 or self.ts.shape[0] < 1:
            msg = f"ts must have shape (n,) with n >= 1, got {self.ts.shape}"
            raise ValueError(msg)
        self.t1 = jnp.max(self.ts) if t1 is None else jnp.asarray(t1, dtype=float)
        if self.steps < 1:
            msg = f"steps must be at least 1, got {self.steps}"
            raise ValueError(msg)
        if jnp.ndim(self.t1) != 0:
            msg = (
                f"t1 must be a scalar, got shape {jnp.shape(self.t1)}; vmap over "
                "a batch"
            )
            raise ValueError(msg)

    def uniform(self) -> Uniform:
        """Return the lattice as a `Uniform` schedule, one release per point."""
        return Uniform(self.t1 - jnp.min(self.ts), self.steps + 1, 1, self.t1)

    def lattice(self) -> Array:
        """Return the ``steps + 1`` lattice points, as `Uniform.lattice`."""
        return self.uniform().lattice()

    def _merged(self) -> Array:
        # Releases first, so the stable sort puts a release before a lattice
        # point it equals, and tied releases in the order of ``ts``.
        return jnp.concat([self.ts, self.lattice()])

    def grid(self) -> Array:
        """Return the merged grid: release times and lattice points, sorted.

        It has static length ``steps + 1 + n``; duplicates are kept. A release
        comes before a lattice point it equals, so the last point is ``t1``.
        """
        return jnp.sort(self._merged())

    def release_index(self) -> Array:
        """Grid index of each release, in the order of ``ts``."""
        return jnp.argsort(jnp.argsort(self._merged()))[: self.ts.shape[0]]

    def release_times(self) -> Array:
        """Return the release times ``ts``, in the given order."""
        return self.ts

    def segments(self) -> tuple[Array, Array]:
        """Return the whole-stream segments ``(grids, on)`` for an adaptive solve.

        One segment between each pair of consecutive sorted release times, then
        one to ``t1``: ``grids`` (n, 2) are its ends and ``on`` (n, n) the
        releases made by its start, which move in it.
        """
        u = jnp.sort(self.ts)
        grids = jnp.stack([u, jnp.concat([u[1:], self.t1[None]])], axis=1)
        # A NaN time gives NaN release states, so NaN rows: solve over zero
        # time, not NaN time.
        grids = jnp.where(jnp.all(jnp.isfinite(grids)), grids, 0.0)
        # By sorted position, not time: of tied releases, only those before
        # a segment's start in ``ts`` order move in it.
        rank = jnp.argsort(jnp.argsort(self.ts))
        on = rank[None, :] <= jnp.arange(self.ts.shape[0])[:, None]
        return grids, on
