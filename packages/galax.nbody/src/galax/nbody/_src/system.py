"""Named particle components and the flat particle system built from them."""

__all__ = ["Component", "Particles", "System", "system_from_components"]

from collections.abc import Sequence
from jaxtyping import Array, Bool, Float, Int, Real
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import unxt as u
from unxt.quantity import AllowValue

AnyReal = Real[Array, "..."] | Real[u.AbstractQuantity, "..."] | float


class Component(eqx.Module):
    """A named group of particles.

    ``q`` and ``p`` are (N, 3) positions and velocities. ``m`` and ``h`` are
    masses and softening lengths, either scalars or (N,). Each may be an array
    or a Quantity. A component with ``massive=False`` exerts no force, so its
    ``m`` must be zero.
    """

    name: str = eqx.field(static=True)
    q: AnyReal
    p: AnyReal
    m: AnyReal
    h: AnyReal
    massive: bool = eqx.field(default=True, static=True)


class Particles(eqx.Module):
    """Flat per-particle arrays, in the units of the owning `System`."""

    q: Float[Array, "N 3"]
    p: Float[Array, "N 3"]
    m: Float[Array, " N"]
    h: Float[Array, " N"]
    species: Int[Array, " N"]
    id: Int[Array, " N"]
    active: Bool[Array, " N"]


class System(eqx.Module):
    """Particles from several components, massive components first.

    ``names`` and ``slices`` give each component's rows of ``particles``, and
    the first ``n_massive`` rows are the massive ones.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import galax.nbody as gn
    >>> tracers = gn.Component(
    ...     name="tracers", q=jnp.ones((3, 3)), p=jnp.zeros((3, 3)),
    ...     m=0.0, h=0.0, massive=False)
    >>> stars = gn.Component(
    ...     name="stars", q=jnp.zeros((2, 3)), p=jnp.zeros((2, 3)), m=1e9, h=0.01)
    >>> system = gn.System.from_components(tracers, stars, units="galactic")
    >>> system.names, system.slices, system.n_massive
    (('stars', 'tracers'), ((0, 2), (2, 5)), 2)
    >>> system.particles.id
    Array([3, 4, 0, 1, 2], dtype=int64)

    """

    particles: Particles
    names: tuple[str, ...] = eqx.field(static=True)
    slices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    n_massive: int = eqx.field(static=True)
    units: u.AbstractUnitSystem = eqx.field(static=True)

    @classmethod
    def from_components(cls, *components: Component, units: Any) -> "System":
        """Build a system from components, converting to ``units``."""
        return system_from_components(components, units=units)

    def split(self, x: Array) -> dict[str, Array]:
        """Split ``x`` along axis 0 into a dict keyed by component name."""
        return {n: x[a:b] for n, (a, b) in zip(self.names, self.slices, strict=True)}


def _strip(x: Any, unit: Any) -> Array:
    return jnp.asarray(u.ustrip(AllowValue, unit, x), dtype=float)


def system_from_components(components: Sequence[Component], *, units: Any) -> System:
    """Return a `System` from ``components``, converted to ``units``.

    Massive components are stable-sorted before massless ones. ``id`` follows
    the input order. The checks run on concrete values, so call this outside
    ``jit``.
    """
    usys = u.unitsystem(units)
    names = [c.name for c in components]
    if len(set(names)) != len(names):
        msg = f"duplicate component names in {names}"
        raise ValueError(msg)

    rows = []
    start = 0
    for c in components:
        q = _strip(c.q, usys["length"])
        p = _strip(c.p, usys["speed"])
        if q.ndim != 2 or q.shape[1] != 3 or p.shape != q.shape:
            msg = f"{c.name!r}: q, p shapes {q.shape}, {p.shape}; need (N, 3) both"
            raise ValueError(msg)
        n = q.shape[0]
        m, h = _strip(c.m, usys["mass"]), _strip(c.h, usys["length"])
        if m.shape not in ((), (n,)) or h.shape not in ((), (n,)):
            msg = f"{c.name!r}: m, h shapes {m.shape}, {h.shape}; need () or ({n},)"
            raise ValueError(msg)
        if not np.all(np.asarray(m) >= 0):
            msg = f"component {c.name!r} has negative mass"
            raise ValueError(msg)
        if not np.all(np.asarray(h) >= 0):
            msg = f"component {c.name!r} has negative h"
            raise ValueError(msg)
        if not c.massive and np.any(np.asarray(m) != 0):
            msg = f"component {c.name!r} is massless but has nonzero m"
            raise ValueError(msg)
        ids = jnp.arange(start, start + n)
        rows.append(
            (c, q, p, jnp.broadcast_to(m, (n,)), jnp.broadcast_to(h, (n,)), ids)
        )
        start += n

    rows.sort(key=lambda r: not r[0].massive)  # stable: keeps order within groups
    sizes = [r[1].shape[0] for r in rows]
    ends = np.cumsum(sizes).tolist()
    particles = Particles(
        q=jnp.concat([r[1] for r in rows]),
        p=jnp.concat([r[2] for r in rows]),
        m=jnp.concat([r[3] for r in rows]),
        h=jnp.concat([r[4] for r in rows]),
        species=jnp.concat([jnp.full(n, i) for i, n in enumerate(sizes)]),
        id=jnp.concat([r[5] for r in rows]),
        active=jnp.ones(start, dtype=bool),
    )
    return System(
        particles=particles,
        names=tuple(r[0].name for r in rows),
        slices=tuple((e - n, e) for e, n in zip(ends, sizes, strict=True)),
        n_massive=sum(n for r, n in zip(rows, sizes, strict=True) if r[0].massive),
        units=usys,
    )
