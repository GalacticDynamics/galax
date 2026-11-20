"""Named particle components and the flat particle system built from them."""

__all__ = ["Component", "Particles", "System"]

from jaxtyping import Array, Float, Int, Real
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from plum import dispatch

import unxt as u
from unxt.quantity import AllowValue

from .checks import input_oks

AnyReal = Real[Array, "..."] | Real[u.AbstractQuantity, "..."] | float
_KINDS = ("massive", "tracer")


class Component(eqx.Module):
    """A named group of particles.

    ``q`` and ``p`` are (N, 3) positions and velocities. ``m`` and ``h`` are
    masses and softening lengths, either scalars or (N,). Each may be an array
    or a Quantity. ``kind`` says how the particles take part in the force sum:
    ``"massive"`` particles exert and feel forces, and ``"tracer"`` particles
    only feel them, so their ``m`` must be zero. Several massive components,
    for example dark matter and stars, may each have their own masses and
    softening.
    """

    name: str = eqx.field(static=True)
    q: AnyReal
    p: AnyReal
    m: AnyReal
    h: AnyReal
    kind: Literal["massive", "tracer"] = eqx.field(default="massive", static=True)

    def __check_init__(self) -> None:
        if self.kind not in _KINDS:
            msg = f"kind must be one of {_KINDS}, got {self.kind!r}"
            raise ValueError(msg)


class Particles(eqx.Module):
    """Flat per-particle arrays, in the units of the owning `System`."""

    q: Float[Array, "N 3"]
    p: Float[Array, "N 3"]
    m: Float[Array, " N"]
    h: Float[Array, " N"]
    id: Int[Array, " N"]


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
    ...     m=0.0, h=0.0, kind="tracer")
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
    @dispatch
    def from_(cls: "type[System]", *args: Any, **kwargs: Any) -> "System":
        """Build a `System`, by default with the constructor.

        Registered methods take a `System` (returned as is) and `Component`
        objects with ``units=`` (see `System.from_components`).
        """
        return cls(*args, **kwargs)

    @classmethod
    def from_components(cls, *components: Component, units: Any) -> "System":
        """Build a system from ``components``, converted to ``units``.

        Massive components are stable-sorted before tracers. ``id``
        follows the input order. The checks run on concrete values, so call
        this outside ``jit``.
        """
        if not components:
            msg = "pass at least one Component"
            raise ValueError(msg)
        usys = u.unitsystem(units)
        names = [c.name for c in components]
        if len(set(names)) != len(names):
            msg = f"duplicate component names in {names}"
            raise ValueError(msg)
        sizes = []
        for c in components:
            q, p = _strip(c.q, usys["length"]), _strip(c.p, usys["speed"])
            if q.ndim != 2 or q.shape[1] != 3 or p.shape != q.shape:
                msg = f"{c.name!r}: q, p shapes {q.shape}, {p.shape}; need (N, 3) both"
                raise ValueError(msg)
            n = q.shape[0]
            m, h = _strip(c.m, usys["mass"]), _strip(c.h, usys["length"])
            if m.shape not in ((), (n,)) or h.shape not in ((), (n,)):
                msg = f"{c.name!r}: m, h shapes {m.shape}, {h.shape}; need () or ({n},)"
                raise ValueError(msg)
            sizes.append(n)
        first_id = dict(zip(names, np.cumsum([0, *sizes[:-1]]).tolist(), strict=True))
        order = sorted(components, key=lambda c: c.kind == "tracer")  # stable
        n_of = dict(zip(names, sizes, strict=True))
        ns = [n_of[c.name] for c in order]

        def flat(attr: str, unit: str, shape: tuple[int, ...]) -> Array:
            return jnp.concat(
                [
                    jnp.broadcast_to(_strip(getattr(c, attr), usys[unit]), (n, *shape))
                    for c, n in zip(order, ns, strict=True)
                ]
            )

        m, h = flat("m", "mass", ()), flat("h", "length", ())
        which = np.repeat(np.arange(len(order)), ns)  # component of each row
        n_massive = sum(
            n for c, n in zip(order, ns, strict=True) if c.kind == "massive"
        )
        m_ok, tracer_ok, h_ok = input_oks(m, h, n_massive)
        for ok, what in (
            (m_ok, "negative or non-finite mass"),
            (h_ok, "negative or non-finite h"),
        ):
            if not np.all(ok):
                name = order[which[np.argmin(np.asarray(ok))]].name
                msg = f"component {name!r} has {what}"
                raise ValueError(msg)
        if not np.all(tracer_ok):
            name = order[which[n_massive + np.argmin(np.asarray(tracer_ok))]].name
            msg = f"component {name!r} is a tracer but has nonzero m"
            raise ValueError(msg)
        ends = np.cumsum(ns).tolist()
        particles = Particles(
            q=flat("q", "length", (3,)),
            p=flat("p", "speed", (3,)),
            m=m,
            h=h,
            id=jnp.concat(
                [
                    first_id[c.name] + jnp.arange(n)
                    for c, n in zip(order, ns, strict=True)
                ]
            ),
        )
        return cls(
            particles=particles,
            names=tuple(c.name for c in order),
            slices=tuple((e - n, e) for e, n in zip(ends, ns, strict=True)),
            n_massive=n_massive,
            units=usys,
        )

    def split(self, x: Array) -> dict[str, Array]:
        """Split ``x`` along axis 0 into a dict keyed by component name."""
        return {n: x[a:b] for n, (a, b) in zip(self.names, self.slices, strict=True)}


def _strip(x: Any, unit: Any) -> Array:
    return jnp.asarray(u.ustrip(AllowValue, unit, x), dtype=float)


@System.from_.dispatch
def from_(cls: type[System], system: System, /) -> System:  # noqa: ARG001
    """Return ``system`` unchanged.

    >>> import jax.numpy as jnp
    >>> import galax.nbody as gn
    >>> z = jnp.zeros((1, 3))
    >>> c = gn.Component(name="a", q=z, p=z, m=1.0, h=0.0)
    >>> s = gn.System.from_(c, units="galactic")
    >>> gn.System.from_(s) is s
    True
    """
    return system


@System.from_.dispatch
def from_(cls: type[System], *components: Component, units: Any) -> System:
    """Build a `System` from `Component` objects; see `System.from_components`."""
    return cls.from_components(*components, units=units)


@System.from_.dispatch
def from_(
    cls: type[System],
    components: list[Component] | tuple[Component, ...],
    /,
    *,
    units: Any,
) -> System:
    """Build a `System` from a list or tuple of `Component` objects."""
    return cls.from_components(*components, units=units)
