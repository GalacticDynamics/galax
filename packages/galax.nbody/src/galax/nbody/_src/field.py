"""Field for N-Body dynamics."""

__all__ = ["NBodyField", "nbody_acceleration"]

from jaxtyping import Array, Bool, Float, Real
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
from plum import dispatch

import unxt as u
from unxt.quantity import AllowValue

import galax.orbits.custom_types as gt
import galax.potential as gp
from .backend import AbstractForceBackend, Sources, Targets
from .direct import DirectSum
from .system import System
from galax.orbits import AbstractOrbitField

_NULL = gp.NullPotential(units="galactic")
_DIRECT = DirectSum()
_SAFE = 1e10


@final
class NBodyField(AbstractOrbitField):
    r"""Dynamics field for N-Body EoM.

    The first ``n_massive`` particles are sources. The rest are massless
    tracers that feel the sources but exert nothing. Masses after the first
    ``n_massive`` rows are ignored; those rows are massless tracers. Inactive
    particles (``active`` False) exert nothing and get zero acceleration, but
    their velocity passes through unchanged, so they keep drifting. Softening
    is given either as one ``eps`` for all particles or per particle as
    ``softening``.

    ``masses`` and ``softening`` are now stored as plain arrays in the external
    potential's units, no longer as Quantities. The scalar ``eps`` keyword
    still works and is stored as the per-particle ``softening``.

    .. warning::

        The call method currently returns a `tuple[Array[float, (3,)],
        Array[float, (3,)]]`. In future, when `unxt.Quantity` is registered with
        `quax.quaxify` for `diffrax.diffeqsolve` then this will return
        `tuple[Quantity[float, (3,), 'length'], Quantity[float, (3,),
        'speed']]`. Later, when `coordinax.AbstractVector` is registered with
        `quax.quaxify` for `diffrax.diffeqsolve` then this will return
        `tuple[CartesianPos3D, CartesianVel3D]`.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.nbody as gn
    >>> import galax.orbits as go
    >>> import galax.potential as gp

    >>> q = u.Q([[-1, 0, 0], [1, 0, 0]], "AU") / 2
    >>> p = u.Q([[0, -1, 0], [0, 1, 0]], "km/s") * 25

    >>> solver = go.OrbitSolver()

    >>> field = gn.NBodyField(
    ...     masses=u.Q([1, 1], "Msun"),
    ...     eps=u.Q(1e-4, "AU"),
    ...     external_potential=gp.NullPotential(units="solarsystem"))

    >>> t0, t1 = u.Q(0, "yr"), u.Q(2, "yr")
    >>> soln = solver.solve(field, (q, p), t0, t1)

    >>> soln.ys[0][-1, :, :].round(4)
    Array([[ 0.799 , -0.6521,  0.    ],
           [-0.799 ,  0.6521,  0.    ]], dtype=float64)

    """

    masses: Float[Array, " N"]
    softening: Float[Array, " N"]
    active: Bool[Array, " N"]
    n_massive: int = eqx.field(static=True)
    external_potential: gp.AbstractPotential = _NULL
    backend: AbstractForceBackend = _DIRECT

    def __init__(
        self,
        masses: Any,
        eps: Any = None,
        external_potential: gp.AbstractPotential = _NULL,
        *,
        softening: Any = None,
        active: Any = None,
        n_massive: int | None = None,
        backend: AbstractForceBackend = _DIRECT,
    ) -> None:
        if (eps is None) == (softening is None):
            msg = "pass exactly one of eps and softening"
            raise ValueError(msg)
        units = external_potential.units
        m = jnp.asarray(u.ustrip(AllowValue, units["mass"], masses), dtype=float)
        h = u.ustrip(AllowValue, units["length"], softening if eps is None else eps)
        n = m.shape[0]
        self.masses = m
        self.softening = jnp.broadcast_to(jnp.asarray(h, dtype=float), (n,))
        self.active = (
            jnp.ones(n, dtype=bool) if active is None else jnp.asarray(active, bool)
        )
        n_massive = n if n_massive is None else n_massive
        if not 0 <= n_massive <= n:
            msg = f"n_massive must be in [0, {n}], got {n_massive}"
            raise ValueError(msg)
        if not isinstance(m, jax.core.Tracer) and bool(jnp.any(m[n_massive:] != 0)):
            msg = (
                f"masses after the first n_massive={n_massive} rows must be zero; "
                "those rows are massless tracers"
            )
            raise ValueError(msg)
        self.n_massive = n_massive
        self.external_potential = external_potential
        self.backend = backend

    @classmethod
    def from_system(
        cls,
        system: System,
        external_potential: gp.AbstractPotential,
        backend: AbstractForceBackend = _DIRECT,
    ) -> "NBodyField":
        """Build a field from ``system``, converting it to the potential's units."""
        ps, us = system.particles, system.units
        return cls(
            u.Q(ps.m, us["mass"]),
            external_potential=external_potential,
            softening=u.Q(ps.h, us["length"]),
            active=ps.active,
            n_massive=system.n_massive,
            backend=backend,
        )

    @property  # type: ignore[misc]
    def units(self) -> u.AbstractUnitSystem:
        return self.external_potential.units

    @property
    def _G(self) -> gt.Sz0:
        us = self.units
        unit = us["length"] ** 3 / (us["mass"] * us["time"] ** 2)
        _result = self.external_potential.constants["G"].ustrip(unit)
        return _result  # type: ignore[no-any-return]

    def acceleration(self, t: Any, q: Float[Array, "N 3"]) -> Float[Array, "N 3"]:
        """Return the accelerations (N, 3) at positions ``q`` and time ``t``.

        Inactive rows get zero acceleration.
        """
        return nbody_acceleration(self, t, q)

    @dispatch.abstract  # type: ignore[override]
    def __call__(
        self, t: Any, qp: tuple[Any, Any], args: tuple[Any, ...], /
    ) -> tuple[Any, Any]:
        raise NotImplementedError  # pragma: no cover


def nbody_acceleration(
    field: NBodyField, t: Any, q: Float[Array, "N 3"]
) -> Float[Array, "N 3"]:
    """Return the accelerations (N, 3) of all particles in ``field``.

    This is the backend's pairwise term from the active massive particles plus
    the external term, zeroed for inactive particles. In ``__call__`` the
    velocity of an inactive row passes through unchanged, so it keeps drifting.
    """
    nm = field.n_massive
    idx = jnp.arange(q.shape[0])
    # Inactive rows are moved to distinct far-away points so neither their
    # forward values nor their gradients can poison the active rows.
    # ponytail: an active particle exactly at an inactive row's safe point with
    # h=0 still NaNs; mask pair weights in the backend if that matters.
    safe = jnp.zeros_like(q).at[:, 0].set(_SAFE * (1.0 + idx))
    q = jnp.where(field.active[:, None], q, safe)
    sources = Sources(
        q=q[:nm], m=(field.masses * field.active)[:nm], h=field.softening[:nm]
    )
    targets = Targets(q=q, h=field.softening, self_index=jnp.where(idx < nm, idx, -1))
    backend = field.backend
    # G multiplies outside the gradient policy, so StopGrad on m keeps dG and
    # NoGrad's custom_jvp never closes over a traced G.
    a = field._G * backend.accelerations(backend.prepare(sources), targets, G=1.0)  # noqa: SLF001
    a = a - jnp.asarray(gp.gradient(field.external_potential, q, t))
    return jnp.where(field.active[:, None], a, 0.0)


@NBodyField.__call__.dispatch
@jax.jit
def __call__(
    self: "NBodyField",
    t: gt.LikeSz0,
    xv: tuple[Real[Array, "N 3"], Real[Array, "N 3"]]
    | tuple[Real[u.AbstractQuantity, "N 3"], Real[u.AbstractQuantity, "N 3"]],
    args: Any,  # noqa: ARG001
    /,
) -> tuple[Real[Array, "N 3"], Real[Array, "N 3"]]:
    units = self.units
    x, v = xv
    x = u.ustrip(AllowValue, units["length"], x)
    v = u.ustrip(AllowValue, units["speed"], v)
    return v, self.acceleration(t, x)
