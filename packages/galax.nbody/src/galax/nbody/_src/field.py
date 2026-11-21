"""Field for N-Body dynamics."""

__all__ = ["NBodyField", "nbody_acceleration"]

import operator

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
from .backend import AbstractForceBackend
from .checks import MASS_MSG, SOFT_MSG, check, input_oks, tracer_msg
from .direct import DirectSum
from .particles import Sources, Targets
from .system import System
from galax.orbits import AbstractOrbitField

_NULL = gp.NullPotential(units="galactic")
_DIRECT = DirectSum()
# Inactive rows and non-finite tracers are parked at multiples of this, in the
# field's length unit.
# ponytail: assumes the external potential and its gradient are finite out
# there. True for the built-in potentials in float64; in float32
# LeeSutoTriaxialNFWPotential overflows. Make it per-potential if that matters.
_SAFE = 1e10


@final
class NBodyField(AbstractOrbitField):
    r"""Dynamics field for N-Body EoM.

    The first ``n_massive`` particles are sources. The rest are massless
    tracers that feel the sources but exert nothing. Masses after the first
    ``n_massive`` rows must be zero. Inputs whose values are known when the
    field is built are checked then and raise `ValueError`. Traced inputs are
    checked with `equinox.error_if` each time the acceleration is computed and
    raise when it runs, with the same message. The exception class then
    depends on how JAX runs the call (for example
    `equinox.EquinoxRuntimeError` under `jax.vmap`, or
    `jax.errors.JaxRuntimeError` under `jax.jit`). Inactive
    particles (``active`` False) exert nothing and get zero acceleration, but
    their velocity passes through unchanged, so they keep drifting. Softening
    is given as exactly one of ``eps`` and ``softening``; either may be a
    scalar for all particles or an (N,) array.

    ``masses`` and ``softening`` are stored as plain arrays in the external
    potential's units. Swapping ``external_potential`` with `equinox.tree_at` keeps
    those numbers and so reinterprets them in the new units. `dataclasses.replace`
    does the same, because it passes the stored plain arrays back in unchanged.
    To change units, build a new field from Quantities.

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
        # Concrete inputs stay concrete here even inside a jit trace, so
        # `_check` sees their values and raises now. Traced inputs stay traced
        # and get the same checks at run time through `eqx.error_if`.
        with jax.ensure_compile_time_eval():
            m = jnp.asarray(u.ustrip(AllowValue, units["mass"], masses), dtype=float)
            h = softening if eps is None else eps
            h = jnp.asarray(u.ustrip(AllowValue, units["length"], h), dtype=float)
            if m.ndim != 1:
                msg = f"masses must be 1-D, got shape {m.shape}"
                raise ValueError(msg)
            n = m.shape[0]
            if h.shape not in ((), (n,)):
                which = "eps" if softening is None else "softening"
                msg = f"{which} must be a scalar or have shape ({n},), got {h.shape}"
                raise ValueError(msg)
            h = jnp.broadcast_to(h, (n,))
            a = jnp.ones(n, bool) if active is None else jnp.asarray(active, bool)
            if a.shape != (n,):
                msg = f"active must have shape ({n},), got {a.shape}"
                raise ValueError(msg)
            if isinstance(n_massive, bool):
                msg = f"n_massive must be an int, not a bool, got {n_massive!r}"
                raise TypeError(msg)
            n_massive = n if n_massive is None else operator.index(n_massive)
            if not 0 <= n_massive <= n:
                msg = f"n_massive must be in [0, {n}], got {n_massive}"
                raise ValueError(msg)
            m_ok, tracer_ok, h_ok = input_oks(m, h, n_massive)
        # `eqx.error_if` must stage outside `ensure_compile_time_eval`, or it
        # reports the wrong message. These traced checks only fire if the
        # array reaches an output; `nbody_acceleration` checks again on its
        # result so a force call always keeps them.
        m = check(m, m_ok, MASS_MSG)
        m = check(m, tracer_ok, tracer_msg(n_massive))
        h = check(h, h_ok, SOFT_MSG)
        self.masses = m
        self.softening = h
        self.active = a
        self.n_massive = n_massive
        self.external_potential = external_potential
        self.backend = backend

    @classmethod
    @dispatch
    def from_(cls: "type[NBodyField]", *args: Any, **kwargs: Any) -> "NBodyField":
        """Build an `NBodyField`, by default with the constructor.

        Registered methods take an `NBodyField` (returned as is) and a
        `System` with an external potential (see `NBodyField.from_system`).
        """
        return cls(*args, **kwargs)

    @classmethod
    def from_system(
        cls,
        system: System,
        external_potential: gp.AbstractPotential,
        backend: AbstractForceBackend = _DIRECT,
    ) -> "NBodyField":
        """Build a field from ``system``.

        ``system.units`` must equal ``external_potential.units``. The system's
        positions and velocities are plain arrays, and the field reads plain
        arrays in its own units, so a mismatch would silently misread them.
        Raises `ValueError` if the units differ.
        """
        ps, us = system.particles, system.units
        if us != external_potential.units:
            pu = external_potential.units
            msg = (
                f"system units {[str(us[d]) for d in us.base_dimensions]} differ "
                f"from the potential's units {[str(pu[d]) for d in pu.base_dimensions]}"
                "; build the System in the potential's units"
            )
            raise ValueError(msg)
        return cls(
            u.Q(ps.m, us["mass"]),
            external_potential=external_potential,
            softening=u.Q(ps.h, us["length"]),
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

        ``t`` and ``q`` are plain arrays in the field's units. ``__call__``
        strips units from the positions and velocities, but it takes ``t`` as a
        plain number, so a Quantity ``t`` is rejected there as well. Inactive
        rows get zero acceleration.
        """
        return nbody_acceleration(self, t, q)

    @dispatch.abstract  # type: ignore[override]
    def __call__(
        self, t: Any, qp: tuple[Any, Any], args: tuple[Any, ...], /
    ) -> tuple[Any, Any]:
        raise NotImplementedError  # pragma: no cover


@NBodyField.from_.dispatch
def from_(cls: type[NBodyField], field: NBodyField, /) -> NBodyField:  # noqa: ARG001
    """Return ``field`` unchanged."""
    return field


@NBodyField.from_.dispatch
def from_(cls: type[NBodyField], system: System, /, **kwargs: Any) -> NBodyField:
    """Build a field from ``system`` and keyword arguments for `from_system`.

    Plum dispatches on positional arguments only, so a potential passed as
    ``external_potential=`` arrives here. Without it this raises `TypeError`.
    """
    if "external_potential" not in kwargs:
        msg = "pass the external potential too: NBodyField.from_(system, potential)"
        raise TypeError(msg)
    return cls.from_system(system, **kwargs)


@NBodyField.from_.dispatch
def from_(
    cls: type[NBodyField],
    system: System,
    external_potential: gp.AbstractPotential,
    /,
    backend: AbstractForceBackend = _DIRECT,
) -> NBodyField:
    """Build a field from ``system``; see `NBodyField.from_system`.

    >>> import jax.numpy as jnp
    >>> import galax.nbody as gn
    >>> import galax.potential as gp
    >>> z = jnp.zeros((2, 3))
    >>> c = gn.Component(name="a", q=z, p=z, m=1.0, h=0.1)
    >>> s = gn.System.from_(c, units="galactic")
    >>> f = gn.NBodyField.from_(s, gp.NullPotential(units="galactic"))
    >>> f.masses, gn.NBodyField.from_(f) is f
    (Array([1., 1.], dtype=float64), True)
    """
    return cls.from_system(system, external_potential, backend=backend)


def _use_rows(field: NBodyField, q: Float[Array, "N 3"]) -> Array:
    """Return which rows (N,) are evaluated where they are.

    Inactive rows and non-finite tracers are not. A non-finite massive row is,
    so its NaN reaches every particle, as its force does.
    """
    idx = jnp.arange(q.shape[0])
    finite = jnp.all(jnp.isfinite(q), axis=-1)
    return field.active & (finite | (idx < field.n_massive))


def _sources_targets(
    field: NBodyField, q: Float[Array, "N 3"]
) -> tuple[Float[Array, "N 3"], Sources, Targets]:
    """Return ``q`` with unused rows moved away, and the sources and targets."""
    nm = field.n_massive
    idx = jnp.arange(q.shape[0])
    # Inactive rows and non-finite tracers are moved to distinct far-away
    # points, so the pair sum and the external force are evaluated there.
    # Neither their forward values nor their gradients (0 * NaN is NaN) can
    # then reach the other rows.
    safe = jnp.zeros_like(q).at[:, 0].set(_SAFE * (1.0 + idx))
    q = jnp.where(_use_rows(field, q)[:, None], q, safe)
    sources = Sources(
        q=q[:nm], m=(field.masses * field.active)[:nm], h=field.softening[:nm]
    )
    targets = Targets(q=q, h=field.softening, self_index=jnp.where(idx < nm, idx, -1))
    return q, sources, targets


def nbody_acceleration(
    field: NBodyField, t: Any, q: Float[Array, "N 3"]
) -> Float[Array, "N 3"]:
    """Return the accelerations (N, 3) of all particles in ``field``.

    This is the backend's pairwise term from the active massive particles plus
    the external term, zeroed for inactive particles. In ``__call__`` the
    velocity of an inactive row passes through unchanged, so it keeps drifting.

    An active tracer with a non-finite position gets a NaN acceleration. That
    NaN, and any NaN cotangent later arriving on its row, stays out of the
    other particles' values and gradients. An active massive particle with a
    non-finite position makes every active particle's acceleration NaN, since
    its force reaches them all; inactive rows still get 0.

    A tracer at a finite position where the external potential's force, or
    its derivative, is not finite (outside a user potential's valid region,
    or exactly at a singular point) gets that non-finite force. Values of the
    other particles are unaffected, but reverse-mode gradients through such a
    step are NaN, because a zero cotangent times a NaN Jacobian is NaN.
    Parked rows are also evaluated far away (`_SAFE`), so a potential that is
    not finite there does the same.

    Traced masses and softening are checked here on the result, so the checks
    run whenever the acceleration is used, even when the inputs themselves
    never reach it (for example with ``n_massive=0``).
    """
    nm = field.n_massive
    # A NaN cotangent on a non-finite tracer stops at the final `where` below.
    use = _use_rows(field, q)
    q, sources, targets = _sources_targets(field, q)
    backend = field.backend
    # G multiplies outside the gradient policy, so StopGrad on m keeps dG and
    # NoGrad's custom_jvp never closes over a traced G.
    a = field._G * backend.accelerations(backend.prepare(sources), targets, G=1.0)  # noqa: SLF001
    a = a - jnp.asarray(gp.gradient(field.external_potential, q, t))
    a = jnp.where(use[:, None], a, jnp.where(field.active[:, None], jnp.nan, 0.0))
    m_ok, tracer_ok, h_ok = input_oks(field.masses, field.softening, nm)
    a = check(a, m_ok, MASS_MSG)
    a = check(a, tracer_ok, tracer_msg(nm))
    return check(a, h_ok, SOFT_MSG)


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
    x = jnp.asarray(u.ustrip(AllowValue, units["length"], x), dtype=float)
    v = jnp.asarray(u.ustrip(AllowValue, units["speed"], v), dtype=float)
    return v, self.acceleration(t, x)
