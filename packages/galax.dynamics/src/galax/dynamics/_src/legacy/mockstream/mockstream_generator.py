"""Generator for mock streams."""

__all__ = ["MockStreamGenerator"]

import functools as ft
import warnings
from dataclasses import KW_ONLY

from jaxtyping import PRNGKeyArray
from typing import Any, TypeAlias, cast, final

import diffrax as dfx
import equinox as eqx
import jax
import jax.extend as jex

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import Quantity as FastQ

import galax.coordinates as gc
import galax.dynamics.custom_types as gt
import galax.fields as gf
from .df import AbstractStreamDF, ChenStreamDF, ProgenitorMassCallable
from galax.dynamics._src.legacy.funcs import default_integrator, evaluate_orbit
from galax.dynamics._src.legacy.integrator import Integrator
from galax.dynamics._src.mockstream.arm import MockStreamArm
from galax.dynamics._src.mockstream.core import MockStream
from galax.dynamics._src.orbit import HamiltonianField, Orbit
from galax.orbits import AbstractOrbitField
from galax.potential import AbstractPotential
from galax.potential.utils import cond_reverse

Carry: TypeAlias = tuple[gt.IntSz0, gt.SzN, gt.SzN]


@final
class _ProgenitorField(AbstractOrbitField):
    """A particle (row 0) and the progenitor (row 1), integrated together.

    The particle moves in the host plus the progenitor's potential centred on
    row 1; the progenitor moves in the host alone.
    """

    host: AbstractPotential
    progenitor: AbstractPotential

    def __check_init__(self) -> None:
        # The progenitor's forces are added to the host's as raw arrays. Checked
        # here, built in each `run`, so a generator changed by `eqx.tree_at` is
        # checked too.
        if self.progenitor.units != self.host.units:
            msg = "progenitor_potential must have the same unit system as potential"
            raise ValueError(msg)

    @property  # type: ignore[misc]
    def units(self) -> u.AbstractUnitSystem:
        return cast("u.AbstractUnitSystem", self.host.units)

    def dx_dt(self, t: gt.Sz0, v: gt.SzN, args: object, /) -> gt.SzN:  # noqa: ARG002
        return v

    def dv_dt(self, t: gt.Sz0, x: gt.SzN, args: object, /) -> gt.SzN:  # noqa: ARG002
        a: gt.SzN = -self.host._gradient(x, t)  # noqa: SLF001
        dx = x[..., 0, :] - x[..., 1, :]
        a = a.at[..., 0, :].add(-self.progenitor._gradient(dx, t))  # noqa: SLF001
        return a


@AbstractOrbitField.terms.dispatch
def terms(
    self: _ProgenitorField, _: dfx.SemiImplicitEuler | gf.Leapfrog, /
) -> tuple[dfx.ODETerm, dfx.ODETerm]:
    return (dfx.ODETerm(self.dx_dt), dfx.ODETerm(self.dv_dt))


@AbstractOrbitField.__call__.dispatch
def call(
    self: _ProgenitorField, t: Any, qp: tuple[Any, Any], args: object = None, /
) -> tuple[gt.SzN, gt.SzN]:
    return self.dx_dt(t, qp[1], args), self.dv_dt(t, qp[0], args)


@final
class MockStreamGenerator(eqx.Module):
    """Generate a mock stellar stream in the specified external potential.

    `jax.jvp` works through `run` with the default integrators. For `jax.grad`
    (reverse mode) give both integrators an `~galax.dynamics.OrbitSolver` with
    ``adjoint=diffrax.RecursiveCheckpointAdjoint()`` and an integer
    ``"max_steps"`` in ``diffeq_kw``. The default `diffrax.ForwardMode` adjoint
    solves in a `jax.lax.while_loop`, which reverse mode can't differentiate;
    in turn the reverse-mode adjoint can't be forward-differentiated. By
    default it keeps about ``sqrt(2 max_steps)`` states per solve and
    recomputes the steps between them on the backward pass; its
    ``checkpoints`` argument trades memory for time. `diffrax.DirectAdjoint`
    supports both modes, e.g. for `jax.hessian`, at a higher cost.
    """

    df: AbstractStreamDF
    """Distribution function for generating mock streams.

    E.g. ``galax.dynamics.mockstream.FardalStreamDF``.
    """

    potential: AbstractPotential
    """Potential in which the progenitor orbits and creates a stream."""

    _: KW_ONLY
    progenitor_potential: AbstractPotential | None = None
    """Potential of the progenitor, centred on the origin, or `None`.

    If given, the stream particles move in `potential` plus this potential
    centred on the progenitor as it orbits (the progenitor itself orbits in
    `potential` alone). This is the default model of Chen et al. 2024
    (arXiv:2408.01496) and gala's ``progenitor_potential``. For Chen+24 use a
    `~galax.potential.PlummerPotential` of the progenitor's mass with
    ``r_s`` = 4 pc, as in the authors' notebooks; the paper uses a point mass
    and finds that 0-4 pc doesn't noticeably change the streams. The
    potential's mass is its own: the ``prog_mass`` given to `run` only sets the
    release distribution.

    Each particle is integrated together with its own copy of the progenitor,
    from the progenitor's state at the particle's release time, as gala does.
    """

    progenitor_integrator: Integrator = eqx.field(
        default=default_integrator, static=True
    )
    """Integrator for the progenitor orbit."""

    stream_integrator: Integrator = eqx.field(default=default_integrator, static=True)
    """Integrator for the stream."""

    @property
    def units(self) -> u.AbstractUnitSystem:
        """Units of the potential."""
        return cast("u.AbstractUnitSystem", self.potential.units)

    # ==========================================================================

    def _progenitor_trajectory(
        self, w0: gc.AbstractPhaseSpaceObject, ts: gt.QuSzTime
    ) -> Orbit:
        """Integrate the progenitor orbit."""
        return cast(
            "Orbit",
            evaluate_orbit(
                self.potential, w0, ts, integrator=self.progenitor_integrator
            ),
        )

    def _integrate_to(
        self,
        w0: gt.BtSz6,
        t0: gt.QuSz0,
        t1: gt.QuSz0,
        prog_w0: gt.Sz6 | None = None,
        /,
    ) -> tuple[gt.Sz6, gt.Sz6]:
        """Integrate a leading and a trailing particle ``(2, 6)`` from t0 to t1.

        Each gets its own solve, so an adaptive solver picks its steps per particle.
        With ``prog_w0``, the progenitor's state at t0, each particle is solved
        together with the progenitor in `progenitor_potential`'s field.
        """
        # `units` is an `eqx.AbstractVar` on the base class, which mypy reads as
        # an init field; on these fields it is a property.
        field: AbstractOrbitField
        if prog_w0 is None:
            field = HamiltonianField(self.potential)  # type: ignore[call-arg]
        else:
            assert self.progenitor_potential is not None  # noqa: S101
            field = _ProgenitorField(self.potential, self.progenitor_potential)  # type: ignore[call-arg]
        # `SaveAt(t1=True)` is ~2.4x slower here on the adaptive scan path (an
        # XLA whole-program effect), so save at the time `t1` instead.
        saveat = jnp.asarray([t1])

        def final_state(w: gt.Sz6) -> gt.Sz6:
            y = w if prog_w0 is None else jnp.stack([w, prog_w0])
            out = self.stream_integrator(field, y, t0, t1, saveat=saveat)
            # (1, 6), or (2, 1, 6) with the progenitor as row 1.
            w1: gt.Sz6 = out.w(units=self.units).reshape(-1, 6)[0]
            return w1

        w = jax.vmap(final_state)(w0)
        return w[0], w[1]

    # ==========================================================================

    @ft.partial(jax.jit)
    def _run_scan(  # TODO: output shape depends on the input shape
        self,
        ts: gt.QuSzTime,
        mock0_lead: MockStreamArm,
        mock0_trail: MockStreamArm,
        prog_ws: gt.SzN | None,
        /,
    ) -> tuple[gt.BtSz6, gt.BtSz6]:
        """Generate stellar stream by scanning over the release model/integration.

        Better for CPU usage.
        """
        w0_lead = mock0_lead.w(units=self.units)
        w0_trail = mock0_trail.w(units=self.units)
        t_f = ts[-1] + u.Q(1e-3, "Myr")  # TODO: not bump in the final time.

        def one_pt_intg(
            carry: Carry, _: gt.IntSz0
        ) -> tuple[Carry, tuple[gt.SzN, gt.SzN]]:
            """Integrate one point along the stream.

            Parameters
            ----------
            carry : tuple[int, SzN, SzN]
                Initial state of the particle at index `idx`.
            idx : int
                Index of the current point.
            """
            i, w0_l_i, w0_t_i = carry
            w_l, w_t = self._integrate_to(
                jnp.vstack([w0_l_i, w0_t_i]),
                ts[i],
                t_f,
                None if prog_ws is None else prog_ws[i],
            )
            # Prepare for next iteration
            carry_out = (i + 1, w0_lead[i + 1, :], w0_trail[i + 1, :])
            return carry_out, (w_l, w_t)

        carry_init = (0, w0_lead[0, :], w0_trail[0, :])
        pt_ids = jnp.arange(len(w0_lead))
        lead_arm_w, trail_arm_w = jax.lax.scan(
            one_pt_intg,  # type: ignore[arg-type]
            carry_init,
            pt_ids,
        )[1]

        return lead_arm_w, trail_arm_w

    @ft.partial(jax.jit)
    def _run_vmap(  # TODO: output shape depends on the input shape
        self,
        ts: gt.QuSzTime,
        mock0_lead: MockStreamArm,
        mock0_trail: MockStreamArm,
        prog_ws: gt.SzN | None,
        /,
    ) -> tuple[gt.BtSz6, gt.BtSz6]:
        """Generate stellar stream by vmapping over the release model/integration.

        Better for GPU usage.
        """
        t_f = ts[-1] + u.Q(1e-3, "Myr")  # TODO: not bump in the final time.

        @ft.partial(jax.jit, inline=True)
        def one_pt_intg(
            i: gt.IntSz0, w0_l_i: gt.Sz6, w0_t_i: gt.Sz6
        ) -> tuple[gt.Sz6, gt.Sz6]:
            return self._integrate_to(
                jnp.vstack([w0_l_i, w0_t_i]),
                ts[i],
                t_f,
                None if prog_ws is None else prog_ws[i],
            )

        w0_lead = mock0_lead.w(units=self.units)
        w0_trail = mock0_trail.w(units=self.units)
        pt_ids = jnp.arange(len(w0_lead))
        lead_arm_w, trail_arm_w = jax.vmap(one_pt_intg)(pt_ids, w0_lead, w0_trail)
        return lead_arm_w, trail_arm_w

    @ft.partial(jax.jit, static_argnames=("vmapped",))
    def run(
        self,
        rng: PRNGKeyArray,
        ts: gt.QuSzTime,
        prog_w0: gc.AbstractPhaseSpaceObject | gt.Sz6,
        prog_mass: gt.FloatQuSz0 | ProgenitorMassCallable,
        *,
        vmapped: bool | None = None,
    ) -> tuple[MockStream, gc.PhaseSpaceCoordinate]:
        """Generate mock stellar stream.

        Parameters
        ----------
        rng : :class:`jaxtyping.PRNGKeyArray`
            Random number generator.
        ts : Quantity[float, (time,), "time"]
            Stripping times.
        prog_w0 : PhaseSpaceCoordinate[float, ()]
            Initial conditions of the progenitor.

            The recommended way to pass in the progenitor's initial conditions
            is as a :class:`~galax.coordinates.PhaseSpaceCoordinate` object with
            a set time. This is the most explicit and is guaranteed to have the
            correct units and no surprises about the progenitor.

            .. note::

                If the time is not set, it is assumed to be the first stripping
                time.

            Alternatively, you can pass in a 6-element array of the Cartesian
            phase-space coordinates (x, y, z, vx, vy, vz) in the same units as
            the potential.

        prog_mass : Quantity[float, (), "mass"] | `ProgenitorMassCallable`
            Mass of the progenitor. May be a Quantity or a callable that returns
            the progenitor mass at the given times.

        vmapped : bool | None, optional keyword-only
            Whether to use `jax.vmap` (`True`) or `jax.lax.scan` (`False`) to
            vectorize the integration. ``vmapped=True`` is recommended for GPU
            usage, while ``vmapped=False`` is recommended for CPU usage.  If
            `None` (default), then `jax.vmap` is used on GPU and `jax.lax.scan`
            otherwise.

        Returns
        -------
        mockstream : :class:`galax.dynamcis.MockStreamArm`
            Leading and/or trailing arms of the mock stream.
        prog_o : :class:`galax.coordinates.PhaseSpaceCoordinate`
            The final phase-space(+time) position of the progenitor.
        """
        if self.progenitor_potential is None and isinstance(self.df, ChenStreamDF):
            warnings.warn(
                "ChenStreamDF without a progenitor_potential is the "
                '"no progenitor" variant of the Chen+24 model; pass e.g. '
                "progenitor_potential=PlummerPotential(m_tot=prog_mass, "
                'r_s=u.Q(4, "pc"), units=...) for its default model.',
                RuntimeWarning,
                stacklevel=2,
            )

        # Parse vmapped
        use_vmap = (
            jex.backend.get_backend().platform == "gpu" if vmapped is None else vmapped
        )

        # Ensure w0 is a PhaseSpacePosition
        w0: gc.PhaseSpaceCoordinate
        if isinstance(prog_w0, gc.PhaseSpaceCoordinate):
            w0 = prog_w0
        elif isinstance(prog_w0, gc.PhaseSpacePosition):
            w0 = gc.PhaseSpaceCoordinate(q=prog_w0.q, p=prog_w0.p, t=ts[0])
        else:
            w0 = gc.PhaseSpaceCoordinate(
                q=u.Q(prog_w0[0:3], self.units["length"]),
                p=u.Q(prog_w0[3:6], self.units["speed"]),
                t=u.uconvert(self.potential.units["time"], ts[0]),
            )
        w0 = eqx.error_if(w0, w0.ndim > 0, "prog_w0 must be scalar")
        # TODO: allow for multiple progenitors

        # If the time stepping passed in is negative, assume this means that all
        # of the initial conditions are at *end time*, and we need to reverse
        # them before treating them as initial conditions
        ts = cond_reverse(ts[1] < ts[0], ts)

        # Integrate the progenitor orbit, evaluating at the stripping times
        prog_o = self._progenitor_trajectory(w0, ts)

        # TODO: here sep out lead vs trailing
        # Generate initial conditions from the DF, along the integrated
        # progenitor orbit. The release times are the stripping times.
        mock0 = self.df.sample(rng, self.potential, prog_o, prog_mass)

        # The progenitor's state at each release time, which each particle's
        # copy of the progenitor starts from.
        prog_ws = (
            None if self.progenitor_potential is None else prog_o.w(units=self.units)
        )
        run = self._run_vmap if use_vmap else self._run_scan
        lead_arm_w, trail_arm_w = run(ts, mock0["lead"], mock0["trail"], prog_ws)

        t = jnp.ones_like(ts) * ts.value[-1]  # TODO: ensure this time is correct

        frame = (
            prog_w0.frame
            if isinstance(prog_w0, gc.AbstractPhaseSpaceObject)
            else gc.frames.simulation_frame
        )

        comps = {}
        comps["lead"] = MockStreamArm(
            q=FastQ(lead_arm_w[:, 0:3], self.units["length"]),
            p=FastQ(lead_arm_w[:, 3:6], self.units["speed"]),
            t=t,
            release_time=mock0["lead"].release_time,
            frame=frame,
        )
        comps["trail"] = MockStreamArm(
            q=FastQ(trail_arm_w[:, 0:3], self.units["length"]),
            p=FastQ(trail_arm_w[:, 3:6], self.units["speed"]),
            t=t,
            release_time=mock0["trail"].release_time,
            frame=frame,
        )

        return MockStream(comps), prog_o[-1]
