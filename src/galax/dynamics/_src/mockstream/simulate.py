"""Simulate a mockstream.

This is private API.

"""

__all__ = ["simulate_stream", "StreamSimulator"]

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax.random as jr
from jaxtyping import PRNGKeyArray
from plum import convert

import quaxed.numpy as jnp
import unxt as u
from dataclassish.converters import Unless

import galax.coordinates as gc
import galax.potential as gp
from .arm import MockStreamArm
from .core import MockStream
from .df_base import AbstractStreamDF
from .df_fardal15 import Fardal15StreamDF
from galax.dynamics._src.cluster.fields import AbstractMassRateField
from galax.dynamics._src.cluster.sample import ReleaseTimeSampler
from galax.dynamics._src.cluster.solver import MassSolver
from galax.dynamics._src.dynamics.field_base import AbstractDynamicsField
from galax.dynamics._src.dynamics.field_hamiltonian import HamiltonianField
from galax.dynamics._src.dynamics.solver import DynamicsSolver
from galax.dynamics._src.orbit import Orbit, compute_orbit

default_dynamics_solver = DynamicsSolver()
default_dynamics_release_model = Fardal15StreamDF()

# default_
# converter_dynamics_solver = lambda x:
converter_dynamics_field = Unless(AbstractDynamicsField, HamiltonianField)


@dataclass
class StreamSimulator:
    """Simulate a mock stellar stream.

    Notes
    -----
    - Support solving for the mass history of the progenitor using the
      mass_history_solver, in which case a mass_release_model is required.
    - Support pre-specified stripping times, in which case the
      mass_history_solver is not required. The mass loss model can be given and
      will be used to change the mass of the cluster. Note that this will not be
      self-consistent in the mass history.

    """

    #: Solver for the dynamics of the progenitor and stream stars.
    dynamics_solver: DynamicsSolver = default_dynamics_solver

    #: The position & velocity release model for the stream stars.
    dynamics_release_model: AbstractStreamDF = default_dynamics_release_model

    #: Solver for the mass history of the progenitor. If `None` (default), the
    #: mass history is not solved. Instead the stream simulator requires an
    #: array of stripping times.
    mass_history_solver: MassSolver | None = None

    # TODO: figure out how to fold this into the mass history solver
    #: The mass field to use for the mass history. If `None` (default), the
    #: progenitor doee not lose mass. An error will be raised if the
    #: `mass_history_solver` is not None.
    mass_release_model: AbstractMassRateField | None = None

    def __check_init__(self) -> None:
        if self.mass_history_solver is not None and self.mass_release_model is None:
            msg = "`mass_history_solver` is not None, so a `mass_release_model` must be provided."  # noqa: E501
            raise ValueError(msg)

    def _call_nomasshistory(
        self,
        dynamics_field: AbstractDynamicsField,
        w0: gc.PhaseSpaceCoordinate,
        M0: u.AbstractQuantity,  # noqa: ARG002
        t0: u.AbstractQuantity,  # noqa: ARG002
        t1: u.AbstractQuantity,  # noqa: ARG002
        /,
        stripping_times: u.AbstractQuantity | None = None,
        **_: Any,
    ) -> tuple[Orbit, MockStream]:
        _ = compute_orbit(
            dynamics_field,
            w0,
            stripping_times,
            dense=False,
            solver=self.dynamics_solver,
        )

        raise NotImplementedError

    def _call_masshistory(
        self,
        dynamics_field: AbstractDynamicsField,
        w0: gc.PhaseSpaceCoordinate,
        M0: u.AbstractQuantity,
        t0: u.AbstractQuantity,
        t1: u.AbstractQuantity,
        /,
        key: PRNGKeyArray,
        n_stars: int,
        mass_params: dict[str, Any] = {},  # noqa: B006
        **_: Any,
    ) -> tuple[Orbit, MockStream]:
        """Simulate a mock stellar stream with mass history."""
        mass_solver = eqx.error_if(
            self.mass_history_solver,
            self.mass_history_solver is None,
            "`mass_history_solver` is None",
        )
        mass_loss_rate = eqx.error_if(
            self.mass_release_model,
            self.mass_release_model is None,
            "`mass_release_model` is None",
        )

        # Step 1) Solve the progenitor's orbit
        ts = jnp.linspace(t0, t1, 2)
        orbit = compute_orbit(
            dynamics_field, w0, ts, dense=True, solver=self.dynamics_solver
        )

        # Step 2) Solve the mass history of the progenitor
        mass_params = mass_params.copy()
        mass_params["orbit"] = orbit
        mass_history = mass_solver.solve(
            mass_loss_rate,
            M0,
            t0,
            t1,
            args=mass_params,
            dense=True,
            vectorize_interpolation=True,
        )

        # Step 3) Sample release model.
        # A) Release time from mass history
        release_time_sampler = ReleaseTimeSampler(
            mass_loss_rate, mass_history, orbit.potential.units
        )
        key, subkey = jr.split(key)
        release_times = release_time_sampler.sample(
            subkey, t0, t1, n_stars=n_stars, mass_params=mass_params
        )

        # B) star positions
        key, subkey = jr.split(key)
        o_at_t = orbit(release_times)
        x_lead, v_lead, x_trail, v_trail = default_dynamics_release_model.sample(
            subkey,
            orbit.potential,
            convert(o_at_t.q, u.Quantity),
            convert(o_at_t.p, u.Quantity),
            M0,
            release_times,
        )
        wlead = gc.PhaseSpaceCoordinate(
            q=x_lead, p=v_lead, t=release_times, frame=w0.frame
        )
        wtrail = gc.PhaseSpaceCoordinate(
            q=x_trail, p=v_trail, t=release_times, frame=w0.frame
        )

        # Step 4) Solve the orbits of the stream stars
        # TODO: enable dense model
        soln_lead = self.dynamics_solver.solve(
            dynamics_field,
            wlead,
            t1,
            dense=False,
            vectorize_interpolation=True,
        )
        stream_lead = MockStreamArm.from_(
            soln_lead, release_time=release_times, frame=wlead.frame
        )

        soln_trail = self.dynamics_solver.solve(
            dynamics_field,
            wtrail,
            t1,
            dense=False,
            vectorize_interpolation=True,
        )
        stream_trail = MockStreamArm.from_(
            soln_trail, release_time=release_times, frame=wtrail.frame
        )

        return orbit, MockStream(lead=stream_lead, trail=stream_trail)

    def __call__(
        self,
        dynamics_field: AbstractDynamicsField,
        w0: gc.PhaseSpaceCoordinate,
        M0: u.AbstractQuantity,
        t0: u.AbstractQuantity,
        t1: u.AbstractQuantity,
        /,
        stripping_times: u.AbstractQuantity | None = None,
        **kw: Any,
    ) -> Any:
        # Step 0) Prep the inputs
        # Sort the times
        t0 = jnp.minimum(t0, t1)
        t1 = jnp.maximum(t0, t1)

        # Step 1) Solve the progenitor's orbit
        if self.mass_history_solver is None:
            t_strip = eqx.error_if(
                stripping_times, stripping_times is None, "`stripping_times` is None"
            )
            out = self._call_nomasshistory(
                dynamics_field,
                w0,
                M0,
                t0,
                t1,
                stripping_times=t_strip,
            )

        else:
            out = self._call_masshistory(
                dynamics_field,
                w0,
                M0,
                t0,
                t1,
                key=kw["key"],
                n_stars=kw["n_stars"],
                mass_params=kw["mass_params"],
            )

        return out


# ===================================================================


def simulate_stream(
    key: PRNGKeyArray,
    dynamics_field: AbstractDynamicsField | gp.AbstractPotential,
    w0: gc.PhaseSpaceCoordinate,
    t0: u.AbstractQuantity,  # start time of the stream, e.g. -3 Gyr
    t1: u.AbstractQuantity,  # end time of the stream, e.g. 0 Gyr
    n_stream_stars: int,  # number of stream stars
    Mc0: u.AbstractQuantity,  # initial mass of the progenitor
    mass_params: dict[str, Any],
    *,
    dynamics_solver: DynamicsSolver | None = None,
    mass_history_solver: MassSolver | None = None,
    mass_field: AbstractMassRateField,  # TODO: or callable
) -> Any:
    """Simulate a mock stellar stream."""
    # Step 0) Prep the inputs
    # Sort the times
    t0 = jnp.minimum(t0, t1)
    t1 = jnp.maximum(t0, t1)
    dynamics_solver = DynamicsSolver() if dynamics_solver is None else dynamics_solver
    dynamics_field = converter_dynamics_field(dynamics_field)
    mass_solver = MassSolver() if mass_history_solver is None else mass_history_solver

    # Step 1) Solve the progenitor's orbit
    orbit = compute_orbit(
        dynamics_field, w0, jnp.linspace(t0, t1, 2), dense=True, solver=dynamics_solver
    )

    # Step 2) Solve the mass history of the progenitor
    mass_params = mass_params.copy()
    mass_params["orbit"] = orbit
    mass_history = mass_solver.solve(
        mass_field,
        Mc0,
        t0,
        t1,
        args=mass_params,
        dense=True,
        vectorize_interpolation=True,
    )

    # Step 3) Sample release model.
    # A) Release time from mass history
    release_time_sampler = ReleaseTimeSampler(
        mass_field, mass_history, orbit.potential.units
    )
    key, subkey = jr.split(key)
    release_times = release_time_sampler.sample(
        subkey, t0, t1, n_stars=n_stream_stars, mass_params=mass_params
    )

    # B) star positions
    key, subkey = jr.split(key)
    o_at_t = orbit(release_times)
    x_lead, v_lead, x_trail, v_trail = default_dynamics_release_model.sample(
        subkey,
        orbit.potential,
        convert(o_at_t.q, u.Quantity),
        convert(o_at_t.p, u.Quantity),
        Mc0,
        release_times,
    )
    wlead = gc.PhaseSpaceCoordinate(q=x_lead, p=v_lead, t=release_times, frame=w0.frame)
    wtrail = gc.PhaseSpaceCoordinate(
        q=x_trail, p=v_trail, t=release_times, frame=w0.frame
    )

    # Step 4) Solve the orbits of the stream stars
    # TODO: enable dense model
    soln_lead = dynamics_solver.solve(
        dynamics_field,
        wlead,
        t1,
        dense=False,
        vectorize_interpolation=True,
    )
    orbit_lead = Orbit.from_(
        soln_lead, frame=wlead.frame, potential=dynamics_field.potential
    )

    soln_trail = dynamics_solver.solve(
        dynamics_field,
        wtrail,
        t1,
        dense=False,
        vectorize_interpolation=True,
    )
    orbit_trail = Orbit.from_(
        soln_trail, frame=wtrail.frame, potential=dynamics_field.potential
    )

    return orbit_lead, orbit_trail, release_times
