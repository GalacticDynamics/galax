"""``galax`` dynamics."""
# ruff:noqa: F401

__all__: list[str] = []

import warnings
from collections.abc import Mapping
from dataclasses import KW_ONLY
from functools import partial
from typing import Any, TypeAlias

import diffrax
import equinox as eqx
from jaxtyping import Array, Float

from unxt import Quantity
from xmmutablemap import ImmutableMap

import galax.potential as gp
from .nbacc import (
    AbstractNBodyAcceleration,
    Acceleration,
    DirectNBodyAcceleration,
    MassQ,
    Position,
    Velocity,
)

TimeQ: TypeAlias = Float[Quantity["time"], ""]

PositionQ: TypeAlias = Float[Quantity["length"], "N 3"]
VelocityQ: TypeAlias = Float[Quantity["speed"], "N 3"]


null_potential = gp.NullPotential()


class TotalVectorField(eqx.Module):  # type: ignore[misc]
    """Total vector field for the N-body problem.

    This vector field combines an external acceleration field and the N-body
    gravitational acceleration.

    """

    _: KW_ONLY
    nbody: AbstractNBodyAcceleration = eqx.field(
        default_factory=partial(
            DirectNBodyAcceleration, softening_length=Quantity(0, "m")
        ),
        static=True,
    )
    """N-body component of the vector field."""

    def __call__(
        self,
        t: Float[Array, ""],
        y: tuple[Position, Velocity],
        args: tuple[gp.AbstractPotentialBase, MassQ],
    ) -> tuple[Velocity, Acceleration]:
        potential, masses = args
        units = potential.units
        q, p = y
        q = Quantity(q, units["length"])

        # External acceleration
        a_ext = -potential._gradient(  # noqa: SLF001
            q, Quantity(t, units["time"])
        )

        # N-body acceleration
        a_nbody = self.nbody(t, q, masses, (potential.constants["G"],))

        # Total acceleration
        a = a_ext + a_nbody

        return (p, a.to_units_value(units["acceleration"]))


class NBodySimulator(eqx.Module):  # type: ignore[misc]
    _: KW_ONLY
    vector_field: TotalVectorField = eqx.field(
        default_factory=TotalVectorField, static=True
    )
    Solver: type[diffrax.AbstractSolver] = eqx.field(
        default=diffrax.Dopri8, static=True
    )
    stepsize_controller: diffrax.AbstractStepSizeController = eqx.field(
        default=diffrax.PIDController(rtol=1e-7, atol=1e-9), static=True
    )
    diffeq_kw: Mapping[str, Any] = eqx.field(
        default=(("max_steps", None),),
        static=True,
        converter=ImmutableMap,
    )
    solver_kw: Mapping[str, Any] = eqx.field(
        default=(), static=True, converter=ImmutableMap
    )

    def __check_init__(self) -> None:
        # Check if the stepsize controller has a minimum stepsize
        if (
            isinstance(self.stepsize_controller, diffrax.PIDController)
            and self.stepsize_controller.dtmin is None
        ):
            warnings.warn(
                "adaptive stepsize with no minimum stepsize can be VERY slow.",
                stacklevel=2,
            )

    def __call__(
        self,
        ic: tuple[PositionQ, VelocityQ],
        mass: MassQ,
        t0: TimeQ,
        t1: TimeQ,
        /,
        *,
        external_potential: gp.AbstractPotentialBase = null_potential,
        snapshot_times: Float[Quantity["time"], "times"] | None = None,
        show_progress: bool | diffrax.TqdmProgressMeter = True,
    ) -> diffrax.Solution:
        units = external_potential.units
        # Save times.
        saveat_snaps = (
            {
                "snapshots": diffrax.SubSaveAt(
                    ts=snapshot_times.to_units_value(units["time"])
                )
            }
            if snapshot_times is not None
            else {}
        )
        saveat = diffrax.SaveAt(subs={"end": diffrax.SubSaveAt(t1=True)} | saveat_snaps)

        # Set up progress meter.
        if isinstance(show_progress, diffrax.TqdmProgressMeter):
            progress_meter = show_progress
        elif show_progress:
            progress_meter = diffrax.TqdmProgressMeter(refresh_steps=100)
        else:
            progress_meter = diffrax.NoProgressMeter()

        y0 = (
            ic[0].to_units_value(units["length"]),
            ic[1].to_units_value(units["speed"]),
        )

        # TODO: enable diffeqsolve to work on Quantity objects.
        return diffrax.diffeqsolve(
            diffrax.ODETerm(self.vector_field),
            self.Solver(**self.solver_kw),
            t0=t0.to_units_value(units["time"]),
            t1=t1.to_units_value(units["time"]),
            dt0=None,
            y0=y0,
            saveat=saveat,
            stepsize_controller=self.stepsize_controller,
            args=(external_potential, mass),
            progress_meter=progress_meter,
            **self.diffeq_kw,
        )
