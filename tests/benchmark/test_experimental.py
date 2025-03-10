"""Benchmark tests for quaxed functions on quantities."""

from collections.abc import Callable
from typing import Any, TypeAlias, TypedDict
from typing_extensions import Unpack

import jax
import jax.random as jr
import pytest
from jax._src.stages import Compiled
from plum import convert

import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp
from galax.utils import loop_strategies

# =============================================================================
# Tools for crafting a benchmark suite


class Arguments:
    __slots__ = ("args", "kwargs")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args: tuple[Any, ...] = args
        self.kwargs: dict[str, Any] = kwargs


Func: TypeAlias = Callable[..., Any]
FuncAndArgs: TypeAlias = tuple[Func, Arguments]
ID: TypeAlias = str

JITOpts = dict[str, Any] | None
ProcessFn: TypeAlias = Callable[[Func, JITOpts, Arguments], FuncAndArgs]


# implements ProcessFn
def process_func(
    func: Func, jit_kwargs: JITOpts, args: Arguments
) -> tuple[Compiled, Arguments]:
    """JIT (with options) and compile the function."""
    jit_kw = jit_kwargs or {}
    return jax.jit(func, **jit_kw), args


class ParameterizationKWArgs(TypedDict):
    """Keyword arguments for a pytest parameterization."""

    argvalues: list[FuncAndArgs]
    ids: list[str]


def process_pytest_paramatrization(
    process_fn: ProcessFn,
    arg_id_values: list[
        tuple[Func, str | None, JITOpts, Unpack[tuple[Arguments, ...]]]
    ],
) -> ParameterizationKWArgs:
    """Process the argvalues."""
    # Get the ID for each parameterization
    get_types = lambda argobj: tuple(str(type(a).__name__) for a in argobj.args)
    ids: list[str] = []
    processed_argvalues: list[tuple[Compiled, Arguments]] = []

    for func, ID, jit_kw, *many_argobjs in arg_id_values:
        for argobj in many_argobjs:
            ids.append(f"{func.__name__}-{ID or '*'}-{get_types(argobj)}")
            processed_argvalues.append(process_fn(func, jit_kw, argobj))

    # Process the argvalues and return the parameterization, with IDs
    return {"argvalues": processed_argvalues, "ids": ids}


# =============================================================================
# Setup

pot = gp.HernquistPotential(1e12, 10, units="galactic")
field = gd.HamiltonianField(pot)

w0 = gc.PhaseSpaceCoordinate(
    q=u.Quantity([15.0, 0.0, 0.0], "kpc"),
    p=u.Quantity([0.0, 220.0, 0.0], "km/s"),
    t=u.Quantity(0.0, "Gyr"),
)
qp0 = (
    convert(w0.q, u.Quantity).ustrip(pot.units["length"]),
    convert(w0.p, u.Quantity).ustrip(pot.units["speed"]),
)
t0 = w0.t.ustrip(pot.units["time"])

ts = jnp.linspace(0.0, 100.0, 1_000)  # [Myr]

# Stream simulation
stream_simulator = gd.experimental.stream.StreamSimulator()
release_times = jnp.linspace(-4_000, -150, 2_000)
t1 = 0
Msat = 1e5  # [Msun]
stream_ics = stream_simulator.init(
    pot, qp0, t0, release_times=release_times, Msat=Msat, key=jr.key(0)
)


static_argnums = {"static_argnums": (0,)}
static_argnames = {"static_argnames": ("solver", "solver_kwargs", "dense")}

funcs_id_and_args: list[tuple[Func, ID, JITOpts, Unpack[tuple[Arguments, ...]]]] = [
    # ================================================
    # Orbit integration
    (
        gd.experimental.integrate.integrate_orbit,
        "scalar",
        static_argnames,
        Arguments(pot, qp0, ts[0], ts[-1], saveat=ts),
    ),
    (
        gd.experimental.integrate.integrate_orbit,
        "field",
        static_argnames,
        Arguments(field, qp0, ts[0], ts[-1], saveat=ts),
    ),
    (
        gd.experimental.integrate.integrate_orbit,
        "scalar-NoLoop",
        static_argnums | static_argnames,
        Arguments(loop_strategies.NoLoop, pot, qp0, ts[0], ts[-1], saveat=ts),
    ),
    (
        gd.experimental.integrate.integrate_orbit,
        "scalar-Scan",
        static_argnums | static_argnames,
        Arguments(loop_strategies.Scan, pot, qp0, ts[0], ts[-1], saveat=ts),
    ),
    (
        gd.experimental.integrate.integrate_orbit,
        "field-Scan",
        static_argnums | static_argnames,
        Arguments(loop_strategies.Scan, field, qp0, ts[0], ts[-1], saveat=ts),
    ),
    (
        gd.experimental.integrate.integrate_orbit,
        "scalar-VMap",
        static_argnums | static_argnames,
        Arguments(loop_strategies.VMap, pot, qp0, ts[0], ts[-1], saveat=ts),
    ),
    (
        gd.experimental.integrate.integrate_orbit,
        "field-VMap",
        static_argnums | static_argnames,
        Arguments(loop_strategies.VMap, field, qp0, ts[0], ts[-1], saveat=ts),
    ),
    # ================================================
    # DF
    (
        gd.experimental.df.Fardal2015DF().sample,
        "Fardal15",
        None,
        Arguments(jr.key(0), pot, 0, qp0[0], qp0[1], Msat),
    ),
    # ================================================
    # Streams
    (
        stream_simulator.init,
        "defaults",
        {"static_argnames": ("solver", "solver_kwargs")},
        Arguments(pot, qp0, t0, release_times=release_times, Msat=Msat, key=jr.key(0)),
    ),
    (
        stream_simulator.init,
        "kinematic_df",
        {"static_argnames": ("solver", "solver_kwargs")},
        Arguments(
            pot,
            qp0,
            t0,
            release_times=release_times,
            Msat=Msat,
            key=jr.key(0),
            kinematic_df=gd.experimental.df.Fardal2015DF(),
        ),
    ),
    (
        stream_simulator.run,
        "no-flag",
        static_argnames,
        Arguments(pot, stream_ics, t1=t0),
    ),
    (
        stream_simulator.run,
        "determine",
        static_argnums | static_argnames,
        Arguments(loop_strategies.Determine, pot, stream_ics, t1=t0),
    ),
    (
        stream_simulator.run,
        "scan",
        static_argnums | static_argnames,
        Arguments(loop_strategies.Scan, pot, stream_ics, t1=t0),
    ),
    (
        stream_simulator.run,
        "vmap",
        static_argnums | static_argnames,
        Arguments(loop_strategies.VMap, pot, stream_ics, t1=t0),
    ),
]


# =============================================================================


@pytest.mark.parametrize(
    ("func", "argobj"),
    **process_pytest_paramatrization(process_func, funcs_id_and_args),
)
@pytest.mark.benchmark(group="quaxed", max_time=1.0, warmup=False)
def test_jit_compile(func, argobj):
    """Test the speed of jitting a function."""
    _ = func.lower(*argobj.args, **argobj.kwargs).compile()


@pytest.mark.parametrize(
    ("func", "argobj"),
    **process_pytest_paramatrization(process_func, funcs_id_and_args),
)
@pytest.mark.benchmark(
    group="galax.dynamics",
    max_time=1.0,  # NOTE: max_time is ignored
    warmup=True,
)
def test_execute(func, argobj):
    """Test the speed of calling the function."""
    _ = jax.block_until_ready(func(*argobj.args, **argobj.kwargs))
