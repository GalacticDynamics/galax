#!/usr/bin/env python
"""Benchmark `galax.nbody`'s diffrax leapfrog against a plain `lax.scan` loop.

Both paths take the same kick-drift-kick steps with one call to
`NBodyField.acceleration` per step: `advance` through `diffrax.diffeqsolve`
with `galax.fields.Leapfrog`, and a reference `jax.lax.scan` that carries
``(q, p, a)``. For each case the script checks the two agree, then reports µs
per step for the forward run and for ``jax.grad`` of ``sum(q_final**2)`` with
respect to ``q0``, and the XLA temp memory of each compiled function.
The gradient rows compare plain reverse mode through the scan, which stores
every step, with diffrax's checkpointed adjoint, so their time and memory
differ by design.

Timing is interleaved. After one warm-up call of each path, the script calls
scan and diffrax alternately for 21 rounds (7 at N = 1000), so both paths see
the same machine load. It divides each call's time by the number of steps and
reports the minimum over the rounds, since load only ever makes a call slower.
The gate uses the difference of the two minimums. The median is printed as an
extra column for information. On a machine whose load average swings widely,
timing one path in a block and then the other made the gate flip between runs.

Both paths are compiled with the field's arrays as runtime arguments, as in
`evolve`. If the scan closed over the field instead, XLA would fold the
masses and softening into its kernel as constants and the comparison would
not be fair.

Exits 1 if the forward diffrax overhead is over 1 µs per step at N = 32
single. Run on a quiet machine, by hand (not part of CI):

    uv run --no-sync python scripts/benchmark_nbody_leapfrog.py
"""

import statistics
import sys
import time

from collections.abc import Callable
from typing import Any

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)  # noqa: FBT003

import diffrax as dfx  # noqa: E402
import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import unxt as u  # noqa: E402

import galax.nbody as gn  # noqa: E402
import galax.potential as gp  # noqa: E402
from galax.nbody._src.evolve import advance  # noqa: E402

Array = jax.Array

DT = 1e-3  # Myr
GATE_US = 1.0
ROUNDS = 21
ROUNDS_LARGE = 7  # for N = 1000, where one call is slow
# (name, N, number of vmapped systems or None, n_steps)
CASES = [
    ("N=32 single", 32, None, 2000),
    ("N=32 vmap 256", 32, 256, 500),
    ("N=1000 single", 1000, None, 200),
]


def scan_leapfrog(
    field: gn.NBodyField, q: Array, p: Array, t0: Array, n_steps: int, dt: Array
) -> tuple[Array, Array]:
    """Return ``(q, p)`` after ``n_steps`` KDK steps, one force call per step."""

    def step(carry: tuple[Array, Array, Array], k: Array) -> tuple[Any, None]:
        q, p, a = carry
        p = p + 0.5 * dt * a
        q = q + dt * p
        a = field.acceleration(t0 + (k + 1) * dt, q)
        return (q, p + 0.5 * dt * a, a), None

    a = field.acceleration(t0, q)
    (q, p, _), _ = jax.lax.scan(step, (q, p, a), jnp.arange(n_steps))
    return q, p


def make_paths(
    static: Any, n_steps: int, n_sys: int | None
) -> dict[str, tuple[Callable, Callable]]:
    """Return the forward and gradient functions of each path, by name.

    Each takes ``(arrays, q, p)``, where ``arrays`` and ``static`` are the two
    halves of the field from `equinox.partition`.
    """
    t0, dt = jnp.asarray(0.0), jnp.asarray(DT)
    adjoint = dfx.RecursiveCheckpointAdjoint()

    def diffrax_fwd(arrays: Any, q: Array, p: Array) -> tuple[Array, Array]:
        field = eqx.combine(arrays, static)
        return advance(field, (q, p), t0 + dt * jnp.arange(n_steps + 1), adjoint)

    def scan_fwd(arrays: Any, q: Array, p: Array) -> tuple[Array, Array]:
        field = eqx.combine(arrays, static)
        return scan_leapfrog(field, q, p, t0, n_steps, dt)

    paths = {}
    for name, fwd in (("scan", scan_fwd), ("diffrax", diffrax_fwd)):
        if n_sys is not None:
            fwd = jax.vmap(fwd, in_axes=(None, 0, 0))  # noqa: PLW2901

        def loss(arrays: Any, q: Array, p: Array, fwd: Callable = fwd) -> Array:
            return jnp.sum(fwd(arrays, q, p)[0] ** 2)

        paths[name] = (fwd, jax.grad(loss, argnums=1))
    return paths


def compile_and_time(
    fs: dict[str, Callable], rounds: int, *args: Any
) -> dict[str, tuple[float, float, int]]:
    """Return ``(min, median, temp bytes)`` per name, seconds per call.

    The functions are called alternately, one call each per round.
    """
    compiled = {k: jax.jit(f).lower(*args).compile() for k, f in fs.items()}
    for c in compiled.values():
        jax.block_until_ready(c(*args))  # warm-up
    times: dict[str, list[float]] = {k: [] for k in fs}
    for _ in range(rounds):
        for k, c in compiled.items():
            start = time.perf_counter()
            jax.block_until_ready(c(*args))
            times[k].append(time.perf_counter() - start)
    return {
        k: (
            min(t),
            statistics.median(t),
            compiled[k].memory_analysis().temp_size_in_bytes,
        )
        for k, t in times.items()
    }


def initial_conditions(n: int, n_sys: int | None, seed: int = 0) -> tuple[Array, Array]:
    """Return random ``(q, p)``, (N, 3) or (n_sys, N, 3), in galactic units."""
    rng = np.random.default_rng(seed)
    shape = (n, 3) if n_sys is None else (n_sys, n, 3)
    q = rng.normal(0, 0.01, shape)  # kpc
    p = rng.normal(0, 5.0, shape) * 1e-3  # kpc/Myr, about 5 km/s
    return jnp.asarray(q), jnp.asarray(p)


def main() -> int:
    """Run the cases, print the table and return the exit code."""
    rows = []
    gate_overhead = None
    for name, n, n_sys, n_steps in CASES:
        field = gn.NBodyField(
            masses=u.Q(np.full(n, 1e5 / n), "Msun"),
            eps=u.Q(5e-4, "kpc"),
            external_potential=gp.NullPotential(units="galactic"),
        )
        q0, p0 = initial_conditions(n, n_sys)
        arrays, static = eqx.partition(field, eqx.is_array)
        paths = make_paths(static, n_steps, n_sys)
        args = (arrays, q0, p0)

        # Same work: both paths must agree to round-off.
        (qs, ps), (qd, pd) = (jax.jit(paths[k][0])(*args) for k in ("scan", "diffrax"))
        gs, gd = (jax.jit(paths[k][1])(*args) for k in ("scan", "diffrax"))
        for a, b, what in ((qs, qd, "q"), (ps, pd, "p"), (gs, gd, "grad")):
            if not np.allclose(a, b, rtol=1e-10, atol=1e-12):
                err = float(jnp.max(jnp.abs(a - b)))
                print(f"{name}: scan and diffrax {what} differ, max {err:.3g}")
                return 2

        rounds = ROUNDS_LARGE if n >= 1000 else ROUNDS
        scale = 1e6 / n_steps
        for i, kind in enumerate(("fwd", "grad")):
            r = compile_and_time({k: v[i] for k, v in paths.items()}, rounds, *args)
            (min_s, med_s, mem_s), (min_d, med_d, mem_d) = r["scan"], r["diffrax"]
            rows.append(
                (
                    *(name, kind, min_s * scale, min_d * scale),
                    (min_d - min_s) * scale,
                    *(med_s * scale, med_d * scale, mem_s, mem_d),
                )
            )
            if name == "N=32 single" and kind == "fwd":
                gate_overhead = (min_d - min_s) * scale

    head = ("case", "pass", "scan min us/step", "diffrax min us/step", "diff us/step")
    head += ("scan med us/step", "diffrax med us/step", "scan temp B", "diffrax temp B")
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for case, kind, *us, mem_s, mem_d in rows:
        cells = (case, kind, *(f"{x:.3f}" for x in us[:2]), f"{us[2]:+.3f}")
        cells += tuple(f"{x:.3f}" for x in us[3:])
        print("| " + " | ".join((*cells, str(mem_s), str(mem_d))) + " |")

    ok = gate_overhead <= GATE_US
    verdict = "PASS" if ok else "FAIL"
    print(
        f"\ngate: forward diffrax overhead at N=32 single is "
        f"{gate_overhead:+.3f} us/step (limit {GATE_US}): {verdict}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
