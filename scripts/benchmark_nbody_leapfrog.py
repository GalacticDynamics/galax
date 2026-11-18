#!/usr/bin/env python
"""Benchmark `galax.nbody`'s diffrax leapfrog against a plain `lax.scan` loop.

Both paths take the same kick-drift-kick steps with one call to
`NBodyField.acceleration` per step: `advance` through `diffrax.diffeqsolve`
with `galax.fields.Leapfrog`, and a reference `jax.lax.scan` that carries
``(q, p, a)``. For each case the script checks the two agree, then reports the
median µs per step over 5 repeats after one warm-up, for the forward run and
for ``jax.grad`` of ``sum(q_final**2)`` with respect to ``q0``, and the XLA
temp memory of each compiled function.

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
import jax.numpy as jnp  # noqa: E402

import unxt as u  # noqa: E402

import galax.nbody as gn  # noqa: E402
import galax.potential as gp  # noqa: E402
from galax.nbody._src.evolve import advance  # noqa: E402

Array = jax.Array

DT = 1e-3  # Myr
GATE_US = 1.0
REPEATS = 5
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
    field: gn.NBodyField, n_steps: int, n_sys: int | None
) -> dict[str, tuple[Callable, Callable]]:
    """Return the forward and gradient functions of each path, by name."""
    t0, dt = jnp.asarray(0.0), jnp.asarray(DT)
    adjoint = dfx.RecursiveCheckpointAdjoint()

    def diffrax_fwd(q: Array, p: Array) -> tuple[Array, Array]:
        return advance(field, (q, p), t0, n_steps, dt, adjoint)

    def scan_fwd(q: Array, p: Array) -> tuple[Array, Array]:
        return scan_leapfrog(field, q, p, t0, n_steps, dt)

    paths = {}
    for name, fwd in (("scan", scan_fwd), ("diffrax", diffrax_fwd)):
        if n_sys is not None:
            fwd = jax.vmap(fwd)  # noqa: PLW2901

        def loss(q: Array, p: Array, fwd: Callable = fwd) -> Array:
            return jnp.sum(fwd(q, p)[0] ** 2)

        paths[name] = (fwd, jax.grad(loss))
    return paths


def compile_and_time(f: Callable, q: Array, p: Array) -> tuple[float, int]:
    """Return the median seconds per call over ``REPEATS`` and the temp bytes."""
    compiled = jax.jit(f).lower(q, p).compile()
    temp = compiled.memory_analysis().temp_size_in_bytes
    jax.block_until_ready(compiled(q, p))  # warm-up
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        jax.block_until_ready(compiled(q, p))
        times.append(time.perf_counter() - start)
    return statistics.median(times), temp


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
        paths = make_paths(field, n_steps, n_sys)

        # Same work: both paths must agree to round-off.
        (qs, ps), (qd, pd) = (jax.jit(paths[k][0])(q0, p0) for k in ("scan", "diffrax"))
        gs, gd = (jax.jit(paths[k][1])(q0, p0) for k in ("scan", "diffrax"))
        for a, b, what in ((qs, qd, "q"), (ps, pd, "p"), (gs, gd, "grad")):
            if not np.allclose(a, b, rtol=1e-10, atol=1e-12):
                err = float(jnp.max(jnp.abs(a - b)))
                print(f"{name}: scan and diffrax {what} differ, max {err:.3g}")
                return 2

        res = {}
        for path, (fwd, grad) in paths.items():
            for kind, f in (("fwd", fwd), ("grad", grad)):
                sec, temp = compile_and_time(f, q0, p0)
                res[path, kind] = (sec / n_steps * 1e6, temp)
        for kind in ("fwd", "grad"):
            (us_s, mem_s), (us_d, mem_d) = res["scan", kind], res["diffrax", kind]
            rows.append((name, kind, us_s, us_d, us_d - us_s, mem_s, mem_d))
        if name == "N=32 single":
            gate_overhead = res["diffrax", "fwd"][0] - res["scan", "fwd"][0]

    head = ("case", "pass", "scan us/step", "diffrax us/step", "diff us/step")
    head += ("scan temp B", "diffrax temp B")
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for case, kind, us_s, us_d, diff, mem_s, mem_d in rows:
        cells = (case, kind, f"{us_s:.3f}", f"{us_d:.3f}", f"{diff:+.3f}")
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
