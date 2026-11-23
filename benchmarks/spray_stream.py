"""Time and memory of `galax.dynamics.experimental.SpraySimulator` per mode.

1002 particles (501 releases) over 5 Gyr, Leapfrog with h = 0.5 Myr, Chen+24
with a 4 pc Plummer progenitor, NFW and Milky Way hosts. Each case is jitted
and compiled first; the timed reps are interleaved across cases and the
median is reported, with the compiled program's ``memory_analysis`` temp.
The gradient is of the sum of positions in the host's mass scale, the
progenitor's state and ``t_age``. Run with::

    uv run python benchmarks/spray_stream.py --host nfw
    uv run python benchmarks/spray_stream.py --host mw --reps 5
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "True")  # must precede jax import

import argparse
import statistics
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import jaxmore.loop_strategies as lstrat

import unxt as u

import galax.dynamics as gd
import galax.potential as gp

gde = gd.experimental
USYS = u.unitsystem("galactic")
MW = gp.MilkyWayPotential(units=USYS)
W_TODAY = jnp.concat(
    [
        jnp.asarray([30.0, 10.0, 20.0]),
        u.ustrip("kpc/Myr", u.Q(jnp.asarray([10.0, -150.0, -20.0]), "km/s")),
    ]
)
MSAT = jnp.asarray(1e5)
N_RELEASE, STEPS_PER_RELEASE = 501, 20  # 10_000 steps of 0.5 Myr
X = (jnp.asarray(1.0), W_TODAY, jnp.asarray(5000.0))
MODES = {
    "whole": gde.WholeStream(),
    "copies-vmap": gde.PerParticle(gde.Copies(), lstrat.VMap),
    "copies-scan": gde.PerParticle(gde.Copies(), lstrat.Scan),
    "track-vmap": gde.PerParticle(gde.SharedTrack(), lstrat.VMap),
    "track-scan": gde.PerParticle(gde.SharedTrack(), lstrat.Scan),
}


def host(name, s):
    if name == "nfw":
        return gp.NFWPotential(m=1e12 * s, r_s=15.0, units=USYS)
    halo = gp.NFWPotential(m=5.4e11 * s, r_s=15.62, units=USYS)
    return gp.CompositePotential(
        **{k: v for k, v in MW.items() if k != "halo"}, halo=halo
    )


def case(host_name, mode, what):
    sim = gde.SpraySimulator(
        gde.Chen2024DF(),
        mode,
        progenitor_potential=lambda m: gp.PlummerPotential(
            m_tot=m, r_s=4e-3, units=USYS
        ),
    )

    def loss(s, w, t_age):
        sched = gde.Uniform(t_age, N_RELEASE, STEPS_PER_RELEASE)
        r = sim(jr.key(0), host(host_name, s), w, MSAT, sched)
        return jnp.sum(r.lead[:, :3]) + jnp.sum(r.trail[:, :3])

    fn = loss if what == "fwd" else jax.grad(loss, argnums=(0, 1, 2))
    compiled = jax.jit(fn).lower(*X).compile()
    temp = compiled.memory_analysis().temp_size_in_bytes / 2**20
    return compiled, temp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=["nfw", "mw"], default="nfw")
    parser.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    parser.add_argument(
        "--what", nargs="+", choices=["fwd", "grad"], default=["fwd", "grad"]
    )
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    cases = {}
    for m in args.modes:
        for w in args.what:
            t = time.perf_counter()
            fn, temp = case(args.host, MODES[m], w)
            jax.block_until_ready(fn(*X))  # warm-up
            cases[m, w] = (fn, temp, [])
            print(f"compiled {m} {w} in {time.perf_counter() - t:.1f} s", flush=True)

    for _ in range(args.reps):  # interleaved
        for fn, _, times in cases.values():
            t = time.perf_counter()
            jax.block_until_ready(fn(*X))
            times.append(time.perf_counter() - t)

    print(f"\nhost={args.host}, reps={args.reps}, load={os.getloadavg()[0]:.0f}\n")
    print("| mode | what | median time (s) | temp (MiB) |")
    print("|---|---|---:|---:|")
    for (m, w), (_, temp, times) in cases.items():
        print(f"| {m} | {w} | {statistics.median(times):.2f} | {temp:.1f} |")


if __name__ == "__main__":
    main()
