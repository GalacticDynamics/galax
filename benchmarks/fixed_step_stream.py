"""Time `fixed_step_stream` under ``jit(vmap(...))`` against `MockStreamGenerator`.

Prints milliseconds per stream for a batch of streams with different host
masses, progenitor positions and keys, on whatever device JAX picks (run once
with ``JAX_PLATFORMS=cpu`` and once on a GPU machine).

    uv run python benchmarks/fixed_step_stream.py [--n-streams 500] [--n-msg 3]
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "True")

import argparse
import time
import warnings

import jax
import jax.numpy as jnp
import jax.random as jr

import unxt as u

import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp

W0 = jnp.array([15.0, 0.0, 10.0, 0.0, 0.17, -0.05])  # kpc, kpc/Myr
PROG_MASS = 1e7  # Msun


def timed(f, *args):
    jax.block_until_ready(f(*args))  # compile
    t0 = time.perf_counter()
    jax.block_until_ready(f(*args))
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-streams", type=int, default=500)
    ap.add_argument("--n-release", type=int, default=1000)
    ap.add_argument("--t-age", type=float, default=3000.0)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--n-msg", type=int, default=3, help="MockStreamGenerator runs")
    ap.add_argument("--df", choices=["fardal", "chen"], default="fardal")
    args = ap.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df = gd.FardalStreamDF() if args.df == "fardal" else gd.ChenStreamDF()
    n = args.n_streams
    k = jr.split(jr.key(0), 3)
    log10_m = 12.0 + 0.1 * jr.normal(k[0], (n,))
    w0s = W0 * (1 + 0.02 * jr.normal(k[1], (n, 6)))
    keys = jr.split(k[2], n)

    @jax.jit
    @jax.vmap
    def batch(log10_m, w0, key):
        pot = gp.NFWPotential(m=10.0**log10_m, r_s=15.0, units="galactic")
        return gd.experimental.fixed_step_stream(
            pot, df, key, w0, PROG_MASS, args.t_age,
            dt=args.dt, n_release=args.n_release, t_age_max=args.t_age,
        )  # fmt: skip

    t_fast = timed(batch, log10_m, w0s, keys)

    pot = gp.NFWPotential(m=1e12, r_s=15.0, units="galactic")
    msg = gd.MockStreamGenerator(df, pot)
    ts = u.Q(jnp.linspace(-args.t_age, 0.0, args.n_release), "Myr")
    m = u.Q(PROG_MASS, "Msun")

    def one(w0, key):
        psc = gc.PhaseSpaceCoordinate(
            q=u.Q(w0[:3], "kpc"), p=u.Q(w0[3:], "kpc/Myr"), t=u.Q(0.0, "Myr")
        )
        return msg.run(key, ts, psc, m)[0]["lead"].q.x

    jax.block_until_ready(one(w0s[0], keys[0]))  # compile
    t0 = time.perf_counter()
    for i in range(args.n_msg):
        jax.block_until_ready(one(w0s[i], keys[i]))
    t_msg = (time.perf_counter() - t0) / args.n_msg

    print(
        f"device {jax.devices()[0].platform}; {2 * args.n_release} particles, "
        f"t_age {args.t_age:g} Myr, dt {args.dt:g} Myr, {args.df}"
    )
    print(
        f"fixed_step_stream, jit(vmap) over {n} streams: "
        f"{t_fast:.2f} s total, {1e3 * t_fast / n:.1f} ms/stream"
    )
    print(
        f"MockStreamGenerator.run, one stream at a time ({args.n_msg} runs): "
        f"{1e3 * t_msg:.0f} ms/stream"
    )
    print(f"speed-up {t_msg / (t_fast / n):.0f}x")


if __name__ == "__main__":
    main()
