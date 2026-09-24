# ruff: noqa: E402
"""Wall-clock comparison of galax's SCFPotential against gala's.

Run with::

    uv run --extra interop-gala python benchmarks/scf_vs_gala.py

Gala's SCFPotential requires a GSL-enabled build. The PyPI wheels are built
with ``GALA_FORCE_GSL=1``, so a plain install suffices on linux-x86_64 and
macOS-arm64.
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "True")  # must precede jax import

import argparse
import sys
import timeit

import numpy as np


def _require_gala() -> None:
    try:
        import gala  # noqa: F401
        from gala._cconfig import GSL_ENABLED
    except ImportError:
        sys.exit(
            "gala is not installed. Run:\n"
            "  uv run --extra interop-gala python benchmarks/scf_vs_gala.py"
        )
    if not GSL_ENABLED:
        sys.exit(
            "gala was built without GSL, so gala.potential.SCFPotential is "
            "unavailable. Reinstall gala from a wheel, or build with "
            "GALA_FORCE_GSL=1."
        )


GATE_POINTS = 1_000
"""Maximum positions compared by the correctness gate.

The gate exists to prove the two implementations agree before any timing is
reported, not to sample the domain exhaustively -- a systematic error shows up
in the first handful of positions. Capping keeps the check cheap at large
``npoints``.
"""


def _time(fn, *, repeat: int = 7, number: int | None = None) -> float:
    """Best-of-`repeat` seconds per call."""
    if number is None:
        # Calibrate so each timing run takes ~50ms.
        number = 1
        while timeit.timeit(fn, number=number) < 0.05:
            number *= 4
    return min(timeit.repeat(fn, repeat=repeat, number=number)) / number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npoints", type=int, nargs="+", default=[1, 100, 1_000, 100_000, 1_000_000]
    )
    parser.add_argument("--nl", type=int, nargs=2, action="append", default=None)
    args = parser.parse_args()
    nls = [tuple(x) for x in (args.nl or [(2, 2), (6, 4), (12, 6)])]

    _require_gala()

    import astropy.units as apyu
    import gala.potential as galap
    import jax
    from gala.units import galactic

    import quaxed.numpy as jnp
    import unxt as u

    import galax.potential as gp

    if not jax.config.jax_enable_x64:
        sys.exit(
            "x64 must be enabled for a fair comparison (galax would be "
            "running float32 against gala's float64)."
        )

    rows = []
    for nmax, lmax in nls:
        Snlm = np.zeros((nmax + 1, lmax + 1, lmax + 1))
        Snlm[0, 0, 0] = 1.0
        Tnlm = np.zeros_like(Snlm)

        gpot = galap.SCFPotential(
            m=1e12 * apyu.Msun,
            r_s=10.0 * apyu.kpc,
            Snlm=Snlm,
            Tnlm=Tnlm,
            units=galactic,
        )
        xpot = gp.SCFPotential(
            m_tot=u.Q(1e12, "Msun"),
            r_s=u.Q(10.0, "kpc"),
            Snlm=jnp.asarray(Snlm),
            Tnlm=jnp.asarray(Tnlm),
            units="galactic",
        )

        # Hoisted out of the `npoints` loop: re-wrapping per iteration does not
        # recompile (jit caches on the wrapped function and the argument avals,
        # not on the wrapper instance -- measured 1699 ms cold, then 1.3 ms for
        # a freshly wrapped call at the same shape), but there is no reason to
        # rebuild it either.
        fn = jax.jit(gp.potential)

        for npoints in args.npoints:
            rng = np.random.default_rng(0)
            xyz = rng.normal(size=(npoints, 3)) * 10.0

            gala_q = xyz.T * apyu.kpc
            galax_q = u.Q(jnp.asarray(xyz), "kpc")
            t = u.Q(0.0, "Gyr")

            jax.block_until_ready(fn(xpot, galax_q, t))  # compile before timing

            # Correctness gate: a fast wrong answer is not a benchmark. Uses an
            # explicit exit (not `assert`) so `python -O` can't skip it.
            #
            # Checked on at most GATE_POINTS positions. The full arrays are
            # still computed -- only the comparison is capped -- so this
            # exercises the same kernel at the same shape, while avoiding an
            # extra full-size gala evaluation and two host transfers per row
            # (at npoints=1e6 that dominated the script's runtime).
            n_gate = min(npoints, GATE_POINTS)
            got = np.asarray(fn(xpot, galax_q, t).value)[:n_gate]
            exp = gpot.energy(gala_q[:, :n_gate]).to_value("kpc2 / Myr2")
            if not np.allclose(got, exp, rtol=1e-10):
                sys.exit(
                    f"correctness gate failed at nmax={nmax}, lmax={lmax}, "
                    f"npoints={npoints} (checked {n_gate}): "
                    f"galax={got!r} gala={exp!r}. "
                    "A fast wrong answer is not a benchmark."
                )

            t_gala = _time(lambda: gpot.energy(gala_q))
            t_galax = _time(lambda: jax.block_until_ready(fn(xpot, galax_q, t)))

            rows.append((nmax, lmax, npoints, t_gala, t_galax))

    print(f"\njax backend: {jax.default_backend()}  x64: {jax.config.jax_enable_x64}\n")
    print("| nmax | lmax | N | gala (ms) | galax (ms) | speedup |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for nmax, lmax, npoints, tg, tx in rows:
        print(
            f"| {nmax} | {lmax} | {npoints:,} | {tg * 1e3:.4g} | "
            f"{tx * 1e3:.4g} | {tg / tx:.2f}x |"
        )
    print(
        "\nNote: gala's small-N rows are dominated by astropy `Quantity` / "
        "`PotentialBase` wrapper overhead (~0.5ms), essentially flat across "
        "(nmax, lmax); its C kernel (`gpot._energy`) alone runs in "
        "~0.003ms. So small-N rows compare Python wrappers, not SCF "
        "kernels -- the real nmax advantage only shows up at large N with "
        "lmax held fixed (see README)."
    )


if __name__ == "__main__":
    main()
