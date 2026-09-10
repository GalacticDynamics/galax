# Benchmarks

Standalone benchmark scripts. These are **not** part of the test suite and are
not run in CI; the CodSpeed benchmarks in `tests/benchmark/` handle regression
tracking.

## `scf_vs_gala.py`

Wall-clock comparison of `galax.potential.SCFPotential` against
`gala.potential.SCFPotential`.

```bash
uv run --extra interop-gala python benchmarks/scf_vs_gala.py
```

Requires a GSL-enabled gala build. Sets `JAX_ENABLE_X64=True` before importing
jax so both libraries run in double precision — without it the comparison is
float32 against float64 and means nothing.

### Small-N rows compare wrappers, not kernels

gala's per-call time at small N is essentially flat across `(nmax, lmax)`
(~0.5ms) because it is dominated by astropy `Quantity` / `PotentialBase` wrapper
overhead — gala's own C entry point (`gpot._energy(xyz, t)`) runs in ~0.003ms,
about 170x faster than its wrapper. So a statement like "galax is 9.7x faster at
N=1" is a statement about Python wrapper overhead, not about SCF numerics. The
script prints this caveat directly beneath the table.

### Isolating the nmax advantage

The default sweep — `(2, 2)`, `(6, 4)`, `(12, 6)` — varies `nmax` and `lmax`
together, so the two effects are confounded. gala calls `gsl_sf_gegenpoly_n`
once per radial order (O(nmax²) total); galax computes all orders in one
`lax.scan` (O(nmax)), and raising `lmax` alongside `nmax` mixes an O(lmax²)
angular cost into the same column.

To isolate the radial (`nmax`) advantage, hold `lmax` fixed and vary only
`nmax`:

```bash
uv run --extra interop-gala python benchmarks/scf_vs_gala.py \
    --nl 2 6 --nl 6 6 --nl 12 6 --nl 24 6 --npoints 100000
```

Measured on this machine (N=100,000, correctness-gated), with a monopole-only
coefficient array (`Snlm[0, 0, 0] = 1`, everything else zero). This is still a
fair timing comparison: both implementations loop over every `(n, l, m)` index
in the expansion regardless of whether its coefficient is zero, so the
zeroed-out terms cost the same as non-zero ones would.

| nmax | lmax |       N | gala (ms) | galax (ms) | speedup |
| ---: | ---: | ------: | --------: | ---------: | ------: |
|    2 |    6 | 100,000 |     34.01 |      6.993 |   4.86x |
|    6 |    6 | 100,000 |     55.82 |      10.99 |   5.08x |
|   12 |    6 | 100,000 |     88.31 |      16.35 |   5.40x |
|   24 |    6 | 100,000 |     155.7 |      28.76 |   5.41x |

Speedup still grows with `nmax`, as the O(nmax) vs. O(nmax²) scaling predicts,
but only from 4.86x to 5.41x — the trend is now a detail rather than the story.

**These numbers replace an earlier table, and the change is worth recording.**
It previously read 0.67x, 0.94x, 1.34x, 2.00x across the same four rows — galax
_losing_ to gala at `nmax = 2` and only pulling ahead by `nmax = 12`. That was
read at the time as galax's angular work eating into its radial advantage, which
was true but not inherent: the summation materialized the harmonics into an
`(l, m, *batch)` grid before contracting it, and materializing a grid to
contract is what stops XLA fusing each term into the sum as it is produced.
Folding the terms instead removed a cost that scaled with `lmax` and not with
`nmax`, which is exactly why it hurt most in the row where `nmax` was smallest.

The lesson generalizes past this table: a speedup that grows with a parameter is
not evidence that the parameter is what the algorithm is good at. Here it was
evidence of a fixed overhead being amortized.
