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
together, so the two effects are confounded: raising `lmax` adds O(lmax²)
redundant Legendre work to galax only (a known `TODO` in `multipole.py`), which
masks galax's `nmax` advantage. gala calls `gsl_sf_gegenpoly_n` once per radial
order (O(nmax²) total); galax computes all orders in one `lax.scan` (O(nmax)).

To isolate the radial (`nmax`) advantage, hold `lmax` fixed and vary only
`nmax`:

```bash
uv run --extra interop-gala python benchmarks/scf_vs_gala.py \
    --nl 2 6 --nl 6 6 --nl 12 6 --nl 24 6 --npoints 100000
```

Measured on this machine (N=100,000, correctness-gated):

| nmax | lmax |       N | gala (ms) | galax (ms) | speedup |
| ---: | ---: | ------: | --------: | ---------: | ------: |
|    2 |    6 | 100,000 |     33.46 |      50.23 |   0.67x |
|    6 |    6 | 100,000 |     55.56 |      59.33 |   0.94x |
|   12 |    6 | 100,000 |      88.5 |      65.86 |   1.34x |
|   24 |    6 | 100,000 |     153.9 |      77.06 |   2.00x |

Speedup grows monotonically with `nmax`, as expected from the O(nmax) vs.
O(nmax²) scaling above. The default mixed sweep understates this because raising
`lmax` alongside `nmax` adds galax-only overhead that eats into the radial win.
