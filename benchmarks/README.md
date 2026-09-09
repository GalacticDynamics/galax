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
