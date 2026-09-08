# SCFPotential — design

**Date:** 2026-09-08 **Issue:**
[#298](https://github.com/GalacticDynamics/galax/issues/298) (feature parity),
[#9](https://github.com/GalacticDynamics/galax/issues/9) (SCF) **Status:**
approved, pending implementation plan

## Goal

Implement gala's Self-Consistent Field (SCF) basis function expansion potential
in galax, and demonstrate with benchmarks that the jitted JAX implementation is
competitive with gala's C implementation.

This is **spec 1 of 2**. It covers evaluation and discrete coefficient fitting.
`compute_coeffs` — fitting coefficients to an analytic density function — is
deferred to spec 2, because JAX has no adaptive quadrature analogue to gala's
`scipy.integrate.nquad` and that fork deserves its own design discussion. It is
also off the hot path and so contributes nothing to the speed claim.

## Scope

In scope:

- `SCFPotential` — potential, density, and (via autodiff) gradient/hessian.
- `gegenbauer_all` — Gegenbauer polynomials via three-term recurrence.
- `phi_nl`, `rho_nl` — the radial basis functions.
- `compute_coeffs_discrete` — coefficients from a particle snapshot.
- gala ↔ galax interop registration.
- Benchmarks: galax-only CodSpeed regression tracking, plus a standalone
  wall-clock comparison against gala.

Out of scope:

- `compute_coeffs` (analytic density → coefficients). Spec 2.
- `SCFInterpolatedPotential`. Deprecated upstream in gala, which directs users
  to `TimeDependentPotential`; galax's time-varying `ParameterField` is the
  general mechanism and already exists.
- gala's `skip_odd` / `skip_even` / `skip_m` / `pool` arguments to
  `compute_coeffs_discrete`. These exist to prune gala's per-term cost, which
  the vectorized implementation does not have. A caller wanting sparsity zeros
  the coefficients.

## Prior art

The `gp/scf` branch (`5cae21e3`, also `upstream/scf`) holds an earlier attempt:
`bfe.py`, `bfe_helper.py`, `coeffs.py`, `coeffs_helper.py`, `gegenbauer.py`,
`utils.py` and four test files. It is not being merged forward, for two reasons:

1. Its `GegenbauerCalculator` builds the polynomials by symbolic sympy
   differentiation of the weight function, dispatched through `lax.switch`. The
   source comments describe it as "still VERY slow for nmax > 7". This would
   defeat the benchmark that motivates the work.
2. The potential class predates three API generations — `_src/core.py` rather
   than `base_single.py`, the pre-2.0 `unxt` quantity API, and no
   `custom_types`.

The normalization constants in `coeffs_helper.py` and the test files are worth
consulting during implementation.

## Architecture

Layout follows `_src/builtin/nfw/`, the existing precedent for a multi-file
family of related potentials.

```
src/galax/potential/
├── scf.py                                  NEW  public re-export shim
├── __init__.py                             MOD  + "scf" module, + SCFPotential
└── _src/builtin/
    ├── __init__.py                         MOD  + SCFPotential
    └── scf/                                NEW
        ├── __init__.py
        ├── gegenbauer.py                   gegenbauer_all
        ├── bfe.py                          SCFPotential, phi_nl, rho_nl
        └── coeffs.py                       compute_coeffs_discrete
src/galax/interop/gala/potential.py         MOD  gala ↔ galax dispatch
tests/unit/potential/scf/                   NEW  unit tests
tests/benchmark/test_potential_scf.py       NEW  galax-only CodSpeed
benchmarks/scf_vs_gala.py                   NEW  standalone wall-clock report
```

Public API:

| Name                                 | Exposed at                                  |
| ------------------------------------ | ------------------------------------------- |
| `SCFPotential`                       | `galax.potential` and `galax.potential.scf` |
| `compute_coeffs_discrete`            | `galax.potential.scf`                       |
| `gegenbauer_all`, `phi_nl`, `rho_nl` | `galax.potential.scf`                       |

`compute_coeffs_discrete` is not a potential, so it is not added to
`_src/builtin/__init__.py`'s `__all__`. `_src/builtin/scf/__init__.py` exports
it; `_src/builtin/__init__.py` re-exports only `SCFPotential`; the public
`potential/scf.py` shim imports the rest directly from `._src.builtin.scf`. This
deep-import pattern already exists in `potential/params.py`, which pulls
`TimeDependentTranslationParameter` from `._src.xfm.translate`.

### The class

`SCFPotential(AbstractSinglePotential)`:

- `m_tot`, `r_s` — `ParameterField`, dimensions `mass` and `length`.
- `Snlm`, `Tnlm` — `ParameterField`, dimensionless, shape
  `(nmax+1, lmax+1, lmax+1)`.
- `nmax`, `lmax` — static `eqx.field`, derived from the coefficient shape.
- `__check_init__` validates that `Snlm` and `Tnlm` share that shape, following
  `AbstractMultipolePotential`'s idiom.

`_potential` and `_density` are hand-written. `_gradient` and `_hessian` are
**not** overridden — `base.py:178,238` derives them by autodiff. `_density` is
written out rather than routed through the Laplacian because SCF has an analytic
density basis, which is both faster and more accurate.

Note the parameter is named `m_tot`, following galax convention, where gala
calls it `m`. The interop layer maps between them.

## Numerics

### Gegenbauer recurrence

```python
def gegenbauer_all(
    nmax: int, alpha: Float[Array, "L"], x: Float[Array, "*batch"]
) -> Float[Array, "{nmax}+1 L *batch"]: ...
```

`nmax` is static. `lax.scan` over `n = 2..nmax` carrying `(C_{n-1}, C_n)` of
shape `(L, *batch)`, seeded `C_0 = 1`, `C_1 = 2·α·x`, stepping

```
n·C_n^α = 2x(n+α−1)·C_{n−1}^α − (n+2α−2)·C_{n−2}^α
```

SCF calls this once per evaluation with `alpha = 2·arange(lmax+1) + 1.5` and
`x = ξ = (s−1)/(s+1)`, obtaining the entire `(nmax+1, lmax+1)` table in one
sweep.

**This is where the performance argument lives.** Gala calls
`gsl_sf_gegenpoly_n(n, α, ξ)` once per `(n,l,m)` term, and GSL's routine itself
runs the recurrence from scratch, costing O(n). Gala therefore performs O(nmax²)
work per `(l, position)` where the single scan obtains all `n` in O(nmax). The
batched advantage should therefore _grow_ with `nmax`, which is a falsifiable
prediction the benchmark can check.

### Basis functions

Transcribed from gala's `bfe_helper.cpp`:

```
φ_nl(s) = −√(4π) · s^l / (1+s)^(2l+1) · C_n^(2l+3/2)(ξ)

ρ_nl(s) =  √(4π) · (K_nl / 2π) · s^l / (s·(1+s)^(2l+3)) · C_n^(2l+3/2)(ξ)

K_nl = ½·n·(n+4l+3) + (l+1)·(2l+1)
```

The angular part reuses `multipole.py`'s `compute_Ylm` unchanged. Gala's
`gsl_sf_legendre_sphPlm(l,m,X)` is exactly `Y_l^m` with the `e^{imφ}` factor
stripped, so gala's

```
Φ_nl · sphPlm · (S·cos(mφ) + T·sin(mφ))
```

is our

```
Φ_nl · (S·Re(Y_l^m) + T·Im(Y_l^m))
```

The final contraction is a dense `einsum` over `(nmax+1, lmax+1, lmax+1)`. The
`m > l` block is structurally zero in the coefficient arrays, as it is in gala,
so no triangular indexing is needed.

### Discrete coefficients

`compute_coeffs_discrete(xyz, mass, nmax, lmax, r_s, *, compute_var=False)`,
transcribed from `coeff_helper.cpp`:

```
Ã_nl = −(2^(8l+6) / (4π·K_nl)) · (n! · (n+2l+1.5) · Γ(2l+1.5)²) / Γ(n+4l+3)

S_nlm = Σ_k (2−δ_m0) · Ã_nl · m_k · φ_nlm(s_k, φ_k, X_k) · cos(m·φ_k)
T_nlm = Σ_k (2−δ_m0) · Ã_nl · m_k · φ_nlm(s_k, φ_k, X_k) · sin(m·φ_k)
```

Factorials and gammas via `gammaln` to avoid overflow at large `n`, `l`. Gala
runs this as a Python `pool.map` over `(n,l,m)` tasks; here it is a single
vectorized reduction. `compute_var=True` additionally returns the `(2,2,...)`
covariance block, matching gala.

Units: `xyz`, `mass`, and `r_s` accept either `unxt` quantities or bare arrays,
stripped with `u.ustrip(AllowValue, ...)` exactly as the potential methods do.
`nmax` and `lmax` are static ints. The returned coefficients are dimensionless
by construction and are returned as bare arrays, so they can be fed straight
back into `SCFPotential(Snlm=..., Tnlm=...)`.

### Known numerical risks

These are the two places implementation is most likely to go wrong. Both have
dedicated tests.

**Condon–Shortley phase.** GSL's `sphPlm` and JAX's `sph_harm_y` may differ by
`(−1)^m`. `multipole.py` already round-trips against gala's `MultipolePotential`
and passes, which suggests the conventions align — but that is inference, not
proof. The scalar reference-implementation test checks it directly against
gala's formula. A discrepancy surfaces at odd `m`.

**The origin.** `safe_vector_norm` / `safe_sqrt` (`_src/utils.py`, added in
`e6ef7d55`) make `r` differentiable at `r = 0`, and are necessary here. They are
**not sufficient**. SCF computes `theta = acos(z/r)`, whose derivative goes as
`1/r`. Terms with `l ≥ 1` carry `s^l`, which kills or bounds that as `r → 0`.
The `l = 0` term does not: `dY/dθ ≡ 0` exactly, so autodiff evaluates
`0 · inf → NaN`. This needs an explicit guard on the angular factor — likely
`jnp.where`, chosen so the primal value stays exact — not merely a safe-norm
swap. The origin test asserts value, gradient, and hessian are all finite at
`r = 0`.

## Testing

| Test                                                            | Needs gala | Catches                         |
| --------------------------------------------------------------- | ---------- | ------------------------------- |
| `gegenbauer_all` vs `scipy.special.gegenbauer`                  | no         | recurrence errors               |
| `nmax=lmax=0` reduces to `HernquistPotential`                   | no         | normalization, overall sign     |
| Scalar reference impl vs vectorized                             | no         | Condon–Shortley phase, indexing |
| Finite value/grad/hessian at `r=0`                              | no         | the `l=0` NaN above             |
| `GalaIOMixin`, atol 1e-8                                        | yes        | everything, against the C       |
| `compute_coeffs_discrete` on Hernquist samples → `S[0,0,0] ≈ 1` | no         | coefficient normalization       |
| `AbstractSinglePotential_Test` + parameter mixins               | no         | the usual class contract        |

The scalar reference implementation is roughly fifteen lines transcribed
directly from `bfe_helper.cpp`, looping over `(n,l,m)`, living in the test file.
It exists to be obviously correct rather than fast, and the vectorized
implementation is asserted equal to it over randomized inputs. This gives the
verifiability of a literal port without paying for it at runtime, and it runs
with no gala installed.

`GalaIOMixin` comes free once the interop dispatch is registered; it already
compares potential, gradient, density, and hessian for every builtin potential,
and the `check_interop` CI job installs gala with `--extra all`. Gala's PyPI
wheels are built with `GALA_FORCE_GSL=1`, so `SCFPotential` — which is
`GSL_only=True` — is available there.

Tests run under `JAX_ENABLE_X64=True` (`pyproject.toml:296`), so comparisons
against gala are float64 on both sides.

## Benchmarks

Two separate artifacts, because they answer different questions.

**`tests/benchmark/test_potential_scf.py`** — galax-only regression tracking.
Runs in the existing CodSpeed job under instrumentation mode, reusing the
`Arguments` / `process_pytest_paramatrization` harness already in
`test_experimental.py` (plan-time call: import from the sibling module, or hoist
to `tests/benchmark/_harness.py`). Separate jit-compile and execute benchmarks,
parametrized over `(nmax, lmax)` and batch size. No gala — the CI job stays on
`uv sync --group test`, and instruction counts are not a sound basis for
cross-language wall-clock claims anyway.

**`benchmarks/scf_vs_gala.py`** — the actual comparison. Standalone wall-clock,
not part of the test suite. Sets `JAX_ENABLE_X64=True` before importing jax,
calls `block_until_ready` on every galax evaluation, excludes warmup and compile
time from the timed region. Sweeps batch size `1 … 10⁶` crossed with
`(nmax, lmax)`. Emits a markdown table and a plot for the pull request
description and the docs. Exits with a clear message if gala is absent or was
built without GSL.

### What the benchmark is expected to show

- **Single position:** galax slower. JAX dispatch overhead dominates a few
  thousand flops of actual work.
- **Crossover:** somewhere in the 10²–10³ particle range.
- **Large batches:** galax faster, and increasingly so with `nmax`, per the
  O(nmax) vs O(nmax²) argument above.

If measurements contradict this, the report states the measurements. The
benchmark will not be reshaped to flatter the result.

## Decisions deferred to the implementation plan

- Whether to import the benchmark harness from `test_experimental.py` or hoist
  it to a shared module.
- The exact form of the `l = 0` angular guard at the origin.
- Whether `phi_nl` / `rho_nl` take pre-computed Gegenbauer tables or compute
  them internally.
