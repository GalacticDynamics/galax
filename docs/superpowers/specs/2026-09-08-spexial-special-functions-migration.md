# Migrating galax's special functions to spexial

**Date:** 2026-09-08 **Status:** design note, not yet scheduled **Related:**
[spexial#22](https://github.com/JAXtronomy/spexial/pull/22),
`docs/superpowers/specs/2026-09-08-scf-potential-design.md`

## Decision

The angular special functions — associated Legendre and real spherical harmonics
— move to `spexial`, and `galax` consumes them from there. The Gegenbauer
polynomials may either move or stay; that one is a genuine toss-up and is
decided by benchmark, not by principle.

This note records **why**, with the specific defects that motivate it, and what
the target API has to look like to be worth adopting.

## What galax owns today

Both live in `galax.potential`, written for the SCF expansion but not specific
to it.

```python
# galax/potential/_src/builtin/scf/gegenbauer.py
@ft.partial(jax.jit, static_argnums=(0,))
def gegenbauer_all(
    nmax: int, alpha: Float[Array, "L"], x: Float[Array, "*batch"], /
) -> Float[Array, "{nmax}+1 L *batch"]: ...


# galax/potential/_src/builtin/multipole.py
def compute_Ylm(
    l: int,
    m: int,
    theta: Float[Array, "*batch"],
    phi: Float[Array, "*batch"],
    *,
    l_max: int,
) -> tuple[Float[Array, "*batch"], Float[Array, "*batch"]]:  # (Re Y_lm, Im Y_lm)
    ...
```

`compute_Ylm` is consumed by `SCFPotential` and by the entire
`MultipolePotential` family. `gegenbauer_all` is consumed by `phi_nl` and
`rho_nl`.

## Why the angular half belongs in spexial

Not because the code is in the wrong folder. Because building SCF surfaced
**three real defects in `jax.scipy.special.sph_harm_y`**, and a library whose
stated purpose is to document and repair exactly that class of gap is where the
repairs belong. All three were found by measurement on this branch, not
inferred.

### Defect 1 — it does not broadcast; it pairs element-wise

`sph_harm_y(n, m, theta, phi, n_max=...)` indexes
`legendre[..., jnp.arange(len(n))]`. It pairs `l[i], m[i], theta[i], phi[i]`
positionally. Passing a length-1 `l`/`m` against a length-N `theta` therefore
returns the correct value **only at index 0** and silently wrong values
everywhere else. It also rejects 0-d input, because `len()` fails on it.

Measured against `scipy.special.lpmv` for `l = 0..3, m = 0..l` over 4 positions:
maximum absolute error **0.1 to 0.7 for every `l >= 1`**. `l = 0` is immune only
because `Y_0^0` is constant.

This had already shipped in galax: `MultipoleInnerPotential` returned different
answers for the same three positions depending on whether they were passed as a
batch or one at a time. No test caught it because every Multipole fixture
evaluates a single position, where index 0 is correct.

galax's current workaround flattens to 1-D, broadcasts `l`/`m` with `jnp.full`,
and reshapes back.

### Defect 2 — every derivative is NaN at both poles, even where the function is constant

`sph_harm_y` differentiates a `sqrt(1 - cos(theta)**2)` internally, so at
`theta = 0` and `theta = pi` all derivatives are NaN — including for
`l = m = 0`, where the function is a constant and the true derivative is exactly
zero.

galax's current workaround splices in the pole value with
`jax.lax.stop_gradient` under a `jnp.where`.

Both workarounds are load-bearing and both are hidden inside a private helper in
a potentials module, which is the wrong home for them.

### Defect 3 — the one that needs a real fix, not a workaround

On the z-axis, the Cartesian gradient and hessian of any `m >= 1` term evaluate
to exactly `0.0`, because the chain rule routes through `theta` and `phi`, which
have no directional derivative there. The true limit is non-zero — measured
`dPhi/dy -> 4.056e-04` approaching `[eps, 0, 5]` for a test quadrupole.

This is currently documented rather than fixed, in both
`cartesian_to_normalized_spherical` and `SCFPotential`. gala has the same
limitation (its C gradient divides by `sin(theta)`).

The correct fix is to evaluate the harmonics in **Cartesian form**, where no
coordinate singularity exists. That is a real piece of numerical work, it
benefits every consumer rather than just SCF, and it is precisely the kind of
thing a special-functions library should own. Doing it inside galax's multipole
module would be building a special-functions library in the wrong repository.

## Proposed spexial API

Two functions, following spexial's existing naming (`eval_gegenbauer` /
`eval_gegenbauers`, scipy-compatible where a scipy analogue exists).

```python
def sph_harm_y(
    l: ArrayLike, m: ArrayLike, theta: ArrayLike, phi: ArrayLike, /
) -> Array:
    """Complex spherical harmonic, broadcasting over all four arguments."""


def sph_legendre_p(l: ArrayLike, m: ArrayLike, x: ArrayLike, /) -> Array:
    """Normalized associated Legendre, sqrt((2l+1)/(4pi) (l-m)!/(l+m)!) P_l^m(x)."""
```

Requirements that make it adoptable, each traceable to a defect above:

1. **Genuine broadcasting** over `l`, `m`, `theta`, `phi`, including 0-d and
   rank >= 2. (Defect 1.)
2. **Finite derivatives at the poles**, ideally via `custom_jvp` rather than a
   `where`+`stop_gradient` splice. (Defect 2.)
3. **Condon–Shortley phase matching GSL's `gsl_sf_legendre_Plm`** and scipy's
   `lpmv`. This was verified to already hold for JAX's current implementation,
   so it is a compatibility constraint to preserve, not a bug to fix. galax has
   a `lpmv`-based reference test that pins it, which should move with the code.
4. **A Cartesian-form evaluation path**, so consumers can get correct gradients
   on the axis. (Defect 3.) This is the piece with real design latitude — it may
   warrant its own function rather than a flag.

Registry rows, in the PR's format:

| Function         | In JAX                              | scipy on JAX arrays | Custom JVP      | Status           |
| ---------------- | ----------------------------------- | ------------------- | --------------- | ---------------- |
| `sph_harm_y`     | yes, **broken for broadcast input** | —                   | needed at poles | extends upstream |
| `sph_legendre_p` | via `sph_harm_y` only               | value only          | needed at poles | only here        |

The `sph_harm_y` row is the interesting one: it is not "only here" and not plain
"extends upstream" — upstream exists and is **wrong** over part of its
documented domain. If the registry has no status for that, it needs one. A row
asserting "upstream returns incorrect values for broadcast `l`/`m`" with a
regression test is more valuable than the function itself, because it cannot
rot: the day JAX fixes it, the test says so.

## The Gegenbauer question is genuinely open

spexial already has:

```python
def eval_gegenbauers(n: int, alpha: ScalarLike, x: ScalarLike, /) -> Vector: ...
def eval_gegenbauer(n: int, alpha: ScalarLike, x: AnyArrayLike, /) -> AnyArray: ...
```

`eval_gegenbauers` is the same algorithm as `gegenbauer_all` — a three-term
recurrence in a `lax.scan` returning all orders in one sweep. Its PR even fixed
the `n = 0` edge case that `gegenbauer_all` special-cases, which is convergent
validation that both got the shape of the problem right.

The difference is the signature. `eval_gegenbauers` takes **scalar** `alpha` and
**scalar** `x`. `gegenbauer_all` takes a **vector** of `alpha` (one per `l`,
since SCF needs `alpha = 2l + 3/2`) and a **batch** of `x`, and fuses both into
a single scan. Adopting spexial's would mean `vmap`ping a scalar core over two
axes.

That is the same asymptotic work. Whether XLA fuses the `vmap`-of-scan as well
as the hand-fused scan is an empirical question, and it is on the exact hot path
the SCF benchmark measures.

**Three outcomes, all acceptable:**

- `vmap` fuses as well → delete `gegenbauer_all`, depend on spexial.
- It does not → propose widening `eval_gegenbauers` to accept a vector `alpha`
  and a batched `x`, contributing `gegenbauer_all`'s body upstream.
- spexial prefers the scalar-core API → `galax` keeps `gegenbauer_all` as a
  documented local fusion, with a comment pointing here.

The point is that this is decided by a number, and the number does not exist
yet.

## Sequencing

Nothing here blocks the SCF work; it all happens after.

1. **spexial ships v0.1.0 to PyPI.** It is not currently installable, so galax
   cannot depend on it at all. Hard prerequisite.
2. **Port the angular functions into spexial**, carrying galax's two workarounds
   as proper implementations and galax's `lpmv`-based Condon–Shortley reference
   test as a regression test.
3. **Add the broadcast-correctness regression test** asserting the behaviour
   upstream gets wrong, so the row cannot rot.
4. **Swap galax's `compute_Ylm`** to delegate. Gate: the full
   `tests/unit/potential` suite unchanged, and in particular
   `test_batched_matches_per_position` and the SCF `lpmv` reference test still
   green. Those two exist because of Defect 1 and are the migration's safety
   net.
5. **Benchmark the Gegenbauer question** with the SCF comparison script from the
   SCF plan's Task 10, then take one of the three outcomes above.
6. **Cartesian-form harmonics** (Defect 3), separately and last, since it is new
   capability rather than a move. When it lands, the `.. warning::` blocks in
   `cartesian_to_normalized_spherical` and `SCFPotential` come out, and the two
   `xfail`s at the origin in `tests/unit/potential/scf/test_scf.py` should be
   revisited.

## What galax keeps regardless

`cartesian_to_normalized_spherical` stays: it is a coordinate transform with
galax-specific singularity policy, not a special function. Its `safe_sqrt` /
`safe_vector_norm` guards and the masking policy are galax's call.

`phi_nl` / `rho_nl` stay: they are SCF basis functions, not general special
functions, even though they are thin wrappers over a Gegenbauer call.

## Open questions

- Does spexial want a `custom_jvp` at the poles, or is a `where` splice
  acceptable there too? The pole derivative is genuinely undefined for `m >= 1`,
  so "finite" is a policy choice, not a mathematical one, and a
  special-functions library may reasonably decline to make it.
- Should `sph_legendre_p` be public, or only `sph_harm_y`? SCF and multipole
  both want the real/imaginary split, which is `sph_legendre_p * cos(m phi)` and
  `* sin(m phi)`; exposing the Legendre directly avoids a complex intermediate
  that both consumers immediately discard.
- Is there a third consumer? If `galax.dynamics` or another JAXtronomy package
  wants these, that strengthens the case and may change the API.
