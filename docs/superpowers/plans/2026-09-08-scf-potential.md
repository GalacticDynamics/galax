# SCFPotential Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement gala's Self-Consistent Field basis function expansion
potential in galax, with benchmarks comparing the jitted JAX implementation
against gala's C.

**Architecture:** `SCFPotential` subclasses `AbstractSinglePotential`. The
Gegenbauer polynomials come from a single `lax.scan` over the three-term
recurrence, producing the whole `(nmax+1, lmax+1)` table in one sweep. The
angular part reuses `multipole.py`'s `compute_Ylm`. Gradient and hessian are
left to autodiff; density is written out analytically.

**Tech Stack:** JAX, equinox, `quaxed.numpy` (imported as `jnp`), `unxt` (as
`u`), plum dispatch for interop, pytest + pytest-codspeed.

**Spec:** `docs/superpowers/specs/2026-09-08-scf-potential-design.md`

## Global Constraints

- Import style: `import quaxed.numpy as jnp`, `import unxt as u`,
  `import galax.potential.custom_types as gt`. Never bare `jax.numpy` in
  potential code.
- Every public function and class needs a docstring. Doctests run in CI
  (`testpaths` includes `src/galax`), so every `>>>` example must produce
  exactly the shown output. Tests run under `JAX_ENABLE_X64=True`
  (`pyproject.toml:296`), so array reprs say `dtype=float64`.
- Commit messages use gitmoji + conventional commits
  (`✨ feat(potential): ...`), enforced by the `commitizen check` pre-commit
  hook. End every commit message with
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Pre-commit runs on every commit: ruff, ruff-format, mypy, blacken-docs,
  prettier, codespell. A
  ```python fence in markdown must be parseable Python (add `: ...` to bare
  signatures).
- Run tests with `uv run pytest`.
- Physical constants: `SQRT_FOURPI = 3.544907701811031` (gala's literal,
  `bfe_helper.cpp`).

## Reference formulae

Transcribed from `gala/potential/scf/src/bfe_helper.cpp` and `bfe.cpp`:

```
xi     = (s - 1) / (s + 1)
K_nl   = 0.5*n*(n + 4l + 3) + (l + 1)*(2l + 1)

phi_nl(s) = -SQRT_FOURPI * s^l * (1+s)^(-2l-1) * C_n^(2l+1.5)(xi)
rho_nl(s) =  SQRT_FOURPI * (K_nl / (2*pi)) * s^l / (s * (1+s)^(2l+3)) * C_n^(2l+1.5)(xi)

Phi = (G*M/r_s)   * sum_nlm phi_nl(s) * sphPlm(l,m,X) * (S_nlm*cos(m*phi) + T_nlm*sin(m*phi))
rho = (M/r_s**3)  * sum_nlm rho_nl(s) * sphPlm(l,m,X) * (S_nlm*cos(m*phi) + T_nlm*sin(m*phi))
```

`sphPlm(l,m,X)` is `Y_l^m` with the `exp(i*m*phi)` factor stripped, so
`sphPlm * cos(m*phi) == Re(Y_l^m)` and `sphPlm * sin(m*phi) == Im(Y_l^m)`, which
is exactly what `multipole.py`'s `compute_Ylm` returns.

**Analytic anchor (verified by hand):** with `nmax = lmax = 0` and
`Snlm[0,0,0] = 1`, `phi_00 = -SQRT_FOURPI/(1+s)` and
`sphPlm(0,0,X) = 1/SQRT_FOURPI`, so `Phi = -G*M/(r + r_s)` — the Hernquist
potential exactly. Likewise `K_00 = 1` gives `rho = M/(2*pi*r_s^3*s*(1+s)^3)`,
the Hernquist density exactly.

---

### Task 1: Gegenbauer recurrence — ALREADY COMPLETE

Implemented and green before this plan was written. Recorded here so the task
numbering matches the dependency order; **skip to Task 2**.

**Files:**

- Created: `src/galax/potential/_src/builtin/scf/gegenbauer.py`
- Created: `src/galax/potential/_src/builtin/scf/__init__.py`
- Created: `src/galax/potential/scf.py`
- Test: `tests/unit/potential/scf/test_gegenbauer.py` (4 tests, passing)

**Interfaces:**

- Produces:
  `gegenbauer_all(nmax: int, alpha: Float[Array, "L"], x: Float[Array, "*batch"], /) -> Float[Array, "{nmax}+1 L *batch"]`.
  `nmax` is a static argument (`jax.jit(static_argnums=(0,))`). Returns
  `C_n^alpha(x)` for `n = 0..nmax`.

- [x] **Step 1–5: complete.** Verify with
      `uv run pytest tests/unit/potential/scf/ src/galax/potential/_src/builtin/scf/ -q`
      (expect 10 passed) and commit the existing working-tree changes:

```bash
git add src/galax/potential/_src/builtin/scf/ src/galax/potential/scf.py tests/unit/potential/scf/
git commit -m "$(cat <<'EOF'
✨ feat(potential): Gegenbauer polynomials via three-term recurrence

One `lax.scan` produces the whole table for n = 0..nmax, carrying every
alpha and every x together, so the cost is O(nmax) rather than the
O(nmax^2) of evaluating each order independently.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Radial basis functions `phi_nl` and `rho_nl`

**Files:**

- Create: `src/galax/potential/_src/builtin/scf/bfe.py`
- Modify: `src/galax/potential/_src/builtin/scf/__init__.py`
- Modify: `src/galax/potential/scf.py`
- Test: `tests/unit/potential/scf/test_bfe.py`

**Interfaces:**

- Consumes: `gegenbauer_all` from Task 1.
- Produces:

  - `phi_nl(nmax: int, lmax: int, s: Float[Array, "*batch"], /) -> Float[Array, "{nmax}+1 {lmax}+1 *batch"]`
  - `rho_nl(nmax: int, lmax: int, s: Float[Array, "*batch"], /) -> Float[Array, "{nmax}+1 {lmax}+1 *batch"]`
  - `SQRT_FOURPI: float`
  - Both jitted with `static_argnums=(0, 1)`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/potential/scf/test_bfe.py`. The scalar reference functions
are transcribed line-for-line from gala's `bfe_helper.cpp` — they exist to be
obviously correct, not fast, and they are what pins the Condon–Shortley
convention and the `K_nl` normalization.

```python
"""Test the SCF radial basis functions against a scalar reference."""

import numpy as np
import scipy.special as sps

import quaxed.numpy as jnp

from galax.potential.scf import phi_nl, rho_nl

SQRT_FOURPI = 3.544907701811031


def _ref_phi_nl(n: int, l: int, s: float) -> float:
    """Scalar transcription of ``phi_nl`` from gala's bfe_helper.cpp."""
    xi = (s - 1) / (s + 1)
    return (
        -SQRT_FOURPI
        * s**l
        * (1 + s) ** (-2 * l - 1)
        * sps.eval_gegenbauer(n, 2 * l + 1.5, xi)
    )


def _ref_rho_nl(n: int, l: int, s: float) -> float:
    """Scalar transcription of ``rho_nl`` from gala's bfe_helper.cpp."""
    xi = (s - 1) / (s + 1)
    knl = 0.5 * n * (n + 4 * l + 3) + (l + 1) * (2 * l + 1)
    return (
        SQRT_FOURPI
        * (knl / (2 * np.pi))
        * s**l
        / (s * (1 + s) ** (2 * l + 3))
        * sps.eval_gegenbauer(n, 2 * l + 1.5, xi)
    )


def test_phi_nl_matches_scalar_reference() -> None:
    """Vectorized `phi_nl` equals the per-(n,l) scalar transcription."""
    nmax, lmax = 6, 4
    s = np.array([0.05, 0.5, 1.0, 3.0, 40.0])

    got = phi_nl(nmax, lmax, jnp.asarray(s))

    assert got.shape == (nmax + 1, lmax + 1, len(s))
    expected = np.array(
        [
            [[_ref_phi_nl(n, l, si) for si in s] for l in range(lmax + 1)]
            for n in range(nmax + 1)
        ]
    )
    assert jnp.allclose(got, expected, rtol=1e-10)


def test_rho_nl_matches_scalar_reference() -> None:
    """Vectorized `rho_nl` equals the per-(n,l) scalar transcription."""
    nmax, lmax = 6, 4
    s = np.array([0.05, 0.5, 1.0, 3.0, 40.0])

    got = rho_nl(nmax, lmax, jnp.asarray(s))

    assert got.shape == (nmax + 1, lmax + 1, len(s))
    expected = np.array(
        [
            [[_ref_rho_nl(n, l, si) for si in s] for l in range(lmax + 1)]
            for n in range(nmax + 1)
        ]
    )
    assert jnp.allclose(got, expected, rtol=1e-10)


def test_phi_00_is_hernquist_shape() -> None:
    """``phi_00(s) == -sqrt(4 pi) / (1 + s)``, the Hernquist anchor."""
    s = jnp.asarray([0.25, 1.0, 7.0])

    got = phi_nl(0, 0, s)

    assert jnp.allclose(got[0, 0], -SQRT_FOURPI / (1 + s), rtol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/potential/scf/test_bfe.py -q` Expected: FAIL —
`ImportError: cannot import name 'phi_nl' from 'galax.potential.scf'`

- [ ] **Step 3: Write minimal implementation**

Create `src/galax/potential/_src/builtin/scf/bfe.py`:

```python
"""Self-Consistent Field basis functions."""

__all__ = ["phi_nl", "rho_nl"]

import functools as ft

from jaxtyping import Array, Float

import jax

import quaxed.numpy as jnp

from .gegenbauer import gegenbauer_all

SQRT_FOURPI = 3.544907701811031
"""``sqrt(4 * pi)``, matching the literal in gala's ``bfe_helper.cpp``."""


def _nl_axes(
    nmax: int, lmax: int, s: Float[Array, "*batch"], /
) -> tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
    """Broadcastable ``n``, ``l`` and Gegenbauer table for the given ``s``."""
    nbatch = jnp.ndim(s)
    ls = jnp.arange(lmax + 1, dtype=s.dtype)
    # n gets one axis for l plus one per batch axis; l gets one per batch axis.
    n = jnp.expand_dims(
        jnp.arange(nmax + 1, dtype=s.dtype), tuple(range(1, 2 + nbatch))
    )
    l = jnp.expand_dims(ls, tuple(range(1, 1 + nbatch)))
    cn = gegenbauer_all(nmax, 2 * ls + 1.5, (s - 1) / (s + 1))
    return n, l, cn


@ft.partial(jax.jit, static_argnums=(0, 1))
def phi_nl(
    nmax: int, lmax: int, s: Float[Array, "*batch"], /
) -> Float[Array, "{nmax}+1 {lmax}+1 *batch"]:
    r"""Radial potential expansion terms.

    $$ \phi_{nl}(s) = -\sqrt{4\pi} \frac{s^l}{(1+s)^{2l+1}} C_n^{2l+3/2}(\xi) $$

    with $\xi = (s-1)/(s+1)$.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from galax.potential.scf import phi_nl

    The $n = l = 0$ term is the Hernquist profile up to normalization:

    >>> bool(jnp.allclose(phi_nl(0, 0, jnp.asarray(1.0)),
    ...                   -3.544907701811031 / 2))
    True

    """
    _, l, cn = _nl_axes(nmax, lmax, s)
    return -SQRT_FOURPI * s**l / (1 + s) ** (2 * l + 1) * cn


@ft.partial(jax.jit, static_argnums=(0, 1))
def rho_nl(
    nmax: int, lmax: int, s: Float[Array, "*batch"], /
) -> Float[Array, "{nmax}+1 {lmax}+1 *batch"]:
    r"""Radial density expansion terms.

    $$ \rho_{nl}(s) = \sqrt{4\pi} \frac{K_{nl}}{2\pi}
                      \frac{s^l}{s(1+s)^{2l+3}} C_n^{2l+3/2}(\xi) $$

    with $K_{nl} = \frac{1}{2}n(n+4l+3) + (l+1)(2l+1)$.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from galax.potential.scf import rho_nl

    >>> rho_nl(0, 0, jnp.asarray(1.0)).shape
    (1, 1)

    """
    n, l, cn = _nl_axes(nmax, lmax, s)
    knl = 0.5 * n * (n + 4 * l + 3) + (l + 1) * (2 * l + 1)
    return SQRT_FOURPI * (knl / (2 * jnp.pi)) * s**l / (s * (1 + s) ** (2 * l + 3)) * cn
```

Then extend the two export files:

```python
# src/galax/potential/_src/builtin/scf/__init__.py
"""Self-Consistent Field (SCF) basis function expansion."""

__all__ = ["gegenbauer_all", "phi_nl", "rho_nl"]

from .bfe import phi_nl, rho_nl
from .gegenbauer import gegenbauer_all
```

```python
# src/galax/potential/scf.py
"""`galax.potential.scf`."""

__all__ = ["gegenbauer_all", "phi_nl", "rho_nl"]

from ._src.builtin.scf import gegenbauer_all, phi_nl, rho_nl
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
`uv run pytest tests/unit/potential/scf/ src/galax/potential/_src/builtin/scf/ -q`
Expected: PASS, all tests including the new doctests.

- [ ] **Step 5: Commit**

```bash
git add src/galax/potential/_src/builtin/scf/ src/galax/potential/scf.py tests/unit/potential/scf/
git commit -m "$(cat <<'EOF'
✨ feat(potential): SCF radial basis functions

`phi_nl` and `rho_nl`, transcribed from gala's bfe_helper.cpp and checked
against a scalar per-(n,l) reference implementation.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `SCFPotential` class and its potential

**Files:**

- Modify: `src/galax/potential/_src/builtin/scf/bfe.py`
- Modify: `src/galax/potential/_src/builtin/scf/__init__.py`
- Modify: `src/galax/potential/_src/builtin/__init__.py`
- Modify: `src/galax/potential/__init__.py`
- Modify: `src/galax/potential/scf.py`
- Test: `tests/unit/potential/scf/test_scf.py`

**Interfaces:**

- Consumes: `phi_nl` from Task 2; `cartesian_to_normalized_spherical` and
  `compute_Ylm` from `galax.potential._src.builtin.multipole`.
- Produces: `SCFPotential(m_tot, r_s, Snlm, Tnlm, *, units, constants=...)` with
  static derived properties `nmax` and `lmax`. Exported from `galax.potential`
  and `galax.potential.scf`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/potential/scf/test_scf.py`:

```python
"""Test the `SCFPotential` class."""

import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp


def _monopole(m_tot: float = 1e12, r_s: float = 10.0) -> gp.SCFPotential:
    """An SCF potential whose only term is the n=l=m=0 monopole."""
    snlm = jnp.zeros((1, 1, 1)).at[0, 0, 0].set(1.0)
    return gp.SCFPotential(
        m_tot=u.Q(m_tot, "Msun"),
        r_s=u.Q(r_s, "kpc"),
        Snlm=snlm,
        Tnlm=jnp.zeros_like(snlm),
        units="galactic",
    )


def test_nmax_lmax_derived_from_coefficients() -> None:
    """`nmax` and `lmax` come from the coefficient array shape."""
    snlm = jnp.zeros((4, 3, 3))
    pot = gp.SCFPotential(
        m_tot=u.Q(1e12, "Msun"),
        r_s=u.Q(10.0, "kpc"),
        Snlm=snlm,
        Tnlm=jnp.zeros_like(snlm),
        units="galactic",
    )

    assert pot.nmax == 3
    assert pot.lmax == 2


def test_mismatched_coefficient_shapes_raise() -> None:
    """`Snlm` and `Tnlm` must share a shape."""
    with pytest.raises(Exception, match="same shape"):
        gp.SCFPotential(
            m_tot=u.Q(1e12, "Msun"),
            r_s=u.Q(10.0, "kpc"),
            Snlm=jnp.zeros((2, 2, 2)),
            Tnlm=jnp.zeros((3, 2, 2)),
            units="galactic",
        )


def test_monopole_is_hernquist_potential() -> None:
    """nmax=lmax=0 with S000=1 reproduces `HernquistPotential` exactly."""
    scf = _monopole()
    hern = gp.HernquistPotential(
        m_tot=u.Q(1e12, "Msun"), r_s=u.Q(10.0, "kpc"), units="galactic"
    )
    xyz = u.Q(np.array([[1.0, 2.0, 3.0], [-8.0, 0.5, 4.0], [0.1, 0.0, 0.0]]), "kpc")
    t = u.Q(0.0, "Gyr")

    assert jnp.allclose(scf.potential(xyz, t), hern.potential(xyz, t), rtol=1e-12)


def test_scalar_input_gives_scalar_output() -> None:
    """A single position returns a scalar, not a length-1 array."""
    got = _monopole().potential(u.Q(np.array([1.0, 2.0, 3.0]), "kpc"), u.Q(0.0, "Gyr"))

    assert got.shape == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/potential/scf/test_scf.py -q` Expected: FAIL —
`AttributeError: module 'galax.potential' has no attribute 'SCFPotential'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/galax/potential/_src/builtin/scf/bfe.py` (and add
`"SCFPotential"` to its `__all__`):

```python
from dataclasses import KW_ONLY
from typing import final

import equinox as eqx
import unxt as u
from unxt.quantity import AllowValue
from xmmutablemap import ImmutableMap

import galax.potential.custom_types as gt
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.builtin.multipole import (
    cartesian_to_normalized_spherical,
    compute_Ylm,
)
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField


@final
class SCFPotential(AbstractSinglePotential):
    r"""Self-Consistent Field (SCF) basis function expansion potential.

    The method of Hernquist & Ostriker (1992) and Lowing et al. (2011), with
    all coefficients real.

    $$ \Phi(r,\theta,\phi) = \frac{G M}{r_s} \sum_{nlm} \phi_{nl}(s)
       \left[ S_{nlm} \Re Y_l^m + T_{nlm} \Im Y_l^m \right] $$

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import galax.potential as gp

    The monopole term alone is the Hernquist potential:

    >>> Snlm = jnp.zeros((1, 1, 1)).at[0, 0, 0].set(1.0)
    >>> pot = gp.SCFPotential(m_tot=u.Q(1e12, "Msun"), r_s=u.Q(10.0, "kpc"),
    ...                       Snlm=Snlm, Tnlm=jnp.zeros_like(Snlm),
    ...                       units="galactic")
    >>> pot.nmax, pot.lmax
    (0, 0)

    """

    m_tot: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass", doc="Scale mass."
    )
    r_s: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="Scale radius."
    )
    Snlm: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        doc=r"Expansion coefficients for the $\cos(m\phi)$ terms, shape "
        r"``(nmax+1, lmax+1, lmax+1)``.",
    )
    Tnlm: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        doc=r"Expansion coefficients for the $\sin(m\phi)$ terms, shape "
        r"``(nmax+1, lmax+1, lmax+1)``.",
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    nmax: int = eqx.field(init=False, static=True, repr=False)
    lmax: int = eqx.field(init=False, static=True, repr=False)

    def __post_init__(self) -> None:
        # NOTE: must call super() -- it applies the unit system. (Do not copy
        # nfw/triaxial.py's __post_init__, which omits this.)
        super().__post_init__()
        shape = self.Snlm(u.Q(0.0, "Gyr")).shape
        object.__setattr__(self, "nmax", shape[0] - 1)
        object.__setattr__(self, "lmax", shape[1] - 1)

    def __check_init__(self) -> None:
        s_shape = self.Snlm(u.Q(0.0, "Gyr")).shape
        t_shape = self.Tnlm(u.Q(0.0, "Gyr")).shape
        if s_shape != t_shape:
            msg = (
                "Snlm and Tnlm must have the same shape. "
                f"Got {s_shape} and {t_shape}."
            )
            raise ValueError(msg)
        if len(s_shape) != 3 or s_shape[1] != s_shape[2]:
            msg = (
                "Snlm and Tnlm must have shape (nmax+1, lmax+1, lmax+1). "
                f"Got {s_shape}."
            )
            raise ValueError(msg)

    # ==========================================================================

    def _angular(
        self, theta: gt.BtFloatSz0, phi: gt.BtFloatSz0, /
    ) -> tuple[gt.BtFloatSz0, gt.BtFloatSz0]:
        """Real and imaginary ``Y_l^m`` on the full ``(l, m)`` grid."""
        lmax = self.lmax
        ls, ms = jnp.tril_indices(lmax + 1)
        cY, sY = jax.vmap(lambda l, m: compute_Ylm(l, m, theta, phi, l_max=lmax))(
            ls, ms
        )
        shape = (lmax + 1, lmax + 1, *jnp.shape(theta))
        return (
            jnp.zeros(shape).at[ls, ms].set(cY),
            jnp.zeros(shape).at[ls, ms].set(sY),
        )

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Q.from_(t, self.units["time"])

        ud = self.units["dimensionless"]
        m_tot = self.m_tot(t, ustrip=self.units["mass"])
        r_s = self.r_s(t, ustrip=self.units["length"])
        Snlm = self.Snlm(t, ustrip=ud)
        Tnlm = self.Tnlm(t, ustrip=ud)

        s, theta, phi = cartesian_to_normalized_spherical(xyz, r_s)
        phinl = phi_nl(self.nmax, self.lmax, s)
        cY, sY = self._angular(theta, phi)

        summation = jnp.einsum("nlm,nl...,lm...->...", Snlm, phinl, cY) + jnp.einsum(
            "nlm,nl...,lm...->...", Tnlm, phinl, sY
        )

        return self.constants["G"].value * m_tot / r_s * summation
```

Note `cartesian_to_normalized_spherical` is called on `xyz` directly, not
`jnp.atleast_2d(xyz)` as `multipole.py` does — the einsum's `...` handles any
batch shape including the scalar case, so no reshape dance is needed.

Register the exports. In `src/galax/potential/_src/builtin/scf/__init__.py` add
`"SCFPotential"` to `__all__` and
`from .bfe import SCFPotential, phi_nl, rho_nl`. In
`src/galax/potential/_src/builtin/__init__.py` add `"SCFPotential"` to `__all__`
and `from .scf import SCFPotential` (alongside the existing
`from .nfw import (...)` block). In `src/galax/potential/__init__.py` add
`"SCFPotential"` to `__all__` under the builtin group, add `"scf"` to the
Modules group, add `scf` to the `from . import io, params, plot` line, and add
`SCFPotential` to the `from ._src.builtin import (...)` list. In
`src/galax/potential/scf.py` add `SCFPotential` to `__all__` and the import.

- [ ] **Step 4: Run tests to verify they pass**

Run:
`uv run pytest tests/unit/potential/scf/ src/galax/potential/_src/builtin/scf/ -q`
Expected: PASS. If `test_monopole_is_hernquist_potential` fails on sign or by a
factor, the bug is in the prefactor or the `compute_Ylm` normalization — compare
against the "Reference formulae" section above, not against intuition.

- [ ] **Step 5: Commit**

```bash
git add src/galax/potential tests/unit/potential/scf/
git commit -m "$(cat <<'EOF'
✨ feat(potential): add SCFPotential

Self-Consistent Field basis function expansion (Hernquist & Ostriker 1992,
Lowing et al. 2011). The monopole term reproduces HernquistPotential
exactly, which pins the normalization.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Analytic density

**Files:**

- Modify: `src/galax/potential/_src/builtin/scf/bfe.py`
- Test: `tests/unit/potential/scf/test_scf.py`

**Interfaces:**

- Consumes: `rho_nl` from Task 2, `SCFPotential._angular` from Task 3.
- Produces: `SCFPotential._density`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/potential/scf/test_scf.py`:

```python
def test_monopole_is_hernquist_density() -> None:
    """nmax=lmax=0 with S000=1 reproduces the Hernquist density exactly."""
    scf = _monopole()
    hern = gp.HernquistPotential(
        m_tot=u.Q(1e12, "Msun"), r_s=u.Q(10.0, "kpc"), units="galactic"
    )
    xyz = u.Q(np.array([[1.0, 2.0, 3.0], [-8.0, 0.5, 4.0]]), "kpc")
    t = u.Q(0.0, "Gyr")

    assert jnp.allclose(scf.density(xyz, t), hern.density(xyz, t), rtol=1e-10)


def test_density_is_not_the_laplacian_path() -> None:
    """`_density` is analytic, and agrees with the Laplacian to 1e-6."""
    scf = _monopole()
    xyz = u.Q(np.array([3.0, 4.0, 5.0]), "kpc")
    t = u.Q(0.0, "Gyr")

    analytic = scf.density(xyz, t)
    via_laplacian = scf.laplacian(xyz, t) / (4 * jnp.pi * scf.constants["G"])

    assert jnp.allclose(analytic, via_laplacian.to(analytic.unit), rtol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/potential/scf/test_scf.py -q -k density`
Expected: FAIL — the inherited `_density` routes through the Laplacian, so
`test_monopole_is_hernquist_density` fails the `rtol=1e-10` bar (autodiff
round-off), and `test_density_is_not_the_laplacian_path` passes trivially. If
the first test unexpectedly passes at 1e-10, tighten to `rtol=1e-14` so it
genuinely discriminates the analytic path.

- [ ] **Step 3: Write minimal implementation**

Add to `SCFPotential`, directly after `_potential`:

```python
@ft.partial(jax.jit)
def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
    xyz = u.ustrip(AllowValue, self.units["length"], xyz)
    t = u.Q.from_(t, self.units["time"])

    ud = self.units["dimensionless"]
    m_tot = self.m_tot(t, ustrip=self.units["mass"])
    r_s = self.r_s(t, ustrip=self.units["length"])
    Snlm = self.Snlm(t, ustrip=ud)
    Tnlm = self.Tnlm(t, ustrip=ud)

    s, theta, phi = cartesian_to_normalized_spherical(xyz, r_s)
    rhonl = rho_nl(self.nmax, self.lmax, s)
    cY, sY = self._angular(theta, phi)

    summation = jnp.einsum("nlm,nl...,lm...->...", Snlm, rhonl, cY) + jnp.einsum(
        "nlm,nl...,lm...->...", Tnlm, rhonl, sY
    )

    return m_tot / r_s**3 * summation
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/potential/scf/ -q` Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/galax/potential/_src/builtin/scf/bfe.py tests/unit/potential/scf/
git commit -m "$(cat <<'EOF'
✨ feat(potential): analytic density for SCFPotential

SCF has a closed-form density basis, so `_density` is written out rather
than routed through the Laplacian -- faster and more accurate.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Finite derivatives at the origin and on the z-axis

The spherical coordinate transform has two singular sets. `theta = acos(z/r)` is
singular at `r = 0`; `phi = atan2(y, x)` has gradient `-y/(x^2+y^2)`, which is
`0/0` on the **entire z-axis**, not just at the origin. Whether either actually
produces a NaN depends on how `safe_vector_norm` interacts with the `s**l`
factors, so this task is **test-first and measurement-driven**: write the tests,
run them, and add a guard only where one goes red.

**Files:**

- Modify: `src/galax/potential/_src/builtin/multipole.py` (only if a test goes
  red)
- Test: `tests/unit/potential/scf/test_scf.py`

**Interfaces:**

- Consumes: `safe_vector_norm` from `galax.potential._src.utils`.
- Produces: no new public API.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/potential/scf/test_scf.py`:

```python
def _quadrupole() -> gp.SCFPotential:
    """An SCF potential with a non-zero m>0 term, to exercise the phi path."""
    snlm = jnp.zeros((2, 3, 3)).at[0, 0, 0].set(1.0).at[0, 2, 2].set(0.1)
    tnlm = jnp.zeros((2, 3, 3)).at[0, 2, 1].set(0.05)
    return gp.SCFPotential(
        m_tot=u.Q(1e12, "Msun"),
        r_s=u.Q(10.0, "kpc"),
        Snlm=snlm,
        Tnlm=tnlm,
        units="galactic",
    )


@pytest.mark.parametrize(
    ("name", "xyz"),
    [
        ("origin", [0.0, 0.0, 0.0]),
        ("z_axis_positive", [0.0, 0.0, 5.0]),
        ("z_axis_negative", [0.0, 0.0, -5.0]),
    ],
)
@pytest.mark.parametrize("method", ["potential", "gradient", "density", "hessian"])
def test_finite_on_coordinate_singularities(name, xyz, method) -> None:
    """Value and derivatives stay finite at r=0 and along the z-axis."""
    pot = _quadrupole()
    q = u.Q(np.array(xyz), "kpc")

    got = getattr(pot, method)(q, u.Q(0.0, "Gyr"))

    assert jnp.all(jnp.isfinite(u.ustrip(got.unit, got))), f"{method} at {name}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/potential/scf/test_scf.py -q -k singularities`
Expected: some subset FAILs with NaN. **Record exactly which combinations fail**
— that determines what Step 3 must fix, and nothing more.

- [ ] **Step 3: Write minimal implementation**

Fix only the cases that went red, in `cartesian_to_normalized_spherical` in
`multipole.py` (shared by both `MultipolePotential` and `SCFPotential`, so the
fix lands once for both):

```python
def cartesian_to_normalized_spherical(
    q: gt.BtSz3, r_s: gt.Sz0, /
) -> tuple[gt.BtFloatSz0, gt.BtFloatSz0, gt.BtFloatSz0]:
    r"""Convert Cartesian coordinates to normalized spherical coordinates.

    .. math::

        r = \sqrt{x^2 + y^2 + z^2}
        X = \cos(\theta) = z / r
        \phi = \tan^{-1}\left(\frac{y}{x}\right)

    Both ``r = 0`` and the ``z``-axis are coordinate singularities. ``r`` uses
    `safe_vector_norm` so that ``dr/dq`` is finite at the origin, and the
    azimuth is offset off the axis so that ``dphi/dq``, which goes as
    ``1/(x^2+y^2)``, does not evaluate ``0/0`` there. Both offsets are at the
    smallest normal float, so no physical position is perturbed.
    """
    r = safe_vector_norm(q)
    s = r / r_s
    theta = jnp.acos(q[..., 2] / r)
    tiny = jnp.finfo(jnp.promote_types(q.dtype, float)).tiny
    phi = jnp.atan2(q[..., 1], q[..., 0] + tiny)
    return s, theta, phi
```

Add `from galax.potential._src.utils import safe_vector_norm` to
`multipole.py`'s imports.

If only the `phi` cases failed, apply only the `atan2` offset and leave the norm
alone. If only the `r=0` cases failed, apply only `safe_vector_norm`. Do not
apply a guard for a case that was already green.

- [ ] **Step 4: Run tests to verify they pass**

Run:
`uv run pytest tests/unit/potential/scf/ tests/unit/potential/builtin/test_multipole.py tests/unit/potential/builtin/test_innermultipole.py tests/unit/potential/builtin/test_outermultipole.py -q`
Expected: PASS. The multipole tests must stay green — this edits shared code.

- [ ] **Step 5: Commit**

```bash
git add src/galax/potential tests/unit/potential/scf/
git commit -m "$(cat <<'EOF'
🐛 fix(potential): finite spherical derivatives on coordinate singularities

`atan2(y, x)` has gradient `-y/(x^2+y^2)`, which is 0/0 along the whole
z-axis, not just at the origin. Offsetting keeps autodiff finite there for
every potential built on `cartesian_to_normalized_spherical`.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `compute_coeffs_discrete`

**Files:**

- Create: `src/galax/potential/_src/builtin/scf/coeffs.py`
- Modify: `src/galax/potential/_src/builtin/scf/__init__.py`
- Modify: `src/galax/potential/scf.py`
- Test: `tests/unit/potential/scf/test_coeffs.py`

**Interfaces:**

- Consumes: `phi_nl` from Task 2, `compute_Ylm` from `multipole.py`.
- Produces:
  `compute_coeffs_discrete(xyz, mass, nmax, lmax, r_s, *, compute_var=False)`
  returning `(Snlm, Tnlm)` bare arrays of shape `(nmax+1, lmax+1, lmax+1)`, or
  `(Snlm, Tnlm, cov)` with `cov` of shape `(2, 2, nmax+1, lmax+1, lmax+1)` when
  `compute_var=True`.

Formula, from gala's `coeff_helper.cpp`:

```
A_nl = -(2^(8l+6) / (4*pi*K_nl)) * (n! * (n + 2l + 1.5) * Gamma(2l+1.5)^2) / Gamma(n+4l+3)
S_nlm = sum_k (2 - delta_m0) * A_nl * mass_k * phi_nl(s_k) * sphPlm(l,m,X_k) * cos(m*phi_k)
T_nlm = sum_k (2 - delta_m0) * A_nl * mass_k * phi_nl(s_k) * sphPlm(l,m,X_k) * sin(m*phi_k)
```

Use `jax.scipy.special.gammaln` for the factorial and gammas — at
`nmax=10, lmax=6` the numerator overflows float64 if computed directly.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/potential/scf/test_coeffs.py`:

```python
"""Test SCF coefficient fitting from a particle snapshot."""

import numpy as np

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
from galax.potential.scf import compute_coeffs_discrete


def _hernquist_samples(n: int, r_s: float, seed: int = 42) -> np.ndarray:
    """Isotropic samples from a Hernquist profile, via its inverse CDF."""
    rng = np.random.default_rng(seed)
    # M(<r)/M_tot = r^2/(r+r_s)^2  =>  r = r_s*sqrt(f)/(1-sqrt(f))
    f = rng.uniform(0.0, 1.0, n)
    r = r_s * np.sqrt(f) / (1 - np.sqrt(f))
    costheta = rng.uniform(-1.0, 1.0, n)
    sintheta = np.sqrt(1 - costheta**2)
    phi = rng.uniform(0.0, 2 * np.pi, n)
    return np.stack(
        [r * sintheta * np.cos(phi), r * sintheta * np.sin(phi), r * costheta],
        axis=-1,
    )


def test_hernquist_samples_recover_the_monopole() -> None:
    """Sampling a Hernquist profile gives S000 ~ 1 and nothing else."""
    n, r_s, m_tot = 200_000, 10.0, 1e12
    xyz = _hernquist_samples(n, r_s)
    mass = np.full(n, m_tot / n)

    Snlm, Tnlm = compute_coeffs_discrete(
        jnp.asarray(xyz), jnp.asarray(mass), nmax=2, lmax=2, r_s=r_s
    )

    assert Snlm.shape == (3, 3, 3)
    # The monopole carries the whole mass, normalized to the scale mass.
    assert jnp.allclose(Snlm[0, 0, 0] / m_tot, 1.0, atol=0.02)
    # Every other term is consistent with zero at this sample size.
    others = Snlm.at[0, 0, 0].set(0.0) / m_tot
    assert jnp.all(jnp.abs(others) < 0.05)
    assert jnp.all(jnp.abs(Tnlm / m_tot) < 0.05)


def test_recovered_coefficients_rebuild_the_potential() -> None:
    """Feeding the fitted coefficients back reproduces Hernquist."""
    n, r_s, m_tot = 200_000, 10.0, 1e12
    xyz = _hernquist_samples(n, r_s)
    mass = np.full(n, m_tot / n)

    Snlm, Tnlm = compute_coeffs_discrete(
        jnp.asarray(xyz), jnp.asarray(mass), nmax=2, lmax=0, r_s=r_s
    )
    scf = gp.SCFPotential(
        m_tot=u.Q(m_tot, "Msun"),
        r_s=u.Q(r_s, "kpc"),
        Snlm=Snlm / m_tot,
        Tnlm=Tnlm / m_tot,
        units="galactic",
    )
    hern = gp.HernquistPotential(
        m_tot=u.Q(m_tot, "Msun"), r_s=u.Q(r_s, "kpc"), units="galactic"
    )
    q = u.Q(np.array([[5.0, 0.0, 0.0], [0.0, 12.0, 3.0]]), "kpc")
    t = u.Q(0.0, "Gyr")

    assert jnp.allclose(scf.potential(q, t), hern.potential(q, t), rtol=0.05)


def test_compute_var_returns_a_covariance_block() -> None:
    """`compute_var=True` adds a (2, 2, ...) covariance array."""
    n, r_s = 5_000, 10.0
    xyz = _hernquist_samples(n, r_s)
    mass = np.full(n, 1.0 / n)

    Snlm, Tnlm, cov = compute_coeffs_discrete(
        jnp.asarray(xyz),
        jnp.asarray(mass),
        nmax=1,
        lmax=1,
        r_s=r_s,
        compute_var=True,
    )

    assert cov.shape == (2, 2, 2, 2, 2)
    assert jnp.all(cov[0, 0] >= 0)  # var(S) is non-negative
    assert jnp.allclose(cov[0, 1], cov[1, 0])  # symmetric
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/potential/scf/test_coeffs.py -q` Expected: FAIL —
`ImportError: cannot import name 'compute_coeffs_discrete'`

- [ ] **Step 3: Write minimal implementation**

Create `src/galax/potential/_src/builtin/scf/coeffs.py`:

```python
"""Fit SCF expansion coefficients to a particle snapshot."""

__all__ = ["compute_coeffs_discrete"]

import functools as ft

from jaxtyping import Array, Float

import jax
from jax.scipy.special import gammaln

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

from .bfe import phi_nl
from galax.potential._src.builtin.multipole import compute_Ylm


@ft.partial(jax.jit, static_argnames=("nmax", "lmax", "compute_var"))
def compute_coeffs_discrete(
    xyz: Float[Array, "N 3"],
    mass: Float[Array, "N"],
    /,
    *,
    nmax: int,
    lmax: int,
    r_s: Float[Array, ""],
    compute_var: bool = False,
) -> tuple[Float[Array, "..."], ...]:
    r"""Compute SCF coefficients from samples of a density distribution.

    Parameters
    ----------
    xyz
        Sample positions, shape ``(N, 3)``. Quantity or bare array.
    mass
        Sample masses, shape ``(N,)``. Quantity or bare array.
    nmax, lmax
        Maximum radial and angular expansion orders. Static.
    r_s
        Scale radius, in the same length unit as ``xyz``.
    compute_var
        Also return the ``(2, 2, nmax+1, lmax+1, lmax+1)`` covariance block.

    Returns
    -------
    Snlm, Tnlm
        Expansion coefficients, shape ``(nmax+1, lmax+1, lmax+1)``, as bare
        arrays in whatever unit ``mass`` was given in. Divide by the scale
        mass before handing them to `SCFPotential`.
    cov
        Only when ``compute_var=True``.

    Notes
    -----
    The per-particle contributions are materialized as a
    ``(nmax+1, lmax+1, lmax+1, N)`` array, so peak memory grows as
    ``nmax * lmax**2 * N``. At ``nmax=12, lmax=6, N=10**6`` that is roughly
    5 GB in float64 -- chunk the particles over several calls and sum the
    results, since the coefficients are a plain sum over ``k``.

    """
    # `xyz` and `r_s` must share a length unit. If `r_s` carries one it sets
    # the scale for both; otherwise every input is taken as a bare array.
    ulen = getattr(r_s, "unit", None)
    xyz = jnp.asarray(u.ustrip(AllowValue, ulen, xyz) if ulen else xyz)
    r_s = jnp.asarray(u.ustrip(AllowValue, ulen, r_s) if ulen else r_s)
    umass = getattr(mass, "unit", None)
    mass = jnp.asarray(u.ustrip(AllowValue, umass, mass) if umass else mass)

    r = jnp.linalg.vector_norm(xyz, axis=-1)
    s = r / r_s
    theta = jnp.acos(xyz[..., 2] / r)
    phi = jnp.atan2(xyz[..., 1], xyz[..., 0])

    # (nmax+1, lmax+1, N)
    phinl = phi_nl(nmax, lmax, s)

    # Angular part on the full (l, m) grid: (lmax+1, lmax+1, N)
    ls, ms = jnp.tril_indices(lmax + 1)
    cY, sY = jax.vmap(lambda l, m: compute_Ylm(l, m, theta, phi, l_max=lmax))(ls, ms)
    shape = (lmax + 1, lmax + 1, len(s))
    cYg = jnp.zeros(shape).at[ls, ms].set(cY)
    sYg = jnp.zeros(shape).at[ls, ms].set(sY)

    # A_nl, via gammaln so the numerator does not overflow.
    n = jnp.arange(nmax + 1, dtype=float)[:, None]
    l = jnp.arange(lmax + 1, dtype=float)[None, :]
    knl = 0.5 * n * (n + 4 * l + 3) + (l + 1) * (2 * l + 1)
    log_ratio = gammaln(n + 1) + 2 * gammaln(2 * l + 1.5) - gammaln(n + 4 * l + 3)
    anl = (
        -jnp.exp((8 * l + 6) * jnp.log(2.0) + log_ratio)
        / (4 * jnp.pi * knl)
        * (n + 2 * l + 1.5)
    )  # (nmax+1, lmax+1)

    # (2 - delta_m0)
    km = 2.0 - (jnp.arange(lmax + 1) == 0).astype(float)  # (lmax+1,)

    # Per-particle contribution: (nmax+1, lmax+1, lmax+1, N)
    weight = anl[:, :, None, None] * km[None, None, :, None] * mass
    contrib_s = weight * phinl[:, :, None, :] * cYg[None]
    contrib_t = weight * phinl[:, :, None, :] * sYg[None]

    Snlm = jnp.sum(contrib_s, axis=-1)
    Tnlm = jnp.sum(contrib_t, axis=-1)

    if not compute_var:
        return Snlm, Tnlm

    var_s = jnp.sum(contrib_s**2, axis=-1)
    var_t = jnp.sum(contrib_t**2, axis=-1)
    covar = jnp.sum(contrib_s * contrib_t, axis=-1)
    cov = jnp.stack([jnp.stack([var_s, covar]), jnp.stack([covar, var_t])])
    return Snlm, Tnlm, cov
```

Note the signature makes `nmax`, `lmax` and `r_s` keyword-only, unlike gala's
positional form, because they are `static_argnames`. Update the tests from Step
1 if they call positionally — they already use keywords.

Add `"compute_coeffs_discrete"` to `__all__` and the import in both
`src/galax/potential/_src/builtin/scf/__init__.py` and
`src/galax/potential/scf.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/potential/scf/ -q` Expected: PASS. If
`test_hernquist_samples_recover_the_monopole` is off by a constant factor, check
`anl` against the `coeff_helper.cpp` formula above term by term; if it is off by
a _sign_, check that `phi_nl` is the one from Task 2 (which carries its own
leading minus).

- [ ] **Step 5: Commit**

```bash
git add src/galax/potential tests/unit/potential/scf/
git commit -m "$(cat <<'EOF'
✨ feat(potential): compute SCF coefficients from a particle snapshot

`compute_coeffs_discrete`, transcribed from gala's coeff_helper.cpp. Gala
maps over (n,l,m) tasks in a process pool; here it is one vectorized
reduction, so the pool and skip_* pruning arguments are not carried over.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: gala interop

**Files:**

- Modify: `src/galax/interop/gala/potential.py`
- Test: `tests/unit/potential/scf/test_scf.py`

**Interfaces:**

- Consumes: `SCFPotential` from Task 3.
- Produces: `gala_to_galax` and `galax_to_gala` dispatches for `SCFPotential`,
  which activate `GalaIOMixin` in Task 8.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/potential/scf/test_scf.py`:

```python
from galax.interop.optional_deps import GSL_ENABLED, OptDeps


@pytest.mark.skipif(not OptDeps.GALA.installed, reason="requires gala")
@pytest.mark.skipif(not GSL_ENABLED, reason="requires gala built with GSL")
def test_roundtrip_through_gala() -> None:
    """galax -> gala -> galax preserves the potential."""
    pot = _quadrupole()
    gala_pot = gp.io.convert_potential(gp.io.GalaLibrary, pot)
    back = gp.io.convert_potential(gp.io.GalaxLibrary, gala_pot)

    xyz = u.Q(np.array([[8.0, 1.0, 2.0], [-3.0, 4.0, 5.0]]), "kpc")
    t = u.Q(0.0, "Gyr")
    assert jnp.allclose(back.potential(xyz, t), pot.potential(xyz, t), rtol=1e-12)


@pytest.mark.skipif(not OptDeps.GALA.installed, reason="requires gala")
@pytest.mark.skipif(not GSL_ENABLED, reason="requires gala built with GSL")
def test_matches_gala_potential_and_density() -> None:
    """galax and gala agree on the potential and the density."""
    import astropy.units as apyu

    pot = _quadrupole()
    gala_pot = gp.io.convert_potential(gp.io.GalaLibrary, pot)
    xyz = np.array([[8.0, 1.0, 2.0], [-3.0, 4.0, 5.0]]).T * apyu.kpc

    assert np.allclose(
        np.asarray(pot.potential(u.Q(xyz.T.value, "kpc"), u.Q(0.0, "Gyr")).value),
        gala_pot.energy(xyz).to_value("kpc2 / Myr2"),
        rtol=1e-10,
    )
    assert np.allclose(
        np.asarray(pot.density(u.Q(xyz.T.value, "kpc"), u.Q(0.0, "Gyr")).value),
        gala_pot.density(xyz).to_value("Msun / kpc3"),
        rtol=1e-10,
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/potential/scf/test_scf.py -q -k gala` Expected:
If gala is installed, FAIL with a plum dispatch error (no method for
`SCFPotential`). If gala is not installed locally, the tests SKIP — in that case
install it first with `uv sync --extra interop-gala --group test` and re-run,
because a skipped test proves nothing.

- [ ] **Step 3: Write minimal implementation**

In `src/galax/interop/gala/potential.py`, after the Multipole block (around line
1341), add:

```python
# -----------------------------------------------------------------------------
# SCF potential


@dispatch
def gala_to_galax(
    gala: galap.SCFPotential, /
) -> gp.SCFPotential | gp.TransformedPotential:
    """Convert a `gala` SCF potential to its `galax` equivalent."""
    params = gala.parameters
    pot = gp.SCFPotential(
        m_tot=params["m"],
        r_s=params["r_s"],
        Snlm=jnp.asarray(params["Snlm"]),
        Tnlm=jnp.asarray(params["Tnlm"]),
        units=gala.units,
    )
    return _apply_xop(_get_xop(gala), pot)


@dispatch
def galax_to_gala(pot: gp.SCFPotential, /) -> galap.SCFPotential:
    """Convert a `galax` SCF potential to a `gala` potential."""
    _error_if_not_all_constant_parameters(pot, "m_tot", "r_s", "Snlm", "Tnlm")

    return galap.SCFPotential(
        m=convert(pot.m_tot(0), APYQuantity),
        r_s=convert(pot.r_s(0), APYQuantity),
        Snlm=np.asarray(pot.Snlm(0).value),
        Tnlm=np.asarray(pot.Tnlm(0).value),
        units=_galax_to_gala_units(pot.units),
    )
```

Check the top of the file for whether `numpy` is already imported as `np`; add
the import if not.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/potential/scf/ -q` Expected: PASS (with gala
installed).

- [ ] **Step 5: Commit**

```bash
git add src/galax/interop tests/unit/potential/scf/
git commit -m "$(cat <<'EOF'
✨ feat(interop): convert SCFPotential to and from gala

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Standard potential test-suite coverage

Wires `SCFPotential` into the class-based test suite every other builtin
potential uses, which brings `GalaIOMixin` — potential, gradient, density and
hessian all compared against gala at `atol=1e-8`.

**Files:**

- Create: `tests/unit/potential/scf/test_scf_class.py`

**Interfaces:**

- Consumes: `SCFPotential` (Task 3), gala interop (Task 7).
- Produces: no new API.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/potential/scf/test_scf_class.py`, following the structure of
`tests/unit/potential/builtin/test_innermultipole.py`:

```python
"""Test `SCFPotential` against the standard potential test suite."""

from typing import Any, override

import pytest
from jaxtyping import Array, Shaped

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
from ..builtin.test_common import ParameterMTotMixin, ParameterRSMixin
from ..test_core import AbstractSinglePotential_Test


class TestSCFPotential(
    AbstractSinglePotential_Test,
    ParameterMTotMixin,
    ParameterRSMixin,
):
    HAS_GALA_COUNTERPART = True

    @pytest.fixture(scope="class")
    @override
    def pot_cls(self) -> type[gp.SCFPotential]:
        return gp.SCFPotential

    @pytest.fixture(scope="class")
    def field_Snlm(self) -> Shaped[Array, "2 2 2"]:
        return jnp.zeros((2, 2, 2)).at[0, 0, 0].set(1.0).at[0, 1, 1].set(0.1)

    @pytest.fixture(scope="class")
    def field_Tnlm(self) -> Shaped[Array, "2 2 2"]:
        return jnp.zeros((2, 2, 2)).at[0, 1, 1].set(0.05)

    @pytest.fixture(scope="class")
    @override
    def fields_(
        self,
        field_m_tot: u.Quantity,
        field_r_s: u.Quantity,
        field_Snlm: Shaped[Array, "2 2 2"],
        field_Tnlm: Shaped[Array, "2 2 2"],
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m_tot": field_m_tot,
            "r_s": field_r_s,
            "Snlm": field_Snlm,
            "Tnlm": field_Tnlm,
            "units": field_units,
        }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/potential/scf/test_scf_class.py -q` Expected:
FAIL. `AbstractSinglePotential_Test` asserts expected values for `potential`,
`gradient`, `density` and `hessian` at a fixed position, and this class does not
yet override them. Read the failure output to get the actual computed values.

- [ ] **Step 3: Write minimal implementation**

Add the expected-value overrides to `TestSCFPotential`, taking the numbers from
the Step 2 failure output. Look at
`tests/unit/potential/builtin/test_innermultipole.py` for the exact method names
and `u.Q(...)` shapes to match. **Cross-check at least the potential value
against gala** before pasting a number in — a test that enshrines a wrong
computed value is worse than no test.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/potential/ -q` Expected: PASS, including the
`GalaIOMixin` gala comparisons.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/potential/scf/
git commit -m "$(cat <<'EOF'
✅ test(potential): SCFPotential in the standard potential test suite

Brings GalaIOMixin, which compares potential, gradient, density and
hessian against gala.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: CodSpeed regression benchmarks

galax-only, no gala. Instruction counts under CodSpeed's instrumentation mode
are a sound basis for tracking galax against its own history, and an unsound
basis for cross-language wall-clock claims — that is Task 10's job.

**Files:**

- Create: `tests/benchmark/test_potential_scf.py`

**Interfaces:**

- Consumes: `SCFPotential` (Task 3), `compute_coeffs_discrete` (Task 6).
- Produces: no new API.

- [ ] **Step 1: Write the benchmark**

Create `tests/benchmark/test_potential_scf.py`. This is a benchmark, not a
correctness test, so there is no RED step — Step 2 verifies it runs and measures
the right thing.

```python
"""Benchmarks for the SCF potential."""

import jax
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp


def _pot(nmax: int, lmax: int) -> gp.SCFPotential:
    Snlm = jnp.zeros((nmax + 1, lmax + 1, lmax + 1)).at[0, 0, 0].set(1.0)
    return gp.SCFPotential(
        m_tot=u.Q(1e12, "Msun"),
        r_s=u.Q(10.0, "kpc"),
        Snlm=Snlm,
        Tnlm=jnp.zeros_like(Snlm),
        units="galactic",
    )


NL = [(2, 2), (6, 4), (12, 6)]
NPOINTS = [1, 1_000, 100_000]


@pytest.mark.parametrize(("nmax", "lmax"), NL, ids=lambda v: f"nl{v}")
@pytest.mark.benchmark(group="galax.potential.scf", max_time=1.0, warmup=False)
def test_compile(nmax, lmax) -> None:
    """Time to trace and compile the potential."""
    pot = _pot(nmax, lmax)
    xyz = u.Q(jnp.ones((1_000, 3)), "kpc")
    _ = jax.jit(pot.potential).lower(xyz, u.Q(0.0, "Gyr")).compile()


@pytest.mark.parametrize("npoints", NPOINTS, ids=lambda v: f"n{v}")
@pytest.mark.parametrize(("nmax", "lmax"), NL, ids=lambda v: f"nl{v}")
@pytest.mark.benchmark(group="galax.potential.scf", max_time=1.0, warmup=True)
def test_potential(nmax, lmax, npoints) -> None:
    """Evaluate the potential on a batch of positions."""
    pot = _pot(nmax, lmax)
    key = jax.random.key(0)
    xyz = u.Q(jax.random.normal(key, (npoints, 3)) * 10.0, "kpc")
    t = u.Q(0.0, "Gyr")
    fn = jax.jit(pot.potential)
    _ = jax.block_until_ready(fn(xyz, t))  # warm up the cache

    _ = jax.block_until_ready(fn(xyz, t))


@pytest.mark.parametrize("npoints", NPOINTS, ids=lambda v: f"n{v}")
@pytest.mark.parametrize(("nmax", "lmax"), NL, ids=lambda v: f"nl{v}")
@pytest.mark.benchmark(group="galax.potential.scf", max_time=1.0, warmup=True)
def test_gradient(nmax, lmax, npoints) -> None:
    """Evaluate the gradient (autodiff) on a batch of positions."""
    pot = _pot(nmax, lmax)
    key = jax.random.key(0)
    xyz = u.Q(jax.random.normal(key, (npoints, 3)) * 10.0, "kpc")
    t = u.Q(0.0, "Gyr")
    fn = jax.jit(pot.gradient)
    _ = jax.block_until_ready(fn(xyz, t))

    _ = jax.block_until_ready(fn(xyz, t))
```

- [ ] **Step 2: Verify the benchmarks run and measure**

Run: `uv run pytest tests/benchmark/test_potential_scf.py -q` Expected: PASS
(without `--codspeed` these run as ordinary tests).

Then confirm CodSpeed collects them:
`uv run pytest tests/benchmark/test_potential_scf.py --codspeed -q` Expected:
benchmark results reported for each parametrization.

- [ ] **Step 3: Commit**

```bash
git add tests/benchmark/test_potential_scf.py
git commit -m "$(cat <<'EOF'
⚡️ perf(potential): CodSpeed benchmarks for SCFPotential

Tracks galax against its own history across (nmax, lmax) and batch size.
Cross-library comparison against gala lives in benchmarks/scf_vs_gala.py,
because instruction counts do not convert to wall clock across runtimes.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Standalone gala comparison

The artifact that actually supports the speed claim.

**Files:**

- Create: `benchmarks/scf_vs_gala.py`
- Create: `benchmarks/README.md`

**Interfaces:**

- Consumes: `SCFPotential` (Task 3), gala interop (Task 7).
- Produces: a CLI script emitting a markdown table.

- [ ] **Step 1: Write the script**

Create `benchmarks/scf_vs_gala.py`. `JAX_ENABLE_X64` must be set before jax is
imported, or the comparison is float32-vs-float64 and meaningless.

```python
"""Wall-clock comparison of galax's SCFPotential against gala's.

Run with::

    uv run --extra interop-gala python benchmarks/scf_vs_gala.py

Gala's SCFPotential requires a GSL-enabled build. The PyPI wheels are built
with ``GALA_FORCE_GSL=1``, so a plain install suffices on linux-x86_64 and
macOS-arm64.
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "True")  # noqa: E402 -- must precede jax

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
    import jax
    import gala.potential as galap
    from gala.units import galactic

    import quaxed.numpy as jnp
    import unxt as u

    import galax.potential as gp

    assert jax.config.jax_enable_x64, "x64 must be on for a fair comparison"

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

        for npoints in args.npoints:
            rng = np.random.default_rng(0)
            xyz = rng.normal(size=(npoints, 3)) * 10.0

            gala_q = xyz.T * apyu.kpc
            galax_q = u.Q(jnp.asarray(xyz), "kpc")
            t = u.Q(0.0, "Gyr")

            fn = jax.jit(xpot.potential)
            jax.block_until_ready(fn(galax_q, t))  # compile before timing

            t_gala = _time(lambda: gpot.energy(gala_q))
            t_galax = _time(lambda: jax.block_until_ready(fn(galax_q, t)))

            rows.append((nmax, lmax, npoints, t_gala, t_galax))

    print(f"\njax backend: {jax.default_backend()}  x64: {jax.config.jax_enable_x64}\n")
    print("| nmax | lmax | N | gala (ms) | galax (ms) | speedup |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for nmax, lmax, npoints, tg, tx in rows:
        print(
            f"| {nmax} | {lmax} | {npoints:,} | {tg * 1e3:.4g} | "
            f"{tx * 1e3:.4g} | {tg / tx:.2f}x |"
        )


if __name__ == "__main__":
    main()
```

Create `benchmarks/README.md`:

````markdown
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
````

Requires a GSL-enabled gala build. Sets `JAX_ENABLE_X64=True` before importing
jax so both libraries run in double precision — without it the comparison is
float32 against float64 and means nothing.

````

- [ ] **Step 2: Run the script and record results**

Run: `uv run --extra interop-gala python benchmarks/scf_vs_gala.py`
Expected: a markdown table. Sanity-check the shape of the result against the
spec's prediction — galax slower at `N=1`, crossing over somewhere in the
`10^2`–`10^3` range, faster and increasingly so above that, with the
advantage growing in `nmax`.

**If the measurements contradict the prediction, report the measurements.**
Do not reshape the benchmark to produce a flattering number. A surprising
result is a finding, and if galax is slower than expected the profile is
worth understanding before anyone claims parity.

- [ ] **Step 3: Verify correctness agreement alongside the timing**

Before trusting any timing, confirm both libraries compute the same thing at
each `(nmax, lmax)` — a fast wrong answer is not a benchmark. Add to the
script, inside the `npoints` loop, before timing:

```python
got = np.asarray(fn(galax_q, t).value)
exp = gpot.energy(gala_q).to_value("kpc2 / Myr2")
assert np.allclose(got, exp, rtol=1e-10), (nmax, lmax, npoints)
```

Run again and confirm it still completes.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/
git commit -m "$(cat <<'EOF'
⚡️ perf(potential): wall-clock benchmark of SCFPotential against gala

Standalone script, not part of CI. Forces x64 before importing jax so both
libraries run in double precision, and asserts the two agree to 1e-10
before reporting any timing.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Final verification

- [ ] Run the full suite: `uv run pytest tests/ src/galax -q -m "not slow"`
- [ ] Run with gala present:
      `uv sync --extra all --group test-all && uv run pytest tests/unit/potential -q`
- [ ] Confirm `SCFPotential` is importable three ways: `gp.SCFPotential`,
      `galax.potential.scf.SCFPotential`,
      `from galax.potential import SCFPotential`
- [ ] Run the comparison script and paste its table into the PR description
- [ ] Confirm pre-commit is clean: `uv run pre-commit run --all-files`
````
