#!/usr/bin/env python
r"""Reproduce `galax.potential`'s Burkert core polynomial, and check its precision.

`BurkertPotential`'s enclosed mass is proportional to the bracket

$$ B(x) = 2\ln(1+x) + \ln(1+x^2) - 2\tan^{-1}x = 4x^3 h(x), \quad x = r / r_s, $$

whose terms cancel to $\sim 4x^3/3$ as $x \to 0$. Up to $x = 0.8$,
``burkert._mass_bracket`` evaluates $h$ as the degree-20 polynomial
``burkert._BRACKET_FIT``. This script

1. refits it: `mpmath.chebyfit` of $h$ on $[0, 0.8]$ at 60 digits, $h$ taken
   from its Taylor series $\sum_j x^{4j}(1/(4j+3) - x/(4j+4))$ near 0 and
   from $B / 4x^3$ elsewhere;
2. checks that the coefficients are bit for bit those in ``_BRACKET_FIT``;
3. measures ``_mass_bracket`` against $B$ at 40 digits: the error of the
   polynomial range $[0, 0.8]$, and over $10^{-8}$ to $10^8$, which also
   covers the direct form used beyond 0.8.

Exits 1 if the coefficients differ or either error exceeds its bound (2 and
4.5 ulp). Needs `mpmath` (in the ``test`` dependency group).

Usage: ``uv run python scripts/fit_burkert_mass_bracket.py``
"""

import sys

import jax
import mpmath as mp
import numpy as np

jax.config.update("jax_enable_x64", True)  # noqa: FBT003

import jax.numpy as jnp  # noqa: E402

from galax.potential._src.builtin import burkert  # noqa: E402

X_MAX = (
    "0.8"  # a string, parsed at the fit's precision: mpf("0.8") here would be 15 digits
)
DEGREE = 20
FIT_DIGITS = 60
EPS = float(np.finfo(np.float64).eps)
BOUNDS = {"fit range [0, 0.8]": 2.0, "1e-8 to 1e8": 4.5}  # ulp


def h(x: mp.mpf, /) -> mp.mpf:
    """Return $B(x) / 4x^3$, by its Taylor series below 0.05 (no cancellation)."""
    if x < mp.mpf("0.05"):
        terms = (
            x ** (4 * j) * (mp.mpf(1) / (4 * j + 3) - x / (4 * j + 4))
            for j in range(40)
        )
        return mp.fsum(terms)
    return (2 * mp.log1p(x) + mp.log1p(x**2) - 2 * mp.atan(x)) / (4 * x**3)


def bracket(x: float, /) -> mp.mpf:
    """Return $B(x)$ at the working precision."""
    xm = mp.mpf(x)
    return (
        4 * xm**3 * h(xm)
        if xm < 1
        else 2 * mp.log1p(xm) + mp.log1p(xm**2) - 2 * mp.atan(xm)
    )


def max_ulp(xs: np.ndarray, /) -> tuple[float, float]:
    """Return the largest relative error of ``_mass_bracket`` in ulp, and where."""
    got = np.asarray(burkert._mass_bracket(jnp.asarray(xs)))  # noqa: SLF001
    ref = np.array([float(bracket(float(x))) for x in xs])
    err = np.abs(got - ref) / np.abs(ref) / EPS
    return float(err.max()), float(xs[err.argmax()])


def main() -> int:
    """Refit, compare with ``_BRACKET_FIT``, and check the error bounds."""
    ok = True

    with mp.workdps(FIT_DIGITS):
        poly, fit_err = mp.chebyfit(h, [0, mp.mpf(X_MAX)], DEGREE + 1, error=True)
    coeffs = tuple(float(c) for c in poly)
    print(f"degree-{DEGREE} fit of h on [0, {X_MAX}], fit error {mp.nstr(fit_err, 3)}")
    print("coefficients, highest degree first:")
    print(repr(coeffs))

    if coeffs == burkert._BRACKET_FIT:  # noqa: SLF001
        print("matches burkert._BRACKET_FIT exactly")
    else:
        ok = False
        print("DIFFERS from burkert._BRACKET_FIT:", file=sys.stderr)
        for k, (new, old) in enumerate(zip(coeffs, burkert._BRACKET_FIT, strict=True)):  # noqa: SLF001
            if new != old:
                print(f"  [{k}] {new!r} != {old!r}", file=sys.stderr)

    mp.mp.dps = 40
    grids = {
        "fit range [0, 0.8]": np.linspace(1e-6, 0.8, 4001),
        "1e-8 to 1e8": np.geomspace(1e-8, 1e8, 6001),
    }
    for name, xs in grids.items():
        err, where = max_ulp(xs)
        bound = BOUNDS[name]
        status = "ok" if err <= bound else "EXCEEDS"
        print(f"{name}: max {err:.2f} ulp at x = {where:.6g} (bound {bound}) {status}")
        ok &= err <= bound

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
