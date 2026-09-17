"""Angular projection of a density onto real spherical harmonics.

Ported from ``bfeax.density_coeffs`` and ``bfeax.sph_harm``, with the
harmonics themselves supplied by `galax`'s own `iter_Ylm` rather than a second
Legendre recurrence.
"""

__all__: tuple[str, ...] = ()

import math

from jaxtyping import Array, Float
from typing import cast

import quaxed.numpy as jnp

import galax.potential.custom_types as gt
from galax.potential._src.builtin.multipole import iter_Ylm

SQRT2: float = math.sqrt(2.0)
"""Real-harmonic normalization for ``m != 0``."""

SYMMETRIES: tuple[str, ...] = ("spherical", "axisymmetric", "triaxial")


def lm_keys(l_max: int, symmetry: str | None = None, /) -> tuple[tuple[int, int], ...]:
    r"""Return the ``(l, m)`` modes retained for a given symmetry.

    Modes that vanish identically under the stated symmetry are skipped,
    which reduces build cost and memory without affecting accuracy.

    - ``None``: every mode, :math:`-l \le m \le l`.
    - ``"spherical"``: ``(0, 0)`` only.
    - ``"axisymmetric"``: azimuthal symmetry kills :math:`m \ne 0`; equatorial
      symmetry kills odd :math:`l`, since
      :math:`Y_{l0}(\pi - \theta) = (-1)^l Y_{l0}(\theta)`.
    - ``"triaxial"``: octant symmetry. :math:`x \to -x` kills odd-:math:`m`
      cosine terms, :math:`y \to -y` kills every sine term (:math:`m < 0`),
      and :math:`z \to -z` requires :math:`l + m` even — together, even
      :math:`l` and even :math:`m \ge 0`.

    Symmetry is never inferred from parameter values: that inference is only
    sound for concrete Python scalars and would silently degrade to ``None``
    under `jax.jit`, making the build depend on whether the caller traced.
    """
    if symmetry is None:
        return tuple((l, m) for l in range(l_max + 1) for m in range(-l, l + 1))
    if symmetry == "spherical":
        return ((0, 0),)
    if symmetry == "axisymmetric":
        return tuple((l, 0) for l in range(0, l_max + 1, 2))
    if symmetry == "triaxial":
        return tuple((l, m) for l in range(0, l_max + 1, 2) for m in range(0, l + 1, 2))
    msg = f"Unknown symmetry {symmetry!r}. Use one of {SYMMETRIES} or None."
    raise ValueError(msg)


def real_ylm(
    l_max: int, keys: tuple[tuple[int, int], ...], uvec: gt.BtSz3, /
) -> Float[Array, "n_modes *batch"]:
    r"""Stack real orthonormal :math:`Y_{lm}` for ``keys``.

    `iter_Ylm` gives :math:`(\Re Y_l^m, \Im Y_l^m)` for :math:`m \ge 0`. The
    real orthonormal basis follows as

    .. math::

        Y_{l0} = \Re Y_l^0, \quad
        Y_{lm} = \sqrt{2}\,\Re Y_l^m, \quad
        Y_{l,-m} = \sqrt{2}\,\Im Y_l^m \quad (m > 0)

    matching ``bfeax.sph_harm.ylm_real`` to 7e-16; both carry the
    Condon-Shortley phase.

    ``uvec`` is a *Cartesian* unit direction, not :math:`(\theta, \phi)`.
    That is what keeps the result differentiable on the z-axis -- see
    `galax.potential._src.builtin.multipole.compute_Ylm`.
    """
    table = {(l, m): (cY, sY) for l, m, cY, sY in iter_Ylm(l_max, uvec)}
    out = []
    for l, m in keys:
        cY, sY = table[(l, abs(m))]
        if m == 0:
            out.append(cY)
        elif m > 0:
            out.append(SQRT2 * cY)
        else:
            out.append(SQRT2 * sY)
    return cast(Float[Array, "n_modes *batch"], jnp.stack(out))
