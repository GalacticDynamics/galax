"""Real and imaginary parts of the spherical harmonics. Private API."""

__all__: tuple[str, ...] = ()

from jaxtyping import Array, Float

from spexial import sph_harm_y_cart_all_terms

import galax.potential.custom_types as gt


def iter_Ylm(
    l_max: int, uvec: gt.BtSz3, /
) -> list[tuple[int, int, Float[Array, "*batch"], Float[Array, "*batch"]]]:
    r"""Give ``(l, m, Re Y_lm, Im Y_lm)`` for every ``0 <= m <= l <= l_max``.

    Evaluated from the Cartesian direction as :math:`N_{lm} p_l^m(z/r)
    ((x+iy)/r)^m`. That form is polynomial in :math:`x` and :math:`y`, so it
    is differentiable on the z-axis; a :math:`(\theta, \phi)` form is not,
    and autodiff through one returns ``nan`` for every term, :math:`m = 0`
    included.

    Uses `spexial`, not `jax.scipy.special.sph_harm_y`, which pairs ``l[i]``
    with ``theta[i]`` instead of broadcasting and whose pole derivatives are
    non-finite.

    Uses ``..._all_terms``, not ``..._all``: the stacked table stops XLA
    folding each term into the caller's summation, measured at 17.7 s against
    10 ms for ``l_max = 12`` over a million positions.
    """
    terms = sph_harm_y_cart_all_terms(l_max, l_max, uvec)
    return [
        (l, m, terms[l][m].real, terms[l][m].imag)
        for m in range(l_max + 1)
        for l in range(m, l_max + 1)
    ]
