"""Real and imaginary parts of the spherical harmonics. Private API."""

__all__: tuple[str, ...] = ()

from jaxtyping import Array, Float

from spexial import sph_harm_y_cart_all_terms

import galax.potential.custom_types as gt


def iter_Ylm(
    l_max: int, uvec: gt.BtSz3, /
) -> list[tuple[int, int, Float[Array, "*batch"], Float[Array, "*batch"]]]:
    r"""Give ``(l, m, Re Y_lm, Im Y_lm)`` for every ``0 <= m <= l <= l_max``.

    A thin adapter over `spexial.sph_harm_y_cart_all_terms`, which runs the
    Legendre and azimuth recurrences *once* across the whole table -- one pass
    per ``m``, advancing :math:`((x+iy)/r)^m` by a single complex multiply and
    walking the Legendre recurrence up in ``l`` from its seed -- and evaluates
    each harmonic from the Cartesian unit direction as :math:`N_{lm}
    p_l^m(z/r) ((x+iy)/r)^m`. That form is polynomial in :math:`x` and
    :math:`y`, and so smooth on the z-axis, where :math:`\theta` and
    :math:`\phi` are not. Neither angle is differentiable at a pole --
    :math:`\mathrm{d}\,\mathrm{acos}(z/r)` is :math:`-1/\sqrt{1 - (z/r)^2}`,
    an infinity there, and :math:`\mathrm{atan2}(y, x)` has gradient
    :math:`(-y, x)/(x^2 + y^2)`, which is :math:`0/0` -- so autodiff through
    them returns ``nan``, and the chain rule carries that into the Cartesian
    gradient and hessian of *every* term, :math:`m = 0` included. The
    Cartesian form gives the true on-axis derivative instead, which for
    :math:`m = 1` is not zero.

    Why `spexial` rather than `jax.scipy.special.sph_harm_y`: upstream pairs
    ``l[i]`` with ``theta[i]`` instead of broadcasting, so it returns silently
    wrong values for a batch of positions, and its pole derivatives are
    non-finite for every :math:`l \ge 1` -- ``nan`` at :math:`\theta = 0`, and
    at :math:`\theta = \pi` ``nan`` except the :math:`m = 1` terms, which come
    back :math:`\pm\infty`. Both are documented in `spexial`, with regression
    tests asserting the defects directly.

    Two things are adapted, and only two. `spexial` returns one complex array
    per term, while every consumer here wants the real and imaginary parts
    separately, since :math:`S_{lm}` and :math:`T_{lm}` are real and multiply
    them independently. And `spexial`'s inner axis carries both signs of the
    order, laid out as SciPy does it -- :math:`m = 0 \ldots l_{max}` first and
    the negative orders at the tail, reachable by negative indexing -- while
    the real expansions used here need only :math:`m \ge 0`. That layout is
    why ``terms[l][m]`` needs no offset: the non-negative half is already the
    front of the axis.

    Note the ``_terms`` spelling: `spexial.sph_harm_y_cart_all` computes the
    same values but returns them *stacked* into one array, and indexing a
    stacked table is what stops XLA folding each term into the caller's
    summation as it is produced -- the whole table is materialized instead.
    Measured on `MultipoleInnerPotential` at ``l_max = 12`` over a million
    positions, the stacked form ran in 17.7 s against 10 ms here, for identical
    values. Every consumer of this function immediately sums the terms.

    Terms come out in m-major order rather than the l-major order of
    ``np.tril_indices``; every caller sums them, so the order is immaterial.
    """
    terms = sph_harm_y_cart_all_terms(l_max, l_max, uvec)
    return [
        (l, m, terms[l][m].real, terms[l][m].imag)
        for m in range(l_max + 1)
        for l in range(m, l_max + 1)
    ]
