r"""Multipole potential with inner and outer expansions.

The potential is a sum of solid harmonics, in the scaled radius :math:`s = r /
r_s` and the real and imaginary parts of the spherical harmonics :math:`Y_{lm}`,
for :math:`0 \le m \le l \le l_\mathrm{max}`. Each term solves Laplace's equation,
so the density and the Laplacian are zero wherever the potential is evaluated,
:math:`r > 0`, and the functions return exact zeros. The mass sits at the origin
(outer terms) or at infinity (inner terms).

.. math::

    \Phi = \frac{G m_{\mathrm{tot}}}{r_s} \sum_{l, m}
    \left[ s^l \left(IS_{lm}\, \mathrm{Re}\,Y_{lm}
                    + IT_{lm}\, \mathrm{Im}\,Y_{lm}\right)
    + s^{-(l+1)} \left(OS_{lm}\, \mathrm{Re}\,Y_{lm}
                    + OT_{lm}\, \mathrm{Im}\,Y_{lm}\right) \right].

.. rubric:: Functions

.. list-table::
    :widths: 45 55

    * - :class:`~galax.potential.MultipolePotential`
      - the potential class, with parameters ``m_tot``, ``r_s``, ``ISlm``,
        ``ITlm``, ``OSlm``, ``OTlm`` and the static ``l_max``
    * - ``potential(p, xyz)``
      - the potential
    * - ``density(p, xyz)``, ``laplacian(p, xyz)``
      - the density and the Laplacian of the potential, both exactly zero
    * - ``gradient(p, xyz)``, ``hessian(p, xyz)``
      - the gradient, shape ``(3,)``, and the Hessian, shape ``(3, 3)``,
        by automatic differentiation of ``potential``

.. rubric:: Units and shapes

The functions take bare arrays in the unit system of the potential, and
``pot.params(t)`` returns the dictionary ``p`` in that system, ``G`` included.
They evaluate one point at a time, so ``xyz`` has shape ``(3,)``; use
``jax.vmap`` to evaluate many points. The coefficient arrays in ``p`` have the
shape ``(l_max + 1, l_max + 1)``, and the functions read ``l_max`` from that
shape. The page :doc:`/potential_modules` describes the contract, the units, the
rule at the origin and the float32 limits.

The full expressions are in :class:`~galax.potential.MultipolePotential`.
"""

__all__ = [
    "MultipolePotential",
    "density",
    "gradient",
    "hessian",
    "laplacian",
    "potential",
]

from ._src.builtin.multipole import (
    MultipolePotential,
    density,
    gradient,
    hessian,
    laplacian,
    potential,
)
