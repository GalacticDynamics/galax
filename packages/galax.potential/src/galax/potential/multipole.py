r"""Multipole inner and outer expansion potential.

The potential is the sum of the inner expansion, :math:`s^l Y_l^m`, and the
outer expansion, :math:`s^{-(l+1)} Y_l^m`, with :math:`s = r / r_s`, up to a
maximum degree :math:`l_\mathrm{max}`. The tables ``ISlm``, ``ITlm`` weight the
inner terms and ``OSlm``, ``OTlm`` the outer terms, and ``m_tot`` and ``r_s``
set the scale :math:`G m_{\mathrm{tot}} / r_s`. It diverges at the origin when
an outer coefficient is non-zero. Every term solves Laplace's equation, so the
``density`` function is exactly zero. The ``density`` of this module and of
``multipoleinner`` and ``multipoleouter`` is the same function.

.. rubric:: Functions

.. list-table::
    :widths: 45 55

    * - :class:`~galax.potential.MultipolePotential`
      - the potential class, with parameters ``m_tot``, ``r_s``, ``ISlm``,
        ``ITlm``, ``OSlm``, ``OTlm``
    * - ``potential(p, xyz)``, ``density(p, xyz)``
      - the potential and the density, which is exactly zero
    * - ``laplacian(p, xyz)``
      - the Laplacian of the potential, which is :math:`4 \pi G` times the density
    * - ``gradient(p, xyz)``, ``hessian(p, xyz)``
      - the gradient, shape ``(3,)``, and the Hessian, shape ``(3, 3)``,
        by automatic differentiation of ``potential``

.. rubric:: Units and shapes

The functions take bare arrays in the unit system of the potential, and
``pot.params(t)`` returns the dictionary ``p`` in that system, ``G`` included.
They evaluate one point at a time, so ``xyz`` has shape ``(3,)``; use
``jax.vmap`` to evaluate many points, or pass an array of shape ``(..., 3)``
directly, which these functions also accept. The page :doc:`/potential_modules`
describes the contract, the units, the rule at the origin and the float32
limits.

.. rubric:: The maximum degree

The functions take :math:`l_\mathrm{max}` from the shape of the coefficient
tables, ``ISlm.shape == (l_max + 1, l_max + 1)``, so no ``l_max`` argument is
passed and the tables and the degree cannot disagree. The ``l_max`` field of
the class is checked against the same shape when the class is built.

.. rubric:: References

The expansion in solid harmonics is in Binney & Tremaine (2008), Sec. 2.4. The
reference list is in the docstring of
:class:`~galax.potential.MultipolePotential`.
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
