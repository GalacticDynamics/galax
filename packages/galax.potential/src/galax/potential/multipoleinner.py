r"""Multipole inner expansion potential.

The potential is a sum of solid harmonics that grow with radius, :math:`s^l
Y_l^m` with :math:`s = r / r_s`, up to a maximum degree :math:`l_\mathrm{max}`.
The tables ``Slm`` and ``Tlm`` weight the real and imaginary parts of
:math:`Y_l^m`, and ``m_tot`` and ``r_s`` set the scale :math:`G m_{\mathrm{tot}}
/ r_s` of the potential. It is a polynomial in :math:`x, y, z`, so its value is
smooth everywhere; the ``gradient`` and ``hessian`` at the exact origin are not
the limits of the polynomial, so evaluate them away from there. Every term
solves Laplace's equation, so the ``density`` function is exactly zero.

.. rubric:: Functions

.. list-table::
    :widths: 45 55

    * - :class:`~galax.potential.MultipoleInnerPotential`
      - the potential class, with parameters ``m_tot``, ``r_s``, ``Slm``, ``Tlm``
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
``jax.vmap`` to evaluate many points. Only ``potential``, ``density`` and
``laplacian`` also accept an array of shape ``(..., 3)`` directly; ``gradient``
and ``hessian`` take a single point. The page :doc:`/potential_modules`
describes the contract, the units, the rule at the origin and the float32
limits.

.. rubric:: The maximum degree

The functions take :math:`l_\mathrm{max}` from the shape of the coefficient
tables, ``Slm.shape == (l_max + 1, l_max + 1)``, so no ``l_max`` argument is
passed and the tables and the degree cannot disagree. The ``l_max`` field of
the class is checked against the same shape when the class is built.

.. rubric:: References

The expansion in solid harmonics is in Binney & Tremaine (2008), Sec. 2.4. The
reference list is in the docstring of :class:`~galax.potential.MultipoleInnerPotential`.
"""

__all__ = [
    "MultipoleInnerPotential",
    "density",
    "gradient",
    "hessian",
    "laplacian",
    "potential",
]

from ._src.builtin.multipoleinner import (
    MultipoleInnerPotential,
    density,
    gradient,
    hessian,
    laplacian,
    potential,
)
