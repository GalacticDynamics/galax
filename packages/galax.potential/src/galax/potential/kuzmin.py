r"""Kuzmin disk potential.

The Kuzmin potential is that of an axisymmetric razor-thin disk. All of the mass
is in the plane :math:`z = 0`, the total mass ``m_tot`` is finite, and ``r_s``
is the scale length. The potential is symmetric under :math:`z \to -z`. The
``density`` function returns exactly zero, because the volume density is zero
off the plane and the delta function on the plane is not represented. The
potential is the :math:`c \to 0` limit of
:class:`~galax.potential.KuzminKutuzovPotential`.

In cylindrical coordinates :math:`(R, z)` the potential is

.. math::

    \Phi(R, z) = -\frac{G m_{\mathrm{tot}}}{\sqrt{R^2 + \left(r_s + |z|\right)^2}},

where :math:`G` is the gravitational constant. The full expressions, limits and
references are in :class:`~galax.potential.KuzminPotential`.

.. rubric:: Functions

.. list-table::
    :widths: 45 55

    * - :class:`~galax.potential.KuzminPotential`
      - the potential class, with parameters ``m_tot``, ``r_s``
    * - ``potential(p, xyz)``, ``density(p, xyz)``
      - the potential and the density
    * - ``laplacian(p, xyz)``
      - the Laplacian of the potential, which is :math:`4 \pi G` times the density
    * - ``gradient(p, xyz)``, ``hessian(p, xyz)``
      - the gradient, shape ``(3,)``, and the Hessian, shape ``(3, 3)``,
        by automatic differentiation of ``potential``

.. rubric:: Units and shapes

The functions take bare arrays in the unit system of the potential, and
``pot.params(t)`` returns the dictionary ``p`` in that system, ``G`` included.
They evaluate one point at a time, so ``xyz`` has shape ``(3,)``; use
``jax.vmap`` to evaluate many points. The page :doc:`/potential_modules`
describes the contract, the units, the rule at the origin and the float32
limits.

.. rubric:: References

The disk is due to Kuzmin (1953) and was rediscovered by Toomre (1963). Binney &
Tremaine (2008) give a textbook treatment. The reference list is in the
docstring of :class:`~galax.potential.KuzminPotential`.
"""

__all__ = [
    "KuzminPotential",
    "density",
    "gradient",
    "hessian",
    "laplacian",
    "potential",
]

from ._src.builtin.kuzmin import (
    KuzminPotential,
    density,
    gradient,
    hessian,
    laplacian,
    potential,
)
