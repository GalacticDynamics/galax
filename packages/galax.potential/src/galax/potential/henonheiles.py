r"""Henon-Heiles potential.

The potential is a harmonic term plus a cubic term in :math:`x` and :math:`y`,
divided by the square of a timescale :math:`t_s`,

.. math::

    \Phi = \frac{1}{t_s^2} \left[ \frac{1}{2} (x^2 + y^2)
           + \lambda (x^2 y - y^3 / 3) \right],

where :math:`\lambda` is ``coeff``. It does not depend on :math:`z`. The
Laplacian is :math:`2 / t_s^2` and the density is that divided by :math:`4 \pi
G`, both uniform in space.

.. rubric:: Functions

.. list-table::
    :widths: 45 55

    * - :class:`~galax.potential.HenonHeilesPotential`
      - the potential class, with parameters ``coeff``, ``timescale``
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

The full expressions are in :class:`~galax.potential.HenonHeilesPotential`.
"""

__all__ = [
    "HenonHeilesPotential",
    "density",
    "gradient",
    "hessian",
    "laplacian",
    "potential",
]

from ._src.builtin.henonheiles import (
    HenonHeilesPotential,
    density,
    gradient,
    hessian,
    laplacian,
    potential,
)
