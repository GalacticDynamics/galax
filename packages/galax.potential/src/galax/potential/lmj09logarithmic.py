r"""Triaxial logarithmic potential of Law, Majewski & Johnston (2009).

The potential is

.. math::

    \Phi = \frac{1}{2} v_c^2 \ln\left(r_s^2 + \frac{x^2}{q_1^2}
           + \frac{y^2}{q_2^2} + \frac{z^2}{q_3^2}\right),

with the coordinates :math:`(x, y)` rotated by the angle ``phi`` in the
:math:`x`-:math:`y` plane. The density follows from Poisson's equation and is
given in :class:`~galax.potential.LMJ09LogarithmicPotential`.

.. rubric:: Functions

.. list-table::
    :widths: 45 55

    * - :class:`~galax.potential.LMJ09LogarithmicPotential`
      - the potential class, with parameters ``v_c``, ``r_s``, ``q1``, ``q2``,
        ``q3``, ``phi``
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
"""

__all__ = [
    "LMJ09LogarithmicPotential",
    "density",
    "gradient",
    "hessian",
    "laplacian",
    "potential",
]

from ._src.builtin.lmj09logarithmic import (
    LMJ09LogarithmicPotential,
    density,
    gradient,
    hessian,
    laplacian,
    potential,
)
