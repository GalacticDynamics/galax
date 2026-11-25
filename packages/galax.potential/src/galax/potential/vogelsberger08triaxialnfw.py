r"""Triaxial NFW potential of Vogelsberger et al. (2008).

The NFW potential evaluated at the radius :math:`\tilde{r} = (r_a + r)\, r_e /
(r_a + r_e)`, where :math:`r_e = \sqrt{x^2 + y^2/q_1^2 + z^2/q_2^2}` with
:math:`q_2^2 = 3 - q_1^2`, :math:`r_a = a_r r_s` and :math:`r` is the spherical
radius. The isopotential surfaces are ellipsoids inside :math:`r_a` and near
spheres outside it. The density is :math:`\nabla^2 \Phi / (4 \pi G)` of this
expression, found by automatic differentiation. Evaluate it away from the
origin, where the Hessian and the Laplacian are set by the floor of the square
root. The full expressions are in
:class:`~galax.potential.Vogelsberger08TriaxialNFWPotential`.

.. rubric:: Functions

.. list-table::
    :widths: 45 55

    * - :class:`~galax.potential.Vogelsberger08TriaxialNFWPotential`
      - the potential class, with parameters ``m``, ``r_s``, ``q1``, ``a_r``
    * - ``potential(p, xyz)``, ``density(p, xyz)``
      - the potential and the density, the latter by automatic differentiation
    * - ``laplacian(p, xyz)``
      - the Laplacian of the potential, the trace of its Hessian
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
    "Vogelsberger08TriaxialNFWPotential",
    "density",
    "gradient",
    "hessian",
    "laplacian",
    "potential",
]

from ._src.builtin.nfw.vogelsberger08 import (
    Vogelsberger08TriaxialNFWPotential,
    density,
    gradient,
    hessian,
    laplacian,
    potential,
)
