r"""Triaxial Hernquist potential.

The Hernquist potential evaluated on the ellipsoidal radius :math:`r' =
\sqrt{x^2 + (y / q_1)^2 + (z / q_2)^2}`,

.. math::

    \Phi = -\frac{G m_{\mathrm{tot}}}{r' + r_s}.

The density is :math:`\nabla^2 \Phi / (4 \pi G)` of this expression, found by
automatic differentiation. Evaluate it away from the origin, where the Hessian
is set by the floor of the square root. The full expressions are in
:class:`~galax.potential.TriaxialHernquistPotential`.

.. rubric:: Functions

.. list-table::
    :widths: 45 55

    * - :class:`~galax.potential.TriaxialHernquistPotential`
      - the potential class, with parameters ``m_tot``, ``r_s``, ``q1``, ``q2``
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
    "TriaxialHernquistPotential",
    "density",
    "gradient",
    "hessian",
    "laplacian",
    "potential",
]

from ._src.builtin.triaxialhernquist import (
    TriaxialHernquistPotential,
    density,
    gradient,
    hessian,
    laplacian,
    potential,
)
