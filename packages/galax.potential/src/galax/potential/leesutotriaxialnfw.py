r"""Lee-Suto triaxial NFW potential.

The model is an analytic approximation to the potential of an NFW halo whose
iso-density surfaces are concentric, coaxial ellipsoids. The potential is the
first-order expansion, in the squared eccentricities, of the exact potential of
the triaxial density, so it is accurate for moderate flattening. The parameter
``m`` sets the mass scale, ``r_s`` is the scale radius, and ``a1``, ``a2`` and
``a3`` are the lengths of the major, intermediate and minor axes, of which only
the ratios matter. The model is only physical for :math:`a_1 \ge a_2 \ge a_3`,
and the total mass is infinite.

The density that the potential approximates is

.. math::

    \rho_{\mathrm{tri}}(\mathbf{x}) = \frac{\rho_s}{(R / r_s)(1 + R / r_s)^2},
    \qquad R^2 = a_1^2 \left(\frac{x^2}{a_1^2} + \frac{y^2}{a_2^2}
    + \frac{z^2}{a_3^2}\right),

which has the inner slope :math:`-1` and the outer slope :math:`-3` of the NFW
profile. The ``density`` function is the Laplacian of the potential divided by
:math:`4 \pi G`, which is not the density above. The full expressions, limits
and references are in :class:`~galax.potential.LeeSutoTriaxialNFWPotential`.

.. rubric:: Functions

.. list-table::
    :widths: 45 55

    * - :class:`~galax.potential.LeeSutoTriaxialNFWPotential`
      - the potential class, with parameters ``m``, ``r_s``, ``a1``, ``a2``, ``a3``
    * - ``potential(p, xyz)``, ``density(p, xyz)``
      - the potential and the density
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

.. rubric:: References

Lee & Suto (2003) give the potential, in their Eq. 19 with the functions of
their Appendix A, and the class docstring also cites Jing & Suto (2002) for the
triaxial NFW density. The reference list is in the docstring of
:class:`~galax.potential.LeeSutoTriaxialNFWPotential`.
"""

__all__ = [
    "LeeSutoTriaxialNFWPotential",
    "density",
    "gradient",
    "hessian",
    "laplacian",
    "potential",
]

from ._src.builtin.nfw.leesuto import (
    LeeSutoTriaxialNFWPotential,
    density,
    gradient,
    hessian,
    laplacian,
    potential,
)
