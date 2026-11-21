r"""Zhao (1996) double power-law potential.

The Zhao model is spherical, with a density that is a double power law. It falls
as :math:`r^{-\gamma}` well inside the scale radius ``r_s`` and as
:math:`r^{-\beta}` well outside it, and ``alpha`` sets the width of the
transition between the two. The parameter ``m`` is the mass inside ``r_s``. The
total mass is finite for :math:`\beta > 3`, and the family contains several
common profiles. The potential and the enclosed mass are closed forms in the
incomplete beta function.

With :math:`u = r / r_s` the density is

.. math::

    \rho(r) = \frac{\rho_s}{u^{\gamma}
        \left(1 + u^{1/\alpha}\right)^{(\beta - \gamma)\alpha}},

where :math:`r` is the distance from the centre and :math:`\rho_s` is fixed by
the mass ``m``. The full expressions, limits and references are in
:class:`~galax.potential.ZhaoPotential`.

.. rubric:: Functions

.. list-table::
    :widths: 45 55

    * - :class:`~galax.potential.ZhaoPotential`
      - the potential class, with parameters ``m``, ``r_s``, ``alpha``, ``beta``,
        ``gamma``
    * - ``potential(p, r)``, ``density(p, r)``, ``laplacian(p, r)``
      - the potential, the density and the Laplacian of the potential
    * - ``mass_enclosed(p, r)``
      - the mass inside radius ``r``
    * - ``dpotential_dr(p, r)``, ``d2potential_dr2(p, r)``
      - the first and second radial derivatives of the potential
    * - ``gradient(p, xyz)``, ``hessian(p, xyz)``
      - the gradient, shape ``(3,)``, and the Hessian, shape ``(3, 3)``

.. rubric:: Units and shapes

The functions take bare arrays in the unit system of the potential, and
``pot.params(t)`` returns the dictionary ``p`` in that system, ``G`` included.
They evaluate one point at a time, so ``r`` is a scalar and ``xyz`` has shape
``(3,)``; use ``jax.vmap`` to evaluate many points. The page
:doc:`/potential_modules` describes the contract, the units, the rule at the
origin and the float32 limits.

.. rubric:: References

Zhao (1996, MNRAS 278, 488) gives the density and the potential of the family.
The reference list is in the docstring of
:class:`~galax.potential.ZhaoPotential`.
"""

__all__ = [
    "ZhaoPotential",
    "d2potential_dr2",
    "density",
    "dpotential_dr",
    "gradient",
    "hessian",
    "laplacian",
    "mass_enclosed",
    "potential",
]

from ._src.builtin.zhao import (
    ZhaoPotential,
    d2potential_dr2,
    density,
    dpotential_dr,
    gradient,
    hessian,
    laplacian,
    mass_enclosed,
    potential,
)
