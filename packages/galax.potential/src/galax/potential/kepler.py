r"""Kepler potential.

The Kepler potential is that of a point mass. All the mass ``m_tot`` is at the
origin, so the model is spherical, the enclosed mass is the same at every radius
and there is no length scale. The ``density`` function returns 0 for :math:`r >
0` and ``inf`` at :math:`r = 0` when ``m_tot`` is not zero, which stands in for
the delta function.

The potential is

.. math::

    \Phi(r) = -\frac{G m_{\mathrm{tot}}}{r},

where :math:`r` is the distance from the point mass and :math:`G` is the
gravitational constant. The full expressions, limits and references are in
:class:`~galax.potential.KeplerPotential`.

.. rubric:: Functions

.. list-table::
    :widths: 45 55

    * - :class:`~galax.potential.KeplerPotential`
      - the potential class, with parameters ``m_tot``
    * - ``potential(p, r)``, ``density(p, r)``, ``laplacian(p, r)``
      - the potential, the density and the Laplacian of the potential
    * - ``mass_enclosed(p, r)``
      - the mass inside radius ``r``
    * - ``dpotential_dr(p, r)``, ``d2potential_dr2(p, r)``
      - the first and second radial derivatives of the potential
    * - ``gradient(p, xyz)``, ``hessian(p, xyz)``
      - the gradient, shape ``(3,)``, and the Hessian, shape ``(3, 3)``
    * - ``point_mass_potential(G, m, r)``
      - the potential energy :math:`-G m / r` of a point mass, from the constant
        ``G``, the mass ``m`` and the radius ``r``

.. rubric:: Units and shapes

The functions take bare arrays in the unit system of the potential, and
``pot.params(t)`` returns the dictionary ``p`` in that system, ``G`` included.
They evaluate one point at a time, so ``r`` is a scalar and ``xyz`` has shape
``(3,)``; use ``jax.vmap`` to evaluate many points. The page
:doc:`/potential_modules` describes the contract, the units, the rule at the
origin and the float32 limits.

.. rubric:: References

Binney & Tremaine (2008) treat bound orbits in this potential. The reference
list is in the docstring of :class:`~galax.potential.KeplerPotential`.
"""

__all__ = [
    "KeplerPotential",
    "d2potential_dr2",
    "density",
    "dpotential_dr",
    "gradient",
    "hessian",
    "laplacian",
    "mass_enclosed",
    "point_mass_potential",
    "potential",
]

from ._src.builtin.kepler import (
    KeplerPotential,
    d2potential_dr2,
    density,
    dpotential_dr,
    gradient,
    hessian,
    laplacian,
    mass_enclosed,
    point_mass_potential,
    potential,
)
