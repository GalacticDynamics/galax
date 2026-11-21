r"""Burkert potential.

The Burkert model is spherical, with a flat core and an infinite total mass. The
density is finite and nearly flat inside the core radius ``r_s`` and falls as
:math:`r^{-3}` far out, so the enclosed mass grows logarithmically without
bound. The parameter ``m`` is the mass inside ``r_s``.

The density is

.. math::

    \rho(r) = \frac{m}{\pi C}\frac{1}{(r + r_s)(r^2 + r_s^2)},
    \qquad C = 3 \ln 2 - \frac{\pi}{2},

where :math:`r` is the distance from the centre. The full expressions, limits
and references are in :class:`~galax.potential.BurkertPotential`.

.. rubric:: Functions

.. list-table::
    :widths: 45 55

    * - :class:`~galax.potential.BurkertPotential`
      - the potential class, with parameters ``m``, ``r_s``
    * - ``potential(p, r)``, ``density(p, r)``, ``laplacian(p, r)``
      - the potential, the density and the Laplacian of the potential
    * - ``mass_enclosed(p, r)``
      - the mass inside radius ``r``
    * - ``dpotential_dr(p, r)``, ``d2potential_dr2(p, r)``
      - the first and second radial derivatives of the potential
    * - ``gradient(p, xyz)``, ``hessian(p, xyz)``
      - the gradient, shape ``(3,)``, and the Hessian, shape ``(3, 3)``
    * - ``rho0(m, r_s)``
      - the central density :math:`m / (\pi C r_s^3)`, from the mass inside ``r_s``
        and the core radius
    * - ``mass_total(p)``
      - the total mass, infinite for ``m > 0`` and 0 otherwise

.. rubric:: Units and shapes

The functions take bare arrays in the unit system of the potential, and
``pot.params(t)`` returns the dictionary ``p`` in that system, ``G`` included.
They evaluate one point at a time, so ``r`` is a scalar and ``xyz`` has shape
``(3,)``; use ``jax.vmap`` to evaluate many points. The page
:doc:`/potential_modules` describes the contract, the units, the rule at the
origin and the float32 limits.

.. rubric:: References

Burkert (1995) introduced the profile as a phenomenological fit to the halos of
dwarf spiral galaxies. The reference list is in the docstring of
:class:`~galax.potential.BurkertPotential`.
"""

__all__ = [
    "BurkertPotential",
    "d2potential_dr2",
    "density",
    "dpotential_dr",
    "gradient",
    "hessian",
    "laplacian",
    "mass_enclosed",
    "mass_total",
    "potential",
    "rho0",
]

from ._src.builtin.burkert import (
    BurkertPotential,
    d2potential_dr2,
    density,
    dpotential_dr,
    gradient,
    hessian,
    laplacian,
    mass_enclosed,
    mass_total,
    potential,
    rho0,
)
