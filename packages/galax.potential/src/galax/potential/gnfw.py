r"""Generalized NFW potential.

The generalized NFW model is a spherical double power-law profile with the NFW
outer slope and a free inner slope. The density falls as :math:`r^{-\gamma}`
inside the scale radius ``r_s`` and as :math:`r^{-3}` outside it, so the total
mass is infinite. The parameter ``gamma`` is the inner slope, with ``gamma = 1``
the NFW profile, and ``m`` is a mass scale and not the total mass. The functions
are written for :math:`0 \le \gamma < 2`, where the central potential is finite.

With :math:`x = r / r_s` the density is

.. math::

    \rho(r) = \rho_s\, x^{-\gamma} (1 + x)^{\gamma - 3},
    \qquad \rho_s = \frac{m}{4 \pi r_s^3},

where :math:`r` is the distance from the centre. The full expressions, limits
and references are in :class:`~galax.potential.gNFWPotential`.

.. rubric:: Functions

.. list-table::
    :widths: 45 55

    * - :class:`~galax.potential.gNFWPotential`
      - the potential class, with parameters ``m``, ``r_s``, ``gamma``
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

The profile is the member :math:`(\alpha, \beta, \gamma) = (1, 3, \gamma)` of
the double power-law family of Zhao (1996), and :math:`\gamma = 1` is the NFW
profile of Navarro, Frenk & White (1996). The reference list is in the docstring
of :class:`~galax.potential.gNFWPotential`.
"""

__all__ = [
    "gNFWPotential",
    "d2potential_dr2",
    "density",
    "dpotential_dr",
    "gradient",
    "hessian",
    "laplacian",
    "mass_enclosed",
    "potential",
]

from ._src.builtin.nfw.generalized import (
    d2potential_dr2,
    density,
    dpotential_dr,
    gNFWPotential,
    gradient,
    hessian,
    laplacian,
    mass_enclosed,
    potential,
)
