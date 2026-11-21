r"""Hernquist potential.

The Hernquist model is spherical, with a central density cusp and a finite total
mass. The density goes as :math:`r^{-1}` well inside the scale radius ``r_s``
and as :math:`r^{-4}` well outside it. The parameter ``m_tot`` is the total
mass.

The density is

.. math::

    \rho(r) = \frac{m_{\mathrm{tot}}\, r_s}{2 \pi\, r\, (r + r_s)^3},

where :math:`r` is the distance from the centre. The density diverges at the
origin, while the potential and the gradient stay finite. The full expressions,
limits and references are in :class:`~galax.potential.HernquistPotential`.

.. rubric:: Functions

.. list-table::
    :widths: 45 55

    * - :class:`~galax.potential.HernquistPotential`
      - the potential class, with parameters ``m_tot``, ``r_s``
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

Hernquist (1990) introduced the model as an analytic description of spherical
galaxies and bulges whose projected profile is close to the de Vaucouleurs
:math:`R^{1/4}` law. The reference list is in the docstring of
:class:`~galax.potential.HernquistPotential`.
"""

__all__ = [
    "HernquistPotential",
    "d2potential_dr2",
    "density",
    "dpotential_dr",
    "gradient",
    "hessian",
    "laplacian",
    "mass_enclosed",
    "potential",
]

from ._src.builtin.hernquist import (
    HernquistPotential,
    d2potential_dr2,
    density,
    dpotential_dr,
    gradient,
    hessian,
    laplacian,
    mass_enclosed,
    potential,
)
