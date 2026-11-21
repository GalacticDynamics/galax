r"""Stone and Ostriker (2015) potential.

The model is spherical, with a constant-density core of radius ``r_c``, a
roughly isothermal zone with :math:`\rho \propto r^{-2}` for :math:`r_c \ll r
\ll r_h`, and a density that falls as :math:`r^{-4}` beyond the halo radius
``r_h``, so the total mass ``m_tot`` is finite. The formulas divide by
:math:`r_h - r_c`, so ``r_c`` and ``r_h`` must differ.

The density is

.. math::

    \rho(r) = \frac{\rho_c}{\left(1 + r^2 / r_c^2\right)\left(1 + r^2 / r_h^2\right)},
    \qquad \rho_c = \frac{m_{\mathrm{tot}} (r_h + r_c)}{2 \pi^2 r_c^2 r_h^2},

where :math:`r` is the distance from the centre and :math:`\rho_c` is the
central density. The full expressions, limits and references are in
:class:`~galax.potential.StoneOstriker15Potential`.

.. rubric:: Functions

.. list-table::
    :widths: 45 55

    * - :class:`~galax.potential.StoneOstriker15Potential`
      - the potential class, with parameters ``m_tot``, ``r_c``, ``r_h``
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

Stone & Ostriker (2015) introduced the model as a potential-density pair for
star clusters with nearly isothermal interiors. The reference list is in the
docstring of :class:`~galax.potential.StoneOstriker15Potential`.
"""

__all__ = [
    "StoneOstriker15Potential",
    "d2potential_dr2",
    "density",
    "dpotential_dr",
    "gradient",
    "hessian",
    "laplacian",
    "mass_enclosed",
    "potential",
]

from ._src.builtin.stoneostriker15 import (
    StoneOstriker15Potential,
    d2potential_dr2,
    density,
    dpotential_dr,
    gradient,
    hessian,
    laplacian,
    mass_enclosed,
    potential,
)
