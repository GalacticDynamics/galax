r"""Isochrone potential.

The isochrone model is spherical, with a constant-density core and a finite
total mass. The potential is harmonic near the centre and approaches the
point-mass potential far from it, and the scale radius ``r_s`` marks the
transition between the two. The parameter ``m_tot`` is the total mass. The model
is called isochrone because the radial period of an orbit in it depends only on
the energy of the orbit.

The potential is

.. math::

    \Phi(r) = -\frac{G m_{\mathrm{tot}}}{r_s + \sqrt{r^2 + r_s^2}},

where :math:`r` is the distance from the centre and :math:`G` is the
gravitational constant. The full expressions, limits and references are in
:class:`~galax.potential.IsochronePotential`.

.. rubric:: Functions

.. list-table::
    :widths: 45 55

    * - :class:`~galax.potential.IsochronePotential`
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

Hénon (1959) introduced the model. The reference list is in the docstring of
:class:`~galax.potential.IsochronePotential`.
"""

__all__ = [
    "IsochronePotential",
    "d2potential_dr2",
    "density",
    "dpotential_dr",
    "gradient",
    "hessian",
    "laplacian",
    "mass_enclosed",
    "potential",
]

from ._src.builtin.isochrone import (
    IsochronePotential,
    d2potential_dr2,
    density,
    dpotential_dr,
    gradient,
    hessian,
    laplacian,
    mass_enclosed,
    potential,
)
