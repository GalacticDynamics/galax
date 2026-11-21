r"""NFW potential.

The NFW model is a spherical halo profile whose density falls as :math:`r^{-1}`
inside the scale radius ``r_s`` and as :math:`r^{-3}` outside it, so the
enclosed mass grows logarithmically without bound and the total mass is
infinite. The density diverges at the centre, and the potential is finite there.
The parameter ``m`` is a mass scale, :math:`m = 4 \pi \rho_s r_s^3`, and it is
not the total mass.

With :math:`x = r / r_s` the density is

.. math::

    \rho(r) = \frac{\rho_s}{x (1 + x)^2},\qquad \rho_s = \frac{m}{4 \pi r_s^3},

where :math:`r` is the distance from the centre. The full expressions, limits
and references are in :class:`~galax.potential.NFWPotential`.

.. rubric:: Functions

.. list-table::
    :widths: 45 55

    * - :class:`~galax.potential.NFWPotential`
      - the potential class, with parameters ``m``, ``r_s``
    * - ``potential(p, r)``, ``density(p, r)``, ``laplacian(p, r)``
      - the potential, the density and the Laplacian of the potential
    * - ``mass_enclosed(p, r)``
      - the mass inside radius ``r``
    * - ``dpotential_dr(p, r)``, ``d2potential_dr2(p, r)``
      - the first and second radial derivatives of the potential
    * - ``gradient(p, xyz)``, ``hessian(p, xyz)``
      - the gradient, shape ``(3,)``, and the Hessian, shape ``(3, 3)``
    * - ``rho0_of_m(p)``
      - the characteristic density :math:`\rho_s = m / (4 \pi r_s^3)`, from
        ``p["m"]`` and ``p["r_s"]``
    * - ``m_of_rho0(p)``
      - the characteristic mass :math:`m = 4 \pi \rho_s r_s^3`, from ``p["rho0"]``
        and ``p["r_s"]``

.. rubric:: Units and shapes

The functions take bare arrays in the unit system of the potential, and
``pot.params(t)`` returns the dictionary ``p`` in that system, ``G`` included.
They evaluate one point at a time, so ``r`` is a scalar and ``xyz`` has shape
``(3,)``; use ``jax.vmap`` to evaluate many points. The page
:doc:`/potential_modules` describes the contract, the units, the rule at the
origin and the float32 limits.

.. rubric:: References

Navarro, Frenk & White (1996, 1997) found that this profile fits the halos in
cold dark matter simulations. The reference list is in the docstring of
:class:`~galax.potential.NFWPotential`.
"""

__all__ = [
    "NFWPotential",
    "d2potential_dr2",
    "density",
    "dpotential_dr",
    "gradient",
    "hessian",
    "laplacian",
    "m_of_rho0",
    "mass_enclosed",
    "potential",
    "rho0_of_m",
]

from ._src.builtin.nfw.base import (
    NFWPotential,
    d2potential_dr2,
    density,
    dpotential_dr,
    gradient,
    hessian,
    laplacian,
    m_of_rho0,
    mass_enclosed,
    potential,
    rho0_of_m,
)
