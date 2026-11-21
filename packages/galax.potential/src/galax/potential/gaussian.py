r"""Gaussian potential.

The Gaussian model is spherical, with a density that is a Gaussian in radius.
The density is finite at the centre, falls faster than any power law, and
integrates to the finite total mass ``m_tot``. The parameter ``r_s`` is the
standard deviation of the Gaussian along each Cartesian axis. The potential and
the enclosed mass have closed forms.

The density is

.. math::

    \rho(r) = \rho_0 \exp\left(-\frac{r^2}{2 r_s^2}\right),
    \qquad \rho_0 = \frac{m_{\mathrm{tot}}}{(2 \pi)^{3/2} r_s^3},

where :math:`r` is the distance from the centre and :math:`\rho_0` is the
central density. The full expressions, limits and references are in
:class:`~galax.potential.GaussianPotential`.

.. rubric:: Functions

.. list-table::
    :widths: 45 55

    * - :class:`~galax.potential.GaussianPotential`
      - the potential class, with parameters ``m_tot``, ``r_s``
    * - ``potential(p, r)``, ``density(p, r)``, ``laplacian(p, r)``
      - the potential, the density and the Laplacian of the potential
    * - ``mass_enclosed(p, r)``
      - the mass inside radius ``r``
    * - ``dpotential_dr(p, r)``, ``d2potential_dr2(p, r)``
      - the first and second radial derivatives of the potential
    * - ``gradient(p, xyz)``, ``hessian(p, xyz)``
      - the gradient, shape ``(3,)``, and the Hessian, shape ``(3, 3)``
    * - ``rho0_of_m(p)``
      - the central density :math:`\rho_0`, from ``p["m_tot"]`` and ``p["r_s"]``
    * - ``m_of_rho0(p)``
      - the total mass, from ``p["rho0"]`` and ``p["r_s"]``

.. rubric:: Units and shapes

The functions take bare arrays in the unit system of the potential, and
``pot.params(t)`` returns the dictionary ``p`` in that system, ``G`` included.
They evaluate one point at a time, so ``r`` is a scalar and ``xyz`` has shape
``(3,)``; use ``jax.vmap`` to evaluate many points. The page
:doc:`/potential_modules` describes the contract, the units, the rule at the
origin and the float32 limits.

.. rubric:: References

The model is not attributed to a particular paper, so the class docstring has no
reference list. The reference list is in the docstring of
:class:`~galax.potential.GaussianPotential`.
"""

__all__ = [
    "GaussianPotential",
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

from ._src.builtin.gaussian.base import (
    GaussianPotential,
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
