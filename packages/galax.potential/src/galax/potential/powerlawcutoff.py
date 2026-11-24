r"""Power-law cutoff potential.

The model is a spherical density that falls as :math:`r^{-\alpha}` at small
radii and is cut off by a Gaussian factor beyond the cutoff radius ``r_c``. The
mass is finite and equal to the parameter ``m_tot``, and the power-law index
``alpha`` satisfies :math:`0 \le \alpha < 3`. The form is the one used for the
bulge of the Milky Way model ``MWPotential2014`` in galpy.

With :math:`s = r / r_c` and :math:`a = (3 - \alpha) / 2` the density is

.. math::

    \rho(r) = \frac{m_{\mathrm{tot}}}{2 \pi \Gamma(a) r_c^3}
    \left(\frac{r_c}{r}\right)^{\alpha} e^{-s^2},

where :math:`r` is the distance from the centre and :math:`\Gamma` is the gamma
function. The full expressions, limits and references are in
:class:`~galax.potential.PowerLawCutoffPotential`.

.. rubric:: Functions

.. list-table::
    :widths: 45 55

    * - :class:`~galax.potential.PowerLawCutoffPotential`
      - the potential class, with parameters ``m_tot``, ``alpha``, ``r_c``
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

The parameters are those of the ``PowerLawCutoffPotential`` of Gala
(Price-Whelan 2017), and the functional form is the one of the
``PowerSphericalPotentialwCutoff`` of galpy (Bovy 2015). The reference list is
in the docstring of :class:`~galax.potential.PowerLawCutoffPotential`.
"""

__all__ = [
    "PowerLawCutoffPotential",
    "d2potential_dr2",
    "density",
    "dpotential_dr",
    "gradient",
    "hessian",
    "laplacian",
    "mass_enclosed",
    "potential",
]

from ._src.builtin.powerlawcutoff import (
    PowerLawCutoffPotential,
    d2potential_dr2,
    density,
    dpotential_dr,
    gradient,
    hessian,
    laplacian,
    mass_enclosed,
    potential,
)
