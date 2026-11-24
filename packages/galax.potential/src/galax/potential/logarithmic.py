r"""Logarithmic potential.

The model is spherical, with a core and a rotation curve that is flat far out.
The parameter ``v_c`` is the asymptotic circular speed and ``r_s`` is the scale
length of the core. The potential is

.. math::

    \Phi(r) = \frac{1}{2} v_c^2 \ln\left(r_s^2 + r^2\right)

and the density is

.. math::

    \rho(r) = \frac{v_c^2 (3 r_s^2 + r^2)}{4 \pi G (r_s^2 + r^2)^2}.

The mass grows linearly with :math:`r` at large radius, so there is no finite
total mass. The full expressions and the reference are in
:class:`~galax.potential.LogarithmicPotential`.

.. rubric:: Functions

.. list-table::
    :widths: 45 55

    * - :class:`~galax.potential.LogarithmicPotential`
      - the potential class, with parameters ``v_c``, ``r_s``
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

In float32 with SI units :math:`r_s^2` overflows, so use float64 or a unit
system in which typical values are of order 1.
"""

__all__ = [
    "LogarithmicPotential",
    "d2potential_dr2",
    "density",
    "dpotential_dr",
    "gradient",
    "hessian",
    "laplacian",
    "mass_enclosed",
    "potential",
]

from ._src.builtin.logarithmic import (
    LogarithmicPotential,
    d2potential_dr2,
    density,
    dpotential_dr,
    gradient,
    hessian,
    laplacian,
    mass_enclosed,
    potential,
)
