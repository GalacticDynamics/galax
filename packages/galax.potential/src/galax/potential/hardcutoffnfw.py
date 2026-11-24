r"""Sharply truncated NFW potential.

The model is the NFW profile inside the truncation radius ``r_t`` and has zero
density outside it, so the mass is finite and the potential outside ``r_t`` is
that of a point mass. The parameter ``m`` is the characteristic mass of the
untruncated NFW profile and ``r_s`` is its scale radius. The density and the
second derivative of the potential jump at ``r_t``.

With :math:`x = r / r_s` the density is

.. math::

    \rho(r) = \begin{cases}
        \dfrac{m}{4 \pi r_s^3} \dfrac{1}{x (1 + x)^2}, & r \le r_t, \\
        0, & r > r_t,
    \end{cases}

where :math:`r` is the distance from the centre. The total mass is :math:`m
[\ln(1 + x_t) - x_t / (1 + x_t)]` with :math:`x_t = r_t / r_s`. The full
expressions, limits and references are in
:class:`~galax.potential.HardCutoffNFWPotential`.

.. rubric:: Functions

.. list-table::
    :widths: 45 55

    * - :class:`~galax.potential.HardCutoffNFWPotential`
      - the potential class, with parameters ``m``, ``r_s``, ``r_t``
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

The truncation is a modelling choice and not the result of a particular paper.
The profile inside ``r_t`` is the NFW profile of Navarro, Frenk & White (1996).
The reference list is in the docstring of
:class:`~galax.potential.HardCutoffNFWPotential`.
"""

__all__ = [
    "HardCutoffNFWPotential",
    "d2potential_dr2",
    "density",
    "dpotential_dr",
    "gradient",
    "hessian",
    "laplacian",
    "mass_enclosed",
    "potential",
]

from ._src.builtin.nfw.truncated import (
    HardCutoffNFWPotential,
    d2potential_dr2,
    density,
    dpotential_dr,
    gradient,
    hessian,
    laplacian,
    mass_enclosed,
    potential,
)
