r"""Monari et al. (2016) bar potential.

The model is a rigidly rotating bar that adds a quadrupole perturbation to an
axisymmetric Galaxy potential. The perturbation is a pure :math:`l = 2`,
:math:`m = 2` quadrupole whose radial profile is a cubic inside the bar length
``Rb`` and an :math:`r^{-3}` tail outside. The parameter ``alpha`` is the bar
strength, ``R0`` the Galactocentric radius of the Sun, ``v0`` the circular
velocity at ``R0``, ``phi_b`` the bar angle at :math:`t = 0` and ``Omega`` the
pattern speed. The density follows from Poisson's equation, vanishes outside
``Rb`` and has zero total mass.

The potential is

.. math::

    \Phi = A_b \, U(r / R_b) \sin^2\theta \, \cos\gamma_b,
    \qquad A_b = \frac{\alpha v_0^2}{3} \left(\frac{R_0}{R_b}\right)^3,

with the bar angle :math:`\gamma_b = 2(\phi - \phi_b - \Omega t)`, where
:math:`r` is the spherical radius, :math:`\theta` the polar angle and
:math:`\phi` the azimuth. The radial function is :math:`U(s) = s^3 - 2` for
:math:`s < 1` and :math:`U(s) = -s^{-3}` for :math:`s \ge 1`. The full
expressions, limits and references are in
:class:`~galax.potential.MonariEtAl2016BarPotential`.

.. rubric:: Functions

.. list-table::
    :widths: 45 55

    * - :class:`~galax.potential.MonariEtAl2016BarPotential`
      - the potential class, with parameters ``alpha``, ``R0``, ``v0``, ``Rb``,
        ``phi_b``, ``Omega``
    * - ``potential(p, xyz, t)``, ``density(p, xyz, t)``
      - the potential and the density
    * - ``laplacian(p, xyz, t)``
      - the Laplacian of the potential, which is :math:`4 \pi G` times the density
    * - ``gradient(p, xyz, t)``, ``hessian(p, xyz, t)``
      - the gradient, shape ``(3,)``, and the Hessian, shape ``(3, 3)``,
        by automatic differentiation of ``potential``

.. rubric:: Units and shapes

The functions take bare arrays in the unit system of the potential, and
``pot.params(t)`` returns the dictionary ``p`` in that system, ``G`` included.
They evaluate one point at a time, so ``xyz`` has shape ``(3,)`` and the time
``t`` is a scalar; use ``jax.vmap`` to evaluate many points. The page
:doc:`/potential_modules` describes the contract, the units, the rule at the
origin and the float32 limits.

.. rubric:: References

The model is the three-dimensional version of the quadrupole bar of Dehnen
(2000), as written by Monari et al. (2016). The reference list is in the
docstring of :class:`~galax.potential.MonariEtAl2016BarPotential`.
"""

__all__ = [
    "MonariEtAl2016BarPotential",
    "density",
    "gradient",
    "hessian",
    "laplacian",
    "potential",
]

from ._src.builtin.monari2016 import (
    MonariEtAl2016BarPotential,
    density,
    gradient,
    hessian,
    laplacian,
    potential,
)
