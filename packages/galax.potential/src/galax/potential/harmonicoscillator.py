r"""Harmonic oscillator potential.

The potential is :math:`\Phi = \frac{1}{2} \sum_i (\omega_i q_i)^2`, where
``omega`` is a scalar shared by the three axes or a vector with one frequency
per axis. The Laplacian is :math:`\sum_i \omega_i^2` and the density is that
divided by :math:`4 \pi G`, both uniform in space. The density needs ``G`` in
``p`` and raises a ``ValueError`` without it, and the Laplacian does not.

.. rubric:: Functions

.. list-table::
    :widths: 45 55

    * - :class:`~galax.potential.HarmonicOscillatorPotential`
      - the potential class, with parameters ``omega``
    * - ``potential(p, xyz)``, ``density(p, xyz)``
      - the potential and the density
    * - ``laplacian(p, xyz)``
      - the Laplacian of the potential, :math:`\sum_i \omega_i^2`
    * - ``gradient(p, xyz)``, ``hessian(p, xyz)``
      - the gradient, shape ``(3,)``, and the Hessian, shape ``(3, 3)``,
        by automatic differentiation of ``potential``

.. rubric:: Units and shapes

The functions take bare arrays in the unit system of the potential, and
``pot.params(t)`` returns the dictionary ``p`` in that system, ``G`` included.
They evaluate one point at a time, so ``xyz`` has shape ``(3,)``; use
``jax.vmap`` to evaluate many points. The page :doc:`/potential_modules`
describes the contract, the units, the rule at the origin and the float32
limits.

The full expressions are in :class:`~galax.potential.HarmonicOscillatorPotential`.
"""

__all__ = [
    "HarmonicOscillatorPotential",
    "density",
    "gradient",
    "hessian",
    "laplacian",
    "potential",
]

from ._src.builtin.harmonicoscillator import (
    HarmonicOscillatorPotential,
    density,
    gradient,
    hessian,
    laplacian,
    potential,
)
