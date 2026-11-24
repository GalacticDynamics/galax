r"""Triaxial Gaussian potential.

The density is a Gaussian in the ellipsoidal radius, with total mass ``m_tot``,
scale radius ``r_s`` and axis ratios ``q1`` (y to x) and ``q2`` (z to x),

.. math::

    \rho(\mathbf{q}) = \frac{m_{\mathrm{tot}}}{q_1 q_2 (2 \pi)^{3/2} r_s^3}
    \exp\left(-\frac{x^2 + y^2 / q_1^2 + z^2 / q_2^2}{2 r_s^2}\right).

The potential has no closed form and is a one-dimensional integral
(Chandrasekhar 1969; Merritt & Fridman 1996), evaluated by Gauss-Legendre
quadrature. The full expressions and the references are in
:class:`~galax.potential.TriaxialGaussianPotential`.

.. rubric:: Functions

.. list-table::
    :widths: 45 55

    * - :class:`~galax.potential.TriaxialGaussianPotential`
      - the potential class, with parameters ``m_tot``, ``r_s``, ``q1``, ``q2``
    * - ``potential(p, xyz, *, integration_order=50)``
      - the potential, by Gauss-Legendre quadrature of a one-dimensional integral
    * - ``density(p, xyz)``
      - the density
    * - ``laplacian(p, xyz)``
      - the Laplacian of the potential, which is :math:`4 \pi G` times the density
    * - ``gradient(p, xyz, *, integration_order=50)``,
        ``hessian(p, xyz, *, integration_order=50)``
      - the gradient, shape ``(3,)``, and the Hessian, shape ``(3, 3)``,
        by automatic differentiation of ``potential``
    * - ``rho0_of_m(p)``
      - the central density from the total mass

.. rubric:: Units and shapes

The functions take bare arrays in the unit system of the potential, and
``pot.params(t)`` returns the dictionary ``p`` in that system, ``G`` included.
They evaluate one point at a time, so ``xyz`` has shape ``(3,)``; use
``jax.vmap`` to evaluate many points. The page :doc:`/potential_modules`
describes the contract, the units, the rule at the origin and the float32
limits.

``potential``, ``gradient`` and ``hessian`` take a keyword-only
``integration_order``, the order of the Gauss-Legendre quadrature, which
defaults to 50. It is static under ``jax.jit``, so each value compiles once.
The class passes its own ``integration_order``.
"""

__all__ = [
    "TriaxialGaussianPotential",
    "density",
    "gradient",
    "hessian",
    "laplacian",
    "potential",
    "rho0_of_m",
]

from ._src.builtin.gaussian.triaxial import (
    TriaxialGaussianPotential,
    density,
    gradient,
    hessian,
    laplacian,
    potential,
    rho0_of_m,
)
