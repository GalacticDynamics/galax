r"""Triaxial Gaussian potential.

The triaxial Gaussian model is a mass distribution of finite total mass, with a
density that is a Gaussian in the ellipsoidal radius :math:`\xi`. The parameter
``m_tot`` is the total mass, ``r_s`` the width along :math:`x`, and ``q1`` and
``q2`` the axis ratios :math:`y / x` and :math:`z / x`. The density is finite and
positive everywhere. The potential has no closed form and is a one-dimensional
integral.

The density is

.. math::

    \rho(\mathbf{x}) = \rho_0 \exp\left(-\frac{\xi^2}{2 r_s^2}\right),
    \qquad \rho_0 = \frac{m_{\mathrm{tot}}}{q_1 q_2 (2 \pi)^{3/2} r_s^3},

where :math:`\xi^2 = x^2 + y^2 / q_1^2 + z^2 / q_2^2`. The full expressions,
limits and references are in :class:`~galax.potential.TriaxialGaussianPotential`.

.. rubric:: Functions

.. list-table::
    :widths: 45 55

    * - :class:`~galax.potential.TriaxialGaussianPotential`
      - the potential class, with parameters ``m_tot``, ``r_s``, ``q1``, ``q2``
    * - ``potential(p, xyz, *, integration_order=50)``
      - the potential, by Gauss-Legendre quadrature of order ``integration_order``
    * - ``density(p, xyz)``, ``laplacian(p, xyz)``
      - the density and the Laplacian of the potential, ``4 pi G`` times the density
    * - ``rho0_of_m(p)``
      - the central density from the mass parameter
    * - ``gradient(p, xyz)``, ``hessian(p, xyz)``
      - the gradient, shape ``(3,)``, and the Hessian, shape ``(3, 3)``,
        by automatic differentiation of ``potential``

``gradient`` and ``hessian`` also take the keyword-only ``integration_order``.

.. rubric:: Units and shapes

The functions take bare arrays in the unit system of the potential, and
``pot.params(t)`` returns the dictionary ``p`` in that system, ``G`` included.
They evaluate one point at a time, so ``xyz`` has shape ``(3,)``; use
``jax.vmap`` to evaluate many points. The page :doc:`/potential_modules`
describes the contract, the units and the float32 limits. The order of the
quadrature is a keyword-only static argument, so it is not traced by ``jax.jit``.

.. rubric:: References

Chandrasekhar (1969) and Merritt & Fridman (1996) give the integral for the
potential. The reference list is in the docstring of
:class:`~galax.potential.TriaxialGaussianPotential`.
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
