r"""Satoh potential.

The Satoh model is a flattened, axisymmetric mass distribution of finite total
mass, symmetric about the plane :math:`z = 0`. The parameter ``m_tot`` is the
total mass, ``a`` the scale length and ``b`` the scale height. The density is
finite and positive everywhere. The potential tends to the Kuzmin disk for
:math:`b \to 0` and to the point-mass potential for :math:`a \to 0`.

The potential is

.. math::

    \Phi(R, z) = -\frac{G m_{\mathrm{tot}}}{\sqrt{R^2 + z^2 + a(a + 2\zeta)}},
    \qquad \zeta = \sqrt{z^2 + b^2},

where :math:`R^2 = x^2 + y^2` and :math:`G` is the gravitational constant. The
full expressions, limits and references are in
:class:`~galax.potential.SatohPotential`.

.. rubric:: Functions

.. list-table::
    :widths: 45 55

    * - :class:`~galax.potential.SatohPotential`
      - the potential class, with parameters ``m_tot``, ``a``, ``b``
    * - ``potential(p, xyz)``, ``density(p, xyz)``
      - the potential and the density
    * - ``laplacian(p, xyz)``
      - the Laplacian of the potential, which is :math:`4 \pi G` times the density
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

.. rubric:: References

Satoh (1980) proposed the model for axisymmetric galaxies from spirals to
ellipticals and used it to model NGC 4697. The potential and the density are the
pair of his Eq. 8. The reference list is in the docstring of
:class:`~galax.potential.SatohPotential`.
"""

__all__ = [
    "SatohPotential",
    "density",
    "gradient",
    "hessian",
    "laplacian",
    "potential",
]

from ._src.builtin.satoh import (
    SatohPotential,
    density,
    gradient,
    hessian,
    laplacian,
    potential,
)
