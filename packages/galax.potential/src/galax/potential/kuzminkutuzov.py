r"""Kuzmin-Kutuzov Staeckel potential.

The Kuzmin-Kutuzov potential is axisymmetric, symmetric under :math:`z \to -z`,
and of Staeckel form in spheroidal coordinates, so every orbit in it has three
exact integrals of motion. The mass distribution is stratified approximately on
spheroids, oblate for :math:`c < a` and prolate for :math:`c > a`, with a finite
total mass ``m_tot`` and a density that falls as :math:`r^{-4}` at large radii.
The parameters ``a`` and ``c`` are the scale lengths in the plane and along
:math:`z`, and the case :math:`c = a` is the isochrone potential.

In cylindrical coordinates :math:`(R, z)` the potential is

.. math::

    \Phi(R, z) = -\frac{G m_{\mathrm{tot}}}
        {\sqrt{R^2 + z^2 + a^2 + c^2 + 2\sqrt{c^2 R^2 + a^2 z^2 + a^2 c^2}}},

where :math:`G` is the gravitational constant. The full expressions, limits and
references are in :class:`~galax.potential.KuzminKutuzovPotential`.

.. rubric:: Functions

.. list-table::
    :widths: 45 55

    * - :class:`~galax.potential.KuzminKutuzovPotential`
      - the potential class, with parameters ``m_tot``, ``a``, ``c``
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

The model was introduced by Kuzmin (1956) and studied by Kuzmin & Kutuzov (1962)
and Dejonghe & de Zeeuw (1988). It is the ``KuzminKutuzovStaeckelPotential`` of
galpy with ``ac = a / c``. The reference list is in the docstring of
:class:`~galax.potential.KuzminKutuzovPotential`.
"""

__all__ = [
    "KuzminKutuzovPotential",
    "density",
    "gradient",
    "hessian",
    "laplacian",
    "potential",
]

from ._src.builtin.kuzminkutuzov import (
    KuzminKutuzovPotential,
    density,
    gradient,
    hessian,
    laplacian,
    potential,
)
