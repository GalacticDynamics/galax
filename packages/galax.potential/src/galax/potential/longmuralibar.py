r"""Long & Murali bar potential.

The Long & Murali model describes a triaxial bar of finite total mass. It is
built by smoothing a thin uniform needle of length :math:`2a` along the bar axis
with the Miyamoto-Nagai density, so the potential is an elementary function and
the density is positive everywhere. The bar lies in the :math:`x`-:math:`y`
plane with its long axis at the position angle ``alpha`` from the :math:`x`
axis. The parameter ``m_tot`` is the total mass, ``a`` the half-length of the
needle, ``b`` the softening length in the plane perpendicular to the bar axis
and ``c`` the vertical softening length. The bar angle is fixed in time.

The potential is the Miyamoto-Nagai potential averaged along the needle,

.. math::

    \Phi = -\frac{G m_{\mathrm{tot}}}{2a} \int_{-a}^{a}
    \frac{\mathrm{d}s}{\sqrt{(x' - s)^2 + \eta^2}},

where :math:`x' = x \cos\alpha + y \sin\alpha` and :math:`y' = -x \sin\alpha + y
\cos\alpha` are the coordinates in the frame of the bar, :math:`\eta^2 = y'^2 +
(b + \zeta)^2` and :math:`\zeta = \sqrt{c^2 + z^2}`. The integral has a closed
form in terms of a logarithm. The full expressions, limits and references are in
:class:`~galax.potential.LongMuraliBarPotential`.

.. rubric:: Functions

.. list-table::
    :widths: 45 55

    * - :class:`~galax.potential.LongMuraliBarPotential`
      - the potential class, with parameters ``m_tot``, ``a``, ``b``, ``c``,
        ``alpha``
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

Long & Murali (1992) introduced the model. The potential is their Eq. 8b. The
reference list is in the docstring of
:class:`~galax.potential.LongMuraliBarPotential`.
"""

__all__ = [
    "LongMuraliBarPotential",
    "density",
    "gradient",
    "hessian",
    "laplacian",
    "potential",
]

from ._src.builtin.longmurali import (
    LongMuraliBarPotential,
    density,
    gradient,
    hessian,
    laplacian,
    potential,
)
