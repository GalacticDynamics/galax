"""Rotation of a potential about the z axis at a constant pattern speed."""

__all__ = ["RotatingPotential"]

import functools as ft
from dataclasses import KW_ONLY

from typing import final

import jax

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax.potential.custom_types as gt
from .base import AbstractTransformedPotential
from galax.potential._src.base import AbstractPotential
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.symmetry import Symmetry, invariances


@final
class RotatingPotential(AbstractTransformedPotential):
    r"""Rotate a potential about the z axis at a constant pattern speed.

    The base potential is rigidly rotated about :math:`+\hat{z}` by the angle

    $$ \phi(t) = \phi_0 + \Omega t, $$

    so that

    $$ \Phi(\mathbf{x}, t) = \Phi_{\rm base}(R_z(-\phi(t))\,\mathbf{x}, t), $$

    where :math:`R_z(\phi)` is the right-handed (counter-clockwise, seen from
    :math:`+z`) rotation by :math:`\phi`. A positive ``Omega`` rotates the base
    potential counter-clockwise, i.e. prograde for a disk whose angular momentum
    points along :math:`+z`. A feature at azimuth :math:`\varphi_{\rm base}` in
    the base potential is at :math:`\varphi_{\rm base} + \phi(t)`. This is the
    same convention as `galax.potential.MonariEtAl2016BarPotential`'s ``Omega``
    and ``phi_b``.

    The gradient, density and hessian are those of the base potential, rotated:
    $\nabla\Phi = R_z(\phi)\,\nabla\Phi_{\rm base}$,
    $\rho = \rho_{\rm base}$ at the rotated position, and
    $H = R_z(\phi)\, H_{\rm base}\, R_z(\phi)^T$.
    So a base potential's own gradient or density (e.g. a multipole expansion's)
    is used, not an autodiff of the wrapped potential.

    ``Omega`` is a pattern speed with dimensions of frequency, e.g.
    ``km / (s kpc)`` or ``1 / Myr``; ``Omega * t`` is in radians. Like any
    `galax` parameter it may be a function of time, but the angle is then
    :math:`\phi_0 + \Omega(t)\, t`, not the integral of :math:`\Omega(t)`.

    See Also
    --------
    galax.potential.TransformedPotential : with
        `galax.coordinates.ops.ConstantRotationZOperator` rotates a potential
        at a fixed angular speed through a `coordinax` operator, without a
        starting angle and with the speed not a potential parameter.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    A static (non-rotating) bar, with its long axis along x:

    >>> bar = gp.LongMuraliBarPotential(m_tot=u.Q(1e10, "Msun"), a=u.Q(4, "kpc"),
    ...     b=u.Q(1, "kpc"), c=u.Q(1, "kpc"), alpha=u.Q(0, "rad"), units="galactic")

    Rotate it at 90 degrees per 100 Myr, starting at 30 degrees:

    >>> import quaxed.numpy as jnp
    >>> Omega = u.Q(jnp.pi / 2 / 100, "1/Myr")
    >>> pot = gp.RotatingPotential(bar, Omega=Omega, phi_0=u.Q(30, "deg"))
    >>> pot
    RotatingPotential(
      base_potential=LongMuraliBarPotential( ... ),
      Omega=ConstantParameter(...), phi_0=ConstantParameter(...)
    )

    At ``t = 100 Myr`` the bar has turned to 120 degrees, so the potential at
    that azimuth is the static bar's along its long axis:

    >>> x = u.Q([2.0, 0, 0], "kpc")
    >>> t = u.Q(100, "Myr")
    >>> ang = jnp.deg2rad(120)
    >>> x120 = u.Q([2.0 * jnp.cos(ang), 2.0 * jnp.sin(ang), 0], "kpc")
    >>> bool(jnp.isclose(pot.potential(x120, t).value, bar.potential(x, t).value))
    True

    The acceleration is the static bar's, rotated by the same angle:

    >>> pot.acceleration(x120, t).value.round(6)
    Array([ 0.000549, -0.000952, -0.      ], dtype=float64)
    >>> bar.acceleration(x, t).value.round(6)
    Array([-0.001099, -0.      , -0.      ], dtype=float64)

    A base potential that is invariant under rotation about z is not changed:

    >>> kep = gp.KeplerPotential(m_tot=u.Q(1e12, "Msun"), units="galactic")
    >>> rkep = gp.RotatingPotential(kep, Omega=Omega)
    >>> bool(jnp.isclose(rkep.potential(x120, t).value, kep.potential(x120, t).value))
    True
    >>> rkep.symmetry
    <Symmetry.SPHERICAL: 'spherical'>

    """

    #: the original potential
    base_potential: AbstractPotential

    _: KW_ONLY

    Omega: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="frequency",
        doc="Pattern speed about +z. Positive is counter-clockwise seen from +z.",
    )

    phi_0: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.Q(0.0, "rad"),
        dimensions="angle",
        doc="Rotation angle at ``t = 0``, measured from +x towards +y.",
    )

    @property
    def symmetry(self) -> Symmetry:
        """The base's symmetry if it is invariant under z-rotation, else none.

        A rotation about z keeps z-rotation, z-reflection and full rotation
        invariance, but not the x and y reflections of e.g. a bar.
        """
        base = self.base_potential.symmetry
        return base if "z_rotation" in invariances(base) else Symmetry.NONE

    def _rotation(self, t: gt.BBtQorVSz0, /) -> gt.BBtSz33:
        r""":math:`R_z(\phi(t))`, shape ``(*t.shape, 3, 3)``."""
        t = u.ustrip(AllowValue, self.units["time"], t)
        p = self.params(t)
        phi = p["phi_0"] + p["Omega"] * t
        c, s = jnp.cos(phi), jnp.sin(phi)
        zero, one = jnp.zeros_like(c), jnp.ones_like(c)
        return jnp.stack(  # type: ignore[no-any-return]
            [
                jnp.stack([c, -s, zero], axis=-1),
                jnp.stack([s, c, zero], axis=-1),
                jnp.stack([zero, zero, one], axis=-1),
            ],
            axis=-2,
        )

    def _to_base(
        self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /
    ) -> tuple[gt.BBtSz33, gt.BBtSz3]:
        """Return the rotation and the position in the base potential's frame."""
        xyz = u.ustrip(AllowValue, self.units["length"], xyz.astype(float))
        rot = self._rotation(t)
        # R^T x, as a row vector: x^T R.
        return rot, (xyz[..., None, :] @ rot)[..., 0, :]

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        _, xyz = self._to_base(xyz, t)
        return self.base_potential._potential(xyz, t)  # noqa: SLF001

    @ft.partial(jax.jit)
    def _gradient(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz3:
        rot, xyz = self._to_base(xyz, t)
        grad = self.base_potential._gradient(xyz, t)  # noqa: SLF001
        return (rot @ grad[..., None])[..., 0]  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        _, xyz = self._to_base(xyz, t)
        return self.base_potential._density(xyz, t)  # type: ignore[no-any-return]  # noqa: SLF001

    @ft.partial(jax.jit)
    def _hessian(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz33:
        rot, xyz = self._to_base(xyz, t)
        hess = self.base_potential._hessian(xyz, t)  # noqa: SLF001
        return rot @ hess @ jnp.swapaxes(rot, -1, -2)  # type: ignore[no-any-return]
