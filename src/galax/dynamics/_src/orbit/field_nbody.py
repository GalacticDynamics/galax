"""Field for N-Body dynamics."""

__all__ = ["NBodyField"]

from typing import Any, final

import equinox as eqx
import jax
from jaxtyping import Array, Float, Real
from plum import dispatch

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax._custom_types as gt
import galax.potential as gp
from .field_base import AbstractOrbitField


@final
class NBodyField(AbstractOrbitField, strict=True):  # type: ignore[call-arg]
    r"""Dynamics field for N-Body EoM.

    .. warning::

        The call method currently returns a `tuple[Array[float, (3,)],
        Array[float, (3,)]]`. In future, when `unxt.Quantity` is registered with
        `quax.quaxify` for `diffrax.diffeqsolve` then this will return
        `tuple[Quantity[float, (3,), 'length'], Quantity[float, (3,),
        'speed']]`. Later, when `coordinax.AbstractVector` is registered with
        `quax.quaxify` for `diffrax.diffeqsolve` then this will return
        `tuple[CartesianPos3D, CartesianVel3D]`.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.dynamics as gd

    >>> q = u.Quantity([[-1, 0, 0], [1, 0, 0]], "AU") / 2
    >>> p = u.Quantity([[0, -1, 0], [0, 1, 0]], "km/s") * 25

    >>> solver = gd.OrbitSolver()

    >>> field = gd.fields.NBodyField(
    ...     masses=u.Quantity([1, 1], "Msun"),
    ...     eps=u.Quantity(1e-4, "AU"),
    ...     external_potential=gp.NullPotential(units="solarsystem"))

    >>> t0, t1 = u.Quantity(0, "yr"), u.Quantity(2, "yr")
    >>> soln = solver.solve(field, (q, p), t0, t1)

    >>> soln.ys[0][-1, :, :].round(4)
    Array([[ 0.799 , -0.6521,  0.    ],
           [-0.799 ,  0.6521,  0.    ]], dtype=float64)

    """

    #: masses of each particle.
    masses: Real[u.Quantity["mass"], "N"] | Real[Array, "N"]

    #: softening length.
    eps: Float[u.Quantity["length"], ""] | gt.FloatSz0

    #: Potential.
    external_potential: gp.AbstractPotential = eqx.field(
        default=gp.NullPotential(units="galactic")
    )

    @property
    def units(self) -> u.AbstractUnitSystem:
        return self.external_potential.units

    @property
    def _G(self) -> gt.Sz0:
        us = self.units
        unit = us["length"] ** 3 / (us["mass"] * us["time"] ** 2)
        return self.external_potential.constants["G"].ustrip(unit)

    @dispatch.abstract
    def __call__(
        self, t: Any, qp: tuple[Any, Any], args: tuple[Any, ...], /
    ) -> tuple[Any, Any]:
        raise NotImplementedError  # pragma: no cover


@NBodyField.__call__.dispatch  # type: ignore[misc]
@jax.jit  # type: ignore[misc]
def __call__(
    self: "NBodyField",
    t: gt.Sz0,
    xv: tuple[Real[Array, "N 3"], Real[Array, "N 3"]]
    | tuple[Real[u.AbstractQuantity, "N 3"], Real[u.AbstractQuantity, "N 3"]],
    args: Any,  # noqa: ARG001
    /,
) -> tuple[Real[Array, "N 3"], Real[Array, "N 3"]]:
    # Break apart the input.
    units = self.units
    x, v = xv
    x = u.ustrip(AllowValue, units["length"], x)
    v = u.ustrip(AllowValue, units["speed"], v)

    # Compute the difference vectors ri - rj between all pairs of points
    diffs = x[:, None, :] - x[None, :, :]  # (N, N, 3)

    # Compute softened squared distances.
    eps = self.eps.ustrip(units["length"])
    soft_d2s = jnp.sum(diffs**2, axis=-1)[:, :, None] + eps**2  # (N, N, 1)
    soft_d3s = soft_d2s * jnp.sqrt(soft_d2s)  # (N, N, 1)

    # Compute pairwise forces.
    ms = self.masses.ustrip(units["mass"])  # (N,)
    m2s = ms[:, None, None] * ms[None, :, None]  # (N, N, 1)
    forces = self._G * m2s / soft_d3s * diffs  # (N, N, 3)

    # Remove self-interaction forces
    # TODO: is this necessary since `diffs` has the 0s?
    mask = 1 - jnp.eye(len(x))[:, :, None]  # Shape (N, N, 1)
    forces = forces * mask

    # Sum forces -> acceleration.
    a_self = jnp.sum(forces, axis=0) / ms[:, None]  # (N, 3)

    # Compute acceleration due to external potential.
    a_external = self.external_potential._gradient(x, t)  # noqa: SLF001

    # Total acceleration.
    a_tot = a_self - a_external

    return v, a_tot
