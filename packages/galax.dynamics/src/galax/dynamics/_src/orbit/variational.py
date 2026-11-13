"""Variational equations for orbit fields.

This is private API.

The augmented state carries ``k`` deviation vectors next to the orbit, in the
layout ``((x, dX), (v, dV))`` with ``x, v`` of shape ``(3,)`` and ``dX, dV`` of
shape ``(3, k)``. The deviations obey the linearized equations of motion, whose
right-hand side is the Jacobian of the field applied to each column.

"""

__all__ = ["tangent_terms"]

from typing import Any

import diffrax as dfx
import jax
from plum import dispatch


def _linearize_columns(
    fn: Any, primals: tuple[Any, ...], tangents: tuple[Any, ...]
) -> tuple[Any, Any]:
    """Evaluate ``fn`` and apply its Jacobian to each column of ``tangents``.

    The function is linearized once and the linear map is mapped over the last
    axis of the tangents, so ``k`` deviation vectors cost one evaluation of the
    field plus ``k`` Jacobian-vector products.
    """
    out, lin = jax.linearize(fn, *primals)
    d_out = jax.vmap(lin, in_axes=-1, out_axes=-1)(*tangents)
    return out, d_out


@dispatch
def tangent_terms(term: dfx.ODETerm, /) -> dfx.ODETerm:
    """Augment an ODE term on ``(x, v)`` with the variational equations.

    The returned term acts on ``((x, dX), (v, dV))`` and returns
    ``((dx/dt, d(dX)/dt), (dv/dt, d(dV)/dt))``.

    Examples
    --------
    >>> import diffrax as dfx
    >>> import jax.numpy as jnp
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd
    >>> from galax.dynamics._src.orbit.variational import tangent_terms

    >>> pot = gp.HarmonicOscillatorPotential(omega=0.1, units="galactic")
    >>> term = gd.fields.HamiltonianField(pot).terms(dfx.Dopri8())
    >>> aug = tangent_terms(term)
    >>> x, v, eye = jnp.array([1.0, 0, 0]), jnp.zeros(3), jnp.eye(6)
    >>> (xd, dXd), (vd, dVd) = aug.vf(0.0, ((x, eye[:3]), (v, eye[3:])), None)
    >>> dVd[0, 0]  # -omega^2
    Array(-0.01, dtype=float64)

    """

    def vf(t: Any, y: Any, args: Any) -> Any:
        (x, dx), (v, dv) = y
        (xd, vd), (dxd, dvd) = _linearize_columns(
            lambda x, v: term.vf(t, (x, v), args), (x, v), (dx, dv)
        )
        return (xd, dxd), (vd, dvd)

    return dfx.ODETerm(vf)


@dispatch
def tangent_terms(
    terms: tuple[dfx.ODETerm, dfx.ODETerm], /
) -> tuple[dfx.ODETerm, dfx.ODETerm]:
    """Augment split ``(drift, kick)`` terms with the variational equations.

    The drift acts on ``(v, dV)`` and the kick on ``(x, dX)``. Because a
    symplectic solver applies the same sub-steps to the deviations as to the
    orbit, the deviations follow the exact tangent map of the integrator.

    Examples
    --------
    >>> import diffrax as dfx
    >>> import jax.numpy as jnp
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd
    >>> from galax.dynamics._src.orbit.variational import tangent_terms

    >>> pot = gp.HarmonicOscillatorPotential(omega=0.1, units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)
    >>> drift, kick = tangent_terms(field.terms(dfx.SemiImplicitEuler()))
    >>> x, eye = jnp.array([1.0, 0, 0]), jnp.eye(3)
    >>> a, dA = kick.vf(0.0, (x, eye), None)
    >>> dA[0, 0]  # -omega^2
    Array(-0.01, dtype=float64)

    """
    return tuple(_tangent_split(term) for term in terms)


def _tangent_split(term: dfx.ODETerm, /) -> dfx.ODETerm:
    def vf(t: Any, y: Any, args: Any) -> Any:
        z, dz = y
        return _linearize_columns(lambda z: term.vf(t, z, args), (z,), (dz,))

    return dfx.ODETerm(vf)
