"""Lyapunov exponents from the variational equations.

This is private API.

"""

__all__ = ["compute_lyapunov"]

from jaxtyping import Array, Float
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax.coordinates as gc
import galax.potential as gp
from .field_base import AbstractOrbitField
from .field_hamiltonian import HamiltonianField
from .solver import OrbitSolver
from .variational import _solve_augmented, tangent_terms
from galax.dynamics._src.utils import parse_to_t_y


@eqx.filter_jit
def compute_lyapunov(
    field: AbstractOrbitField | gp.AbstractPotential,
    w0: gc.AbstractPhaseSpaceCoordinate,
    t1: Any,
    /,
    *,
    n_renorm: int,
    k: int = 1,
    solver: OrbitSolver | None = None,
    dt0: float | None = None,
) -> tuple[Float[Array, "*batch n_renorm"], Float[Array, "*batch n_renorm k"]]:
    r"""Compute running estimates of the Lyapunov exponents of an orbit.

    We integrate ``k`` deviation vectors with the variational equations,
    orthonormalize them by a QR decomposition at ``n_renorm`` evenly spaced
    times between ``w0.t`` and ``t1``, and add up the logarithms of the
    diagonal of :math:`R`. Dividing by the elapsed time gives the running
    estimate :math:`\\lambda_j(t_i)`. With ``k=1`` this is the largest exponent;
    with ``k=6`` it is the full spectrum.

    The deviation norm mixes positions and velocities in the field's units.
    The limit does not depend on that choice. With a constant-step solver the
    renormalization interval must be a multiple of ``dt0``.

    Parameters
    ----------
    field
        The orbit field. A potential is wrapped in a
        `galax.dynamics.fields.HamiltonianField`.
    w0
        The initial phase-space coordinate, with any batch shape.
    t1
        The final time. A `unxt.Quantity` is converted to the field's units.
    n_renorm
        The number of renormalization intervals.
    k
        The number of exponents, from 1 to 6.
    solver
        The solver. Defaults to `galax.dynamics.OrbitSolver`.
    dt0
        Initial step size, needed for constant-step solvers.

    Returns
    -------
    ts
        The renormalization times, shape ``(*batch, n_renorm)``, in the field's
        time unit.
    lam
        The running estimates, shape ``(*batch, n_renorm, k)``, in the field's
        inverse time unit.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    A harmonic oscillator is regular, so its exponents tend to zero:

    >>> pot = gp.HarmonicOscillatorPotential(omega=u.Q(0.1, "1/Myr"),
    ...                                      units="galactic")
    >>> w0 = gc.PhaseSpaceCoordinate(q=u.Q([1.0, 0, 0], "kpc"),
    ...                              p=u.Q([0, 0.1, 0], "kpc/Myr"),
    ...                              t=u.Q(0, "Myr"))
    >>> ts, lam = gd.orbit.compute_lyapunov(pot, w0, u.Q(1, "Gyr"),
    ...                                     n_renorm=10, k=6)
    >>> lam.shape
    (10, 6)
    >>> bool(jnp.all(jnp.abs(lam[-1]) < 1e-2))
    True

    """
    thefield = (
        field if isinstance(field, AbstractOrbitField) else HamiltonianField(field)  # type: ignore[call-arg]
    )
    solver = OrbitSolver() if solver is None else solver
    units = thefield.units
    t1 = u.ustrip(AllowValue, units["time"], t1)
    terms = tangent_terms(thefield.terms(solver.solver))
    q0 = jnp.linalg.qr(jnp.ones((6, k)) + jnp.eye(6, k))[0]

    t0, (x0, v0) = parse_to_t_y(None, w0, ustrip=units)
    batch = x0.shape[:-1]
    t0 = jnp.broadcast_to(t0, batch).reshape(-1)
    x0, v0 = x0.reshape(-1, 3), v0.reshape(-1, 3)

    def single(t0: Any, x0: Any, v0: Any) -> Any:
        tis = jnp.linspace(t0, t1, n_renorm + 1)

        def interval(carry: Any, tt: Any) -> Any:
            x, v, q = carry
            y0 = ((x, q[:3]), (v, q[3:]))
            (xs, dxs), (vs, dvs) = _solve_augmented(
                terms, solver, tt[0], tt[1], y0, dfx.SaveAt(t1=True), dt0
            )
            q, r = jnp.linalg.qr(jnp.concat([dxs[-1], dvs[-1]], axis=0))
            return (xs[-1], vs[-1], q), jnp.log(jnp.abs(jnp.diag(r)))

        _, logr = jax.lax.scan(interval, (x0, v0, q0), (tis[:-1], tis[1:]))
        lam = jnp.cumsum(logr, axis=0) / jnp.abs(tis[1:] - t0)[:, None]
        return tis[1:], lam

    ts, lam = jax.vmap(single)(t0, x0, v0)
    return ts.reshape(*batch, n_renorm), lam.reshape(*batch, n_renorm, k)
