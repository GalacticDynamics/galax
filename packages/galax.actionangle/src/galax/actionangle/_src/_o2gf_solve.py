"""O2GF action and angle linear solves, with sampling diagnostics.

The two linear systems follow Sanders & Binney (2014) and gala's
`_action_prepare` and `_angle_prepare`. Every sum over the T samples is a
weighted sum, so a sample with weight 0 drops out exactly.
"""

__all__ = ["action_system", "angle_system", "solve_o2gf"]

from typing import Final

import jax
import jax.numpy as jnp
import numpy as np

from ._utils import fill_masked, finite_rows, masked_span, safe_div

# Floor (rad) of the threshold on the span of n . theta for a resonant n-vector.
_RESONANT: Final = 1e-6


def action_system(
    J_toy: jax.Array,
    X: jax.Array,
    n: jax.Array,
    w: jax.Array,
    /,
) -> tuple[jax.Array, jax.Array]:
    """Matrix and right-hand side for the true actions and generating function.

    ``J_toy`` has shape (T, 3), ``n`` is the (N, 3) n-vectors, ``X`` the
    (T, N) phases ``theta @ n.T`` and ``w`` the (T,) sample weight. Returns
    ``A`` of shape (3 + N, 3 + N) and ``b`` of shape (3 + N,).
    """
    cosv = jnp.cos(X)  # (T, N)
    wcos = w[:, None] * cosv
    C_T = 2.0 * n.T * wcos.sum(axis=0)  # (3, N)
    top = jnp.concat([w.sum() * jnp.eye(3, dtype=X.dtype), C_T], axis=1)
    low = jnp.concat([C_T.T, 4.0 * (n @ n.T) * (wcos.T @ cosv)], axis=1)
    b = jnp.concat(
        [(w[:, None] * J_toy).sum(axis=0), 2.0 * ((J_toy @ n.T) * wcos).sum(axis=0)]
    )
    return jnp.concat([top, low], axis=0), b


def angle_system(
    theta: jax.Array,
    X: jax.Array,
    t: jax.Array,
    w: jax.Array,
    /,
) -> tuple[jax.Array, jax.Array]:
    """Matrix and right-hand side for the angle offsets, frequencies and dS/dJ.

    ``theta`` is the (T, 3) toy angle, already unwrapped along T, and ``X``
    the (T, N) phases ``theta @ n.T``. Returns ``A`` of shape
    (6 + 3N, 6 + 3N) and ``b`` of shape (6 + 3N,).
    """
    eye3 = jnp.eye(3, dtype=theta.dtype)
    sinv = jnp.sin(X)  # (T, N)
    wsin = w[:, None] * sinv
    s1 = -2.0 * wsin.sum(axis=0)  # (N,)
    s2 = -2.0 * (t[:, None] * wsin).sum(axis=0)
    wt = w * t
    ss = 4.0 * (wsin.T @ sinv)  # (N, N)

    # Rows 0:6 are the transpose of columns 0:6; the S block is block-diagonal.
    left = jnp.concat(
        [jnp.kron(eye3, s1[:, None]), jnp.kron(eye3, s2[:, None])], axis=1
    )
    swt = wt.sum()
    moments = jnp.array([[w.sum(), swt], [swt, (wt * t).sum()]])
    top = jnp.concat([jnp.kron(moments, eye3), left.T], axis=1)
    low = jnp.concat([left, jnp.kron(eye3, ss)], axis=1)
    A = jnp.concat([top, low], axis=0)
    b = jnp.concat(
        [
            (w[:, None] * theta).sum(axis=0),
            (wt[:, None] * theta).sum(axis=0),
            (-2.0 * (wsin.T @ theta)).T.reshape(-1),  # (3, N) -> (3N,)
        ]
    )
    return A, b


def _solve_masked(A: jax.Array, b: jax.Array, drop: jax.Array) -> jax.Array:
    """Minimum-norm least-squares solution of ``A x = b``, with ``x[drop] = 0``.

    The rows and columns of the dropped unknowns are replaced by those of the
    identity. ``A`` is scaled by its diagonal before the pseudo-inverse, so the
    cutoff on small singular values is relative to a well-scaled matrix. On a
    regular system this equals `jnp.linalg.solve` to round-off. JAX gives
    `jnp.linalg.pinv` a closed-form derivative, so gradients stay finite on a
    singular system too.
    """
    keep = ~drop
    A = jnp.where(keep[:, None] & keep[None, :], A, jnp.diag(drop.astype(A.dtype)))
    b = jnp.where(keep, b, 0.0)
    a = jnp.abs(jnp.diag(A))
    d = 1 / jnp.sqrt(jnp.where(a > 0, a, 1.0))
    # singular values below 100 eps of the largest are cut; static for the jvp
    rtol = 100 * float(np.finfo(A.dtype).eps)
    x: jax.Array = d * (jnp.linalg.pinv(d[:, None] * A * d, rtol=rtol) @ (d * b))
    return x


def solve_o2gf(
    J_toy: jax.Array,
    theta_raw: jax.Array,
    t: jax.Array,
    nvecs: np.ndarray,
    /,
    rest: jax.Array,
    coarse: jax.Array | bool,
) -> tuple[jax.Array, ...]:
    """Solve the O2GF action and angle systems.

    ``J_toy`` and ``theta_raw`` have shape (T, 3) and ``t`` is (T,). A sample
    with any non-finite entry gets weight 0. Before unwrapping, its angles are
    replaced by the previous good sample (the next one for leading samples), so
    the unwrap sees no jump there. ``rest`` (bool, shape (3,)) marks the axes
    at rest, and ``coarse`` an orbit whose angles step too far to unwrap.

    Returns ``(J[3], S_n[N], theta0[3], Omega[3], dSn_dJ[3N], n_bad_cover,
    n_nan, n_resonant)``. ``n_nan`` is the number of masked samples. See
    `o2gf_actions_angles_freqs` for the resonant, badly covered and rest-axis
    n-vectors. Dropping the resonant modes does not make every system
    regular: in a spherical potential the ``cos(n . theta)`` of n-vectors
    that differ by a resonant vector are still linearly dependent, and there
    the minimum norm picks the S_n, where gala's solve is singular.
    """
    # Mask non-finite samples: weight 0, zeroed J and t, angles filled before
    # unwrapping.
    ok = finite_rows(J_toy, theta_raw, t)
    w = ok.astype(jnp.result_type(J_toy, theta_raw, float))
    J = jnp.where(ok[:, None], J_toy, 0.0)
    tt = jnp.where(ok, t, 0.0)
    theta = jnp.unwrap(fill_masked(theta_raw, ok), axis=0)

    # Span of each n . theta over the kept samples.
    n = jnp.asarray(nvecs, dtype=theta.dtype)
    X = theta @ n.T  # (T, N)
    span = masked_span(X, ok[:, None])
    # Modes that involve an axis at rest are left out and not counted.
    skip = jnp.any((nvecs != 0) & rest, axis=1)
    # Resonance threshold. In float32 the round-off of the unwrapped angles
    # exceeds 1e-6 rad.
    eps = float(np.finfo(theta.dtype).eps)
    tol = jnp.maximum(
        _RESONANT, 100 * eps * jnp.max(jnp.abs(jnp.where(ok[:, None], X, 0.0)))
    )
    below = (span < tol) & ~skip
    # A constant n . theta is a resonance only if each angle it involves turns
    # through 2 pi on its own; otherwise the window is too short to tell.
    turns = masked_span(theta, ok[:, None]) >= 2 * jnp.pi
    resonant = below & jnp.all((nvecs == 0) | turns, axis=1)
    # Badly covered: under one turn, more than pi per sample, or the whole
    # orbit too coarse to unwrap.
    bad = (span < 2 * jnp.pi) | (safe_div(span, w.sum()) > jnp.pi) | coarse
    bad &= ~(skip | resonant)

    # Solve for x = (J, S_n) and y = (theta0, Omega, dS_n/dJ), with the
    # dropped modes fixed at 0.
    drop = below | skip
    A, b = action_system(J, X, n, w)
    x = _solve_masked(A, b, jnp.concat([jnp.zeros(3, bool), drop]))
    A, b = angle_system(theta, X, tt, w)
    y = _solve_masked(A, b, jnp.concat([jnp.zeros(6, bool), jnp.tile(drop, 3)]))
    return (
        x[:3],  # J
        x[3:],  # S_n
        y[:3],  # theta0
        y[3:6],  # Omega
        y[6:],  # dS_n/dJ, (3N,)
        bad.sum(),  # n_bad_cover
        (~ok).sum(),  # n_nan
        resonant.sum(),  # n_resonant
    )
