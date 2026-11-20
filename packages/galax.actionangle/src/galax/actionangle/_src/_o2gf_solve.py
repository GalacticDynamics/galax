"""O2GF action and angle linear solves, with sampling diagnostics.

The two linear systems follow Sanders & Binney (2014) and gala's
`_action_prepare` and `_angle_prepare`. Every sum over the T samples is a
weighted sum, so a sample with weight 0 drops out exactly.
"""

__all__ = ["action_system", "angle_system", "solve_o2gf"]

import jax
import jax.numpy as jnp
import numpy as np

from ._utils import safe_div


def action_system(
    J_toy: jax.Array,
    theta: jax.Array,
    nvecs: np.ndarray,
    w: jax.Array,
    /,
) -> tuple[jax.Array, jax.Array]:
    """Matrix and right-hand side for the true actions and generating function.

    ``J_toy`` and ``theta`` have shape (T, 3), ``nvecs`` is (N, 3) and ``w``
    is the (T,) sample weight. Returns ``A`` of shape (3 + N, 3 + N) and ``b``
    of shape (3 + N,).
    """
    n = jnp.asarray(nvecs, dtype=theta.dtype)  # (N, 3)
    cosv = jnp.cos(theta @ n.T)  # (T, N)
    wcos = w[:, None] * cosv
    C_T = 2.0 * n.T * wcos.sum(axis=0)  # (3, N)
    top = jnp.concat([w.sum() * jnp.eye(3, dtype=theta.dtype), C_T], axis=1)
    low = jnp.concat([C_T.T, 4.0 * (n @ n.T) * (wcos.T @ cosv)], axis=1)
    b = jnp.concat(
        [(w[:, None] * J_toy).sum(axis=0), 2.0 * ((J_toy @ n.T) * wcos).sum(axis=0)]
    )
    return jnp.concat([top, low], axis=0), b


def angle_system(
    theta: jax.Array,
    t: jax.Array,
    nvecs: np.ndarray,
    w: jax.Array,
    /,
) -> tuple[jax.Array, jax.Array]:
    """Matrix and right-hand side for the angle offsets, frequencies and dS/dJ.

    ``theta`` is the (T, 3) toy angle, already unwrapped along T. Returns ``A``
    of shape (6 + 3N, 6 + 3N) and ``b`` of shape (6 + 3N,).
    """
    N = len(nvecs)
    eye3 = jnp.eye(3, dtype=theta.dtype)
    n = jnp.asarray(nvecs, dtype=theta.dtype)
    sinv = jnp.sin(theta @ n.T)  # (T, N)
    wsin = w[:, None] * sinv
    s1 = -2.0 * wsin.sum(axis=0)  # (N,)
    s2 = -2.0 * (t[:, None] * wsin).sum(axis=0)
    ss = 4.0 * (wsin.T @ sinv)  # (N, N)

    # Rows 0:6 are the transpose of columns 0:6; the S block is block-diagonal.
    left = jnp.zeros((3 * N, 6), dtype=theta.dtype)
    for i in range(3):
        left = left.at[i * N : (i + 1) * N, i].set(s1)
        left = left.at[i * N : (i + 1) * N, 3 + i].set(s2)
    top = jnp.concat(
        [
            jnp.concat([w.sum() * eye3, (w * t).sum() * eye3, left.T[:3]], axis=1),
            jnp.concat(
                [(w * t).sum() * eye3, (w * t * t).sum() * eye3, left.T[3:]], axis=1
            ),
        ],
        axis=0,
    )
    low = jnp.concat([left, jnp.kron(jnp.eye(3, dtype=theta.dtype), ss)], axis=1)
    A = jnp.concat([top, low], axis=0)
    b = jnp.concat(
        [
            (w[:, None] * theta).sum(axis=0),
            (w[:, None] * t[:, None] * theta).sum(axis=0),
            (-2.0 * (wsin.T @ theta)).T.reshape(-1),  # (3, N) -> (3N,)
        ]
    )
    return A, b


def _fill_masked(x: jax.Array, ok: jax.Array) -> jax.Array:
    """Replace rows where ``ok`` is False by the previous good row (else the next)."""
    i = jnp.arange(len(ok))
    prev = jax.lax.cummax(jnp.where(ok, i, -1))
    nxt = jax.lax.cummin(jnp.where(ok, i, len(ok)), reverse=True)
    idx = jnp.where(prev >= 0, prev, jnp.minimum(nxt, len(ok) - 1))
    return x[idx]


def solve_o2gf(
    J_toy: jax.Array,
    theta_raw: jax.Array,
    t: jax.Array,
    nvecs: np.ndarray,
    /,
) -> tuple[jax.Array, ...]:
    """Solve the O2GF action and angle systems.

    ``J_toy`` and ``theta_raw`` have shape (T, 3) and ``t`` is (T,). A sample
    with any non-finite entry gets weight 0. Before unwrapping, its angles are
    replaced by the previous good sample (the next one for leading samples), so
    the unwrap sees no jump there.

    Returns ``(J[3], S_n[N], theta0[3], Omega[3], dSn_dJ[3N], n_bad_cover,
    n_nan)``. ``n_bad_cover`` counts the n-vectors for which the span of
    ``n . theta`` is under 2 pi (window too short) or the span per sample
    exceeds pi (sampling too coarse). ``n_nan`` is the number of masked samples.
    """
    ok = jnp.isfinite(J_toy).all(axis=1) & jnp.isfinite(theta_raw).all(axis=1)
    ok &= jnp.isfinite(t)
    w = ok.astype(jnp.result_type(J_toy, theta_raw, float))
    J = jnp.where(ok[:, None], J_toy, 0.0)
    tt = jnp.where(ok, t, 0.0)
    theta = jnp.unwrap(_fill_masked(theta_raw, ok), axis=0)

    A, b = action_system(J, theta, nvecs, w)
    x = jnp.linalg.solve(A, b)
    A, b = angle_system(theta, tt, nvecs, w)
    y = jnp.linalg.solve(A, b)

    X = theta @ jnp.asarray(nvecs, dtype=theta.dtype).T  # (T, N)
    span = jnp.where(ok[:, None], X, -jnp.inf).max(axis=0) - jnp.where(
        ok[:, None], X, jnp.inf
    ).min(axis=0)
    span = jnp.where(ok.any(), span, 0.0)
    bad = (span < 2 * jnp.pi) | (safe_div(span, w.sum()) > jnp.pi)
    return (
        x[:3],
        x[3:],
        y[:3],
        y[3:6],
        y[6:],
        bad.sum(),
        (~ok).sum(),
    )
