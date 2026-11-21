"""O2GF action and angle linear solves, with sampling diagnostics.

The two linear systems follow Sanders & Binney (2014) and gala's
`_action_prepare` and `_angle_prepare`. Every sum over the T samples is a
weighted sum, so a sample with weight 0 drops out exactly.
"""

__all__ = ["action_system", "angle_system", "solve_o2gf"]

import jax
import jax.numpy as jnp
import numpy as np

from ._utils import fill_masked, safe_div

# Floor (rad) of the threshold on the span of n . theta for a resonant n-vector.
_RESONANT = 1e-6


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
    rest: jax.Array | None = None,
) -> tuple[jax.Array, ...]:
    """Solve the O2GF action and angle systems.

    ``J_toy`` and ``theta_raw`` have shape (T, 3) and ``t`` is (T,). A sample
    with any non-finite entry gets weight 0. Before unwrapping, its angles are
    replaced by the previous good sample (the next one for leading samples), so
    the unwrap sees no jump there.

    Returns ``(J[3], S_n[N], theta0[3], Omega[3], dSn_dJ[3N], n_bad_cover,
    n_nan, n_resonant)``. ``n_nan`` is the number of masked samples.

    An n-vector is resonant when ``n . theta`` spans less than
    ``max(1e-6, 100 eps max|n . theta|)`` rad over the orbit (1e-6 rad in
    float64), and each toy angle with ``n_i != 0`` spans at least 2 pi on its
    own, e.g. ``n = (0, 2, -2)`` in a spherical potential, where
    ``Omega_phi = Omega_z``. Its column is then collinear with the action
    columns, so it is left out of both solves, with S_n and dS_n/dJ set to 0,
    and counted in ``n_resonant``. An n-vector below the threshold whose
    angles do not each span 2 pi (a window too short to tell) is left out
    too, but counted in ``n_bad_cover``. ``rest`` (bool, shape (3,)) marks
    axes whose toy action is 0 along the whole orbit, e.g. z for a planar
    orbit; the n-vectors with a nonzero component on such an axis are left
    out too, and are not counted.

    ``n_bad_cover`` counts the other n-vectors for which the span of
    ``n . theta`` is under 2 pi (window too short) or the span per sample
    exceeds pi (sampling too coarse). Those not below the resonance
    threshold stay in the solves, as in gala.

    Both systems are solved by minimum-norm least squares. On a regular system
    this matches gala's ``scipy.linalg.solve``. Dropping the resonant modes
    does not make every system regular: in a spherical potential the
    ``cos(n . theta)`` of n-vectors that differ by a resonant vector are
    still linearly dependent, and there the minimum norm picks the S_n, where
    gala's solve is singular.
    """
    ok = jnp.isfinite(J_toy).all(axis=1) & jnp.isfinite(theta_raw).all(axis=1)
    ok &= jnp.isfinite(t)
    w = ok.astype(jnp.result_type(J_toy, theta_raw, float))
    J = jnp.where(ok[:, None], J_toy, 0.0)
    tt = jnp.where(ok, t, 0.0)
    theta = jnp.unwrap(fill_masked(theta_raw, ok), axis=0)

    X = theta @ jnp.asarray(nvecs, dtype=theta.dtype).T  # (T, N)
    span = jnp.where(ok[:, None], X, -jnp.inf).max(axis=0) - jnp.where(
        ok[:, None], X, jnp.inf
    ).min(axis=0)
    span = jnp.where(ok.any(), span, 0.0)
    skip = (
        jnp.zeros(len(nvecs), bool)
        if rest is None
        else jnp.any((nvecs != 0) & rest, axis=1)
    )
    # In float32 the round-off of the unwrapped angles exceeds 1e-6 rad.
    eps = float(np.finfo(theta.dtype).eps)
    tol = jnp.maximum(
        _RESONANT, 100 * eps * jnp.max(jnp.abs(jnp.where(ok[:, None], X, 0.0)))
    )
    below = (span < tol) & ~skip
    # A constant n . theta is a resonance only if each angle it involves turns
    # through 2 pi on its own; otherwise the window is too short to tell.
    th_span = jnp.where(ok[:, None], theta, -jnp.inf).max(axis=0) - jnp.where(
        ok[:, None], theta, jnp.inf
    ).min(axis=0)
    turns = jnp.where(ok.any(), th_span, 0.0) >= 2 * jnp.pi
    resonant = below & jnp.all((nvecs == 0) | turns, axis=1)
    bad = (span < 2 * jnp.pi) | (safe_div(span, w.sum()) > jnp.pi)
    bad &= ~(skip | resonant)

    drop = below | skip
    A, b = action_system(J, theta, nvecs, w)
    x = _solve_masked(A, b, jnp.concat([jnp.zeros(3, bool), drop]))
    A, b = angle_system(theta, tt, nvecs, w)
    y = _solve_masked(A, b, jnp.concat([jnp.zeros(6, bool), jnp.tile(drop, 3)]))
    return (
        x[:3],
        x[3:],
        y[:3],
        y[3:6],
        y[6:],
        bad.sum(),
        (~ok).sum(),
        resonant.sum(),
    )
