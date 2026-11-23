"""O2GF actions, angles and frequencies of loop and box orbits.

The method of Sanders & Binney (2014), following gala's ``find_actions_o2gf``.
A loop orbit is aligned so that it circulates about z and is mapped through an
isochrone toy; a box orbit is mapped through a harmonic-oscillator toy. The
true actions, angles and frequencies then come from two linear solves over the
samples of the orbit (`solve_o2gf`).
"""

__all__ = ["NAN_MAX", "O2GFResult"]

import functools as ft
import warnings

from jaxtyping import Array
from typing import Any, Final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from plum import dispatch

import unxt as u

import galax.coordinates as gc
import galax.potential as gp
from ._o2gf_nvecs import BOX_STEPS, LOOP_STEPS, n_vectors
from ._o2gf_orbit import align_circulation, circulation, is_loop
from ._o2gf_solve import solve_o2gf
from ._o2gf_toy import (
    IsochroneToy,
    OscillatorToy,
    _oscillator_seed,
    fit_isochrone,
    fit_oscillator,
    toy_aaf,
)
from ._parse import parse_qpt, parse_w, wrap_result
from ._utils import G_in_units, fill_masked, mod_2pi, safe_div
from .result import ActionsAnglesFrequencies

# An orbit with more masked (non-finite) toy samples than this is flagged. gala
# drops the NaN samples and allows one.
NAN_MAX: Final = 1

# An axis is at rest when its toy action is below this fraction of the total
# at every sample.
_REST: Final = 1e-12
# The oscillator toy has collapsed when a fitted omega_i on an axis not at rest
# is below this fraction of its seed. On 137 random triaxial-NFW and LMJ09 box
# orbits and 898 box runs of a near-planar sweep (triaxial NFW, triaxial
# Hernquist, LMJ09), the toys that did not collapse had omega/seed >= 0.12 and
# the collapsed ones <= 5.4e-3.
_COLLAPSE: Final = 0.02
# A box orbit is thin when the smallest mean toy action on an axis not at rest
# is below this fraction of the sum. In the sweep the near-planar boxes with
# bad actions reached 5.8e-5 and one borderline orbit sat at 1.7e-4. The
# smallest unflagged random 3-D box was at 1.6e-3, and a good LMJ09 box with a
# thin in-plane axis at 6.4e-4. None of 79 random 3-D boxes fell below it.
_THIN_BOX: Final = 3e-4
_FAMILIES: Final = (None, "loop", "box")


class O2GFResult(eqx.Module):
    """Result of `o2gf_actions_angles_freqs`.

    Every array has the batch shape of the orbits in front. ``aaf`` holds the
    actions, angles and frequencies, and ``is_loop`` says which path each
    orbit took. Loop orbits are in the aligned frame, ordered (R, φ, z);
    ``aligned_axis`` is the axis that was swapped with z (2 when none was).
    The swap is a reflection, so after one J_phi is -L_x or -L_y, as in
    gala. Box orbits are ordered (x, y, z). The angles are those at
    ``t = 0``, in ``[0, 2 pi)``. On an axis whose toy action is 0 along
    the whole orbit (e.g. z for a planar orbit) the action is 0 and the angle
    and frequency are NaN.

    ``toy`` is the toy each orbit used, fitted or given. Its fields are bare
    arrays in ``pot.units``, a unit system the code controls, and a ``toy``
    passed in must be in ``pot.units`` too. When the family is
    chosen per orbit (``family=None`` and no ``toy``), it is the pair
    ``(IsochroneToy, OscillatorToy)`` and ``nvecs`` is the pair ``(loop_nvecs,
    box_nvecs)``. ``sn`` holds the generating function S_n for each n-vector,
    with the unit of ``aaf.actions`` when that carries units and bare
    otherwise. ``dsn_dj`` is its derivative, dimensionless and always a bare
    array. Both are flattened from shape (3, N). Box entries
    are then zero-padded to the loop length, per block of ``dsn_dj``.

    ``family_mismatch`` is True when the path that ran, fixed by ``family``
    or by the type of ``toy``, is not the one `is_loop` picks for the orbit
    (in either direction). It is always False with ``family=None`` and no
    ``toy``.

    ``n_resonant`` counts the n-vectors whose angle ``n . theta`` is constant
    along the orbit, e.g. ``(0, 2, -2)`` in a spherical potential: it spans
    less than ``max(1e-6, 100 eps max|n . theta|)`` rad, while each toy angle
    it involves spans at least 2 pi. They are left out of the fit, with
    S_n = 0, and do not flag the orbit. ``n_bad_cover`` counts the other
    n-vectors, except those that involve a rest axis, whose angle covers
    less than 2 pi, or whose span divided by the number of kept (unmasked)
    samples exceeds pi (the mean step, as in gala). All of them count when
    some toy angle is predicted to step by more than pi / 2 between two kept
    samples, e.g. across a gap in ``t`` or with samples out of time order,
    since the angles then cannot be unwrapped. The step is predicted from
    the faster of the toy frequency and the angle's mean rate along the
    orbit, because a fitted toy's frequency can be several times too slow
    (2 to 3.4 times for Omega_z on the disk loops tested). Those below the
    resonance threshold are left out of the fit too.
    ``n_nan`` counts the samples dropped for non-finite toy values. Samples
    with a non-finite input are also left out of the classification and of
    the toy fit.
    ``fit_converged`` says whether the toy fit converged (True for a given
    toy). ``toy_degenerate`` is True when a box orbit's oscillator toy has
    collapsed: a fitted omega_i on an axis not at rest is below 0.02 of its
    seed. This happens
    on many triaxial box orbits, where the best variance fit wants a
    negative omega_i^2 and the fit stops at omega_i = 0 with the solver
    converged. The actions are then wrong by orders of magnitude, often
    with coverage that passes. It is always False on the loop path.
    ``thin_box`` is True when a box orbit is nearly planar or nearly linear:
    the smallest mean toy action on an axis not at rest is below 3e-4 of the
    sum. The fit then moves every action, by up to a few percent in plane
    and up to the whole J_z. It is always False on the loop path. An
    orbit is ``flagged`` when ``n_bad_cover > 0``, more than `NAN_MAX`
    samples were dropped, the fit did not converge, ``toy_degenerate`` is
    True, ``thin_box`` is True, or ``family_mismatch`` is True. A flagged
    orbit keeps its values, so check ``flagged`` before using them. When
    the toy fit is ill-conditioned, the result for an orbit can change when
    the other orbits in the batch change.
    """

    aaf: ActionsAnglesFrequencies
    toy: IsochroneToy | OscillatorToy | tuple[IsochroneToy, OscillatorToy]
    is_loop: Array
    aligned_axis: Array
    sn: Array
    dsn_dj: Array
    n_bad_cover: Array
    n_resonant: Array
    n_nan: Array
    fit_converged: Array
    toy_degenerate: Array
    thin_box: Array
    family_mismatch: Array
    flagged: Array
    nvecs: np.ndarray | tuple[np.ndarray, np.ndarray] = eqx.field(static=True)


class _Out(NamedTuple):
    """Per-orbit output of one path."""

    toy: Any
    ok: Array
    axis: Array
    J: Array
    th0: Array
    Om: Array
    sn: Array
    dsn: Array
    n_bad: Array
    n_nan: Array
    n_res: Array
    loop: Array
    degenerate: Array
    thin: Array
    mismatch: Any = None


def _finite(q: Array, p: Array, t: Array) -> Array:
    """Mark the samples with finite ``q``, ``p`` and ``t``.

    The fits and the classification keep these. A non-finite input sample
    gives non-finite toy values, so the solves drop it too.
    """
    return jnp.isfinite(q).all(axis=1) & jnp.isfinite(p).all(axis=1) & jnp.isfinite(t)


def _path(
    loop: bool,  # noqa: FBT001
    pot: Any,
    G: Array,
    q: Array,
    p: Array,
    t: Array,
    toy: Any,
    nvecs: np.ndarray,
) -> _Out:
    """Run the loop or the box path on one orbit."""
    keep = _finite(q, p, t)
    if loop:
        q, p, axis = align_circulation(q, p, circulation(q, p, keep))
    else:
        axis = jnp.asarray(2)
    fitted = toy is None
    if fitted:
        # Fit in the default float: optimistix's Levenberg-Marquardt mixes in
        # float64 under jax_enable_x64, which a float32 problem rejects.
        f = jnp.result_type(float)
        qf, pf, tf = q.astype(f), p.astype(f), t.astype(f)
        toy, ok = (
            fit_isochrone(pot, qf, pf, tf, keep)
            if loop
            else fit_oscillator(qf, pf, tf, keep)
        )
        toy = jax.tree.map(lambda x: x.astype(q.dtype), toy)
    else:
        ok = jnp.asarray(True)  # noqa: FBT003
    # Map the masked samples through the toy at a kept sample, so their NaN
    # does not reach the gradient, then mark them NaN for the solve to drop.
    J, theta, Om_toy = toy_aaf(toy, G, fill_masked(q, keep), fill_masked(p, keep))
    J = jnp.where(keep[:, None], J, jnp.nan)
    theta = jnp.where(keep[:, None], theta, jnp.nan)
    # An axis whose toy action stays 0 (a planar orbit) has no angle to fit.
    aJ = jnp.abs(J)
    # The mean toy action per axis, for the thin-box check below.
    mJ = jax.lax.stop_gradient(jnp.nanmean(aJ, axis=0))
    tiny = (aJ <= _REST * aJ.sum(axis=1, keepdims=True)) | ~jnp.isfinite(aJ)
    rest = jax.lax.stop_gradient(jnp.all(tiny, axis=0))
    # Solve about the mean time, which conditions the angle system far better
    # (it matters in float32), then step the angles back to t = 0.
    tc = jnp.mean(jnp.where(jnp.isfinite(t), t, 0.0))
    # The unwrap along T needs each toy angle to step by less than pi between
    # kept samples. A gap in t, or samples out of time order, breaks that.
    # Predict each step from the faster of the toy frequency and the angle's
    # mean rate along the orbit: a fitted toy's frequency can be several
    # times too slow (Omega_z on a disk loop), and the angle's own rate
    # varies by ~1.5x along the orbit, so flag predicted steps above pi / 2.
    dt = jnp.abs(jnp.diff(fill_masked(t, keep)))
    dth = jnp.abs(mod_2pi(jnp.diff(fill_masked(theta, keep), axis=0) + jnp.pi) - jnp.pi)
    mean_rate = safe_div(jnp.nansum(dth, axis=0), dt.sum())
    rate = jnp.fmax(jnp.abs(Om_toy[1:]), mean_rate)
    step = jnp.where(rest, 0.0, rate * dt[:, None])
    # Equal sample times leave the frequencies undetermined.
    coarse = (jnp.nanmax(step, initial=0.0) > jnp.pi / 2) | (dt.sum() == 0)
    coarse = jax.lax.stop_gradient(coarse)
    J, sn, th0, Om, dsn, n_bad, n_nan, n_res = solve_o2gf(
        J, theta, t - tc, nvecs, rest, coarse
    )
    th0 = mod_2pi(th0 - Om * tc)
    # On an axis at rest J is 0, and its angle and frequency are undefined.
    J = jnp.where(rest, 0.0, J)
    th0 = jnp.where(rest, jnp.nan, th0)
    Om = jnp.where(rest, jnp.nan, Om)
    if loop:
        degenerate = thin = jnp.asarray(False)  # noqa: FBT003
    else:
        # On a thin box the n_z != 0 modes absorb the in-plane fit residual,
        # which moves every action. Use the toy actions, which do not depend
        # on that solve.
        thin = jnp.min(jnp.where(rest, jnp.inf, mJ)) < _THIN_BOX * mJ.sum()
        # A small negative J_i is not a sign of collapse: a near-planar box
        # carries a J_z offset of arbitrary sign with a good toy.
        degenerate = jnp.asarray(False)  # noqa: FBT003
        if fitted:
            omega0 = _oscillator_seed(
                fill_masked(qf, keep), fill_masked(tf, keep), keep
            )
            degenerate |= jnp.any(~rest & (toy.omega < _COLLAPSE * omega0))
    return _Out(
        toy,
        ok,
        axis,
        J,
        th0,
        Om,
        sn,
        dsn,
        n_bad,
        n_nan,
        n_res,
        jnp.asarray(loop),
        degenerate,
        thin,
    )


def _pad(sn: Array, dsn: Array, n: int) -> tuple[Array, Array]:
    """Zero-pad S_n to length n, and each of the 3 blocks of dS_n/dJ."""
    k = n - len(sn)
    dsn = jnp.pad(dsn.reshape(3, -1), ((0, 0), (0, k))).reshape(-1)
    return jnp.pad(sn, (0, k)), dsn


@ft.partial(jax.jit, static_argnames=("n_max", "family"))
def _o2gf(
    pot: Any, q: Array, p: Array, t: Array, toy: Any, *, n_max: int, family: Any
) -> _Out:
    """Batched core; ``q``, ``p`` are (..., T, 3) and ``t`` is (..., T)."""
    G = jnp.asarray(G_in_units(pot), dtype=q.dtype)
    loop_n, box_n = n_vectors(n_max, LOOP_STEPS), n_vectors(n_max, BOX_STEPS)

    def one(q: Array, p: Array, t: Array, toy: Any) -> _Out:
        sel = is_loop(circulation(q, p, _finite(q, p, t)))
        if family is not None:
            loop = family == "loop"
            out = _path(loop, pot, G, q, p, t, toy, loop_n if loop else box_n)
            # The forced path may not be the orbit's family.
            return out._replace(mismatch=sel != loop)
        lo = _path(True, pot, G, q, p, t, None, loop_n)  # noqa: FBT003
        bo = _path(False, pot, G, q, p, t, None, box_n)  # noqa: FBT003
        sn, dsn = _pad(bo.sn, bo.dsn, len(loop_n))
        bo = bo._replace(sn=sn, dsn=dsn)
        mixed = jax.tree.map(
            lambda a, b: jnp.where(sel, a, b),
            lo._replace(toy=None),
            bo._replace(toy=None),
        )
        out = mixed._replace(toy=(lo.toy, bo.toy), mismatch=jnp.asarray(False))  # noqa: FBT003
        return out

    for _ in q.shape[:-2]:
        one = jax.vmap(one)
    return one(q, p, t, toy)


def _run(
    pot: Any,
    q: Array,
    p: Array,
    t: Array,
    as_quantity: bool,  # noqa: FBT001
    n_max: int,
    toy: Any,
    family: Any,
) -> O2GFResult:
    if q.ndim < 2 or q.shape[-2] < 2:
        msg = (
            "O2GF needs orbits with a time axis of at least 2 samples, so q "
            f"must have shape (..., T, 3) with T >= 2, got {q.shape}; a single "
            "phase-space point has no time axis"
        )
        raise ValueError(msg)
    if family not in _FAMILIES:
        msg = f"family must be None, 'loop' or 'box', got {family!r}"
        raise ValueError(msg)
    if toy is not None:
        if not isinstance(toy, IsochroneToy | OscillatorToy):
            msg = f"toy must be an IsochroneToy or OscillatorToy, got {type(toy)}"
            raise TypeError(msg)

        def isq(x: Any) -> bool:
            return isinstance(x, u.AbstractQuantity)

        if any(isq(x) for x in jax.tree.leaves(toy, is_leaf=isq)):
            msg = (
                f"the fields of {type(toy).__name__} must be bare arrays in "
                f"pot.units ({pot.units}), not Quantities"
            )
            raise TypeError(msg)
        toy_family = "loop" if isinstance(toy, IsochroneToy) else "box"
        if family not in (None, toy_family):
            msg = (
                f"family={family!r} contradicts a {type(toy).__name__}, "
                f"which needs family={toy_family!r}"
            )
            raise ValueError(msg)
        family = toy_family
        # broadcast the toy over the orbit batch; an omega keeps its last axis
        core = 1 if isinstance(toy, OscillatorToy) else 0
        batch = q.shape[:-2]
        toy = jax.tree.map(
            lambda x: jnp.broadcast_to(
                jnp.asarray(x, dtype=q.dtype),
                batch + jnp.shape(x)[jnp.ndim(x) - core :],
            ),
            toy,
        )

    loop_n, box_n = n_vectors(n_max, LOOP_STEPS), n_vectors(n_max, BOX_STEPS)
    for name, n in (("loop", loop_n), ("box", box_n)):
        if family in (None, name) and len(n) == 0:
            msg = f"n_max={n_max!r} gives no n-vectors for the {name} path"
            raise ValueError(msg)
    out = _o2gf(pot, q, p, t, toy, n_max=n_max, family=family)
    nvecs: np.ndarray | tuple[np.ndarray, np.ndarray] = (
        (loop_n, box_n) if family is None else loop_n if family == "loop" else box_n
    )
    with warnings.catch_warnings():
        # equinox warns about any array in a static field; nvecs is a
        # trace-time NumPy constant, so static is what we want.
        warnings.filterwarnings("ignore", "A JAX array is being set as static")
        aaf = wrap_result(out.J, out.th0, out.Om, pot, as_quantity)
        return O2GFResult(
            aaf=aaf,
            toy=out.toy,
            is_loop=out.loop,
            aligned_axis=out.axis,
            # S_n has the units of an action, so it follows aaf.actions
            sn=(
                u.Q(out.sn, pot.units["length"] ** 2 / pot.units["time"])
                if as_quantity
                else out.sn
            ),
            dsn_dj=out.dsn,
            n_bad_cover=out.n_bad,
            n_resonant=out.n_res,
            n_nan=out.n_nan,
            fit_converged=out.ok,
            toy_degenerate=out.degenerate,
            thin_box=out.thin,
            family_mismatch=out.mismatch,
            flagged=(out.n_bad > 0)
            | (out.n_nan > NAN_MAX)
            | ~out.ok
            | out.degenerate
            | out.thin
            | out.mismatch,
            nvecs=nvecs,
        )


@dispatch
def o2gf_actions_angles_freqs(
    pot: gp.AbstractPotential,
    q: Any,
    p: Any,
    t: Any,
    /,
    *,
    n_max: int,
    toy: IsochroneToy | OscillatorToy | None = None,
    family: str | None = None,
) -> O2GFResult:
    """O2GF from positions and velocities (..., T, 3) and times (..., T)."""
    if jnp.ndim(t) == 0 or jnp.shape(t)[-1] != jnp.shape(q)[-2]:
        msg = (
            f"t needs a time axis of length T: q has shape {jnp.shape(q)}, "
            f"so t needs shape (..., {jnp.shape(q)[-2]}), but t has shape "
            f"{jnp.shape(t)}."
        )
        raise ValueError(msg)
    q, p, t, as_quantity = parse_qpt(pot, q, p, t)
    return _run(pot, q, p, t, as_quantity, n_max, toy, family)


@dispatch
def o2gf_actions_angles_freqs(
    pot: gp.AbstractPotential,
    w: gc.AbstractPhaseSpaceCoordinate,
    /,
    *,
    n_max: int,
    toy: IsochroneToy | OscillatorToy | None = None,
    family: str | None = None,
) -> O2GFResult:
    """O2GF from an orbit, whose time axis is the last batch axis."""
    q, p, t = parse_w(pot, w)
    return _run(pot, q, p, t, True, n_max, toy, family)  # noqa: FBT003
