"""Dispatch stubs for action-angle-frequency estimation."""

__all__ = [
    "staeckel_fudge_delta",
    "staeckel_fudge",
    "isochrone_actions_angles_freqs",
    "harmonic_oscillator_actions_angles_freqs",
    "o2gf_actions_angles_freqs",
]

from typing import Any

from plum import dispatch


@dispatch.abstract
def staeckel_fudge_delta(*args: Any, **kwargs: Any) -> Any:
    """Estimate the focal distance Δ for the Staeckel fudge.

    Overloads:

    - ``(pot, q, t)``
    - ``(pot, w)``

    Δ is the focal distance of the prolate-spheroidal coordinates the fudge
    works in: eq. 9 of Sanders (2012), evaluated at each given point, as in
    gala. Only positions are used, so the velocities of ``w`` are ignored.

    Parameters
    ----------
    pot : `galax.potential.AbstractPotential`
        The potential. The Staeckel fudge assumes it is axisymmetric about the
        z axis.
    q : Quantity[length] or Array, shape (..., 3)
        Cartesian positions. A bare array is in ``pot.units``.
    t : Quantity[time] or Array, shape (...)
        Times, broadcast against the batch shape of ``q``. For per-point
        times with ``q`` of shape (N, 3), use ``t`` of shape (N,): shape
        (N, 1) broadcasts to (N, N).
    w : `galax.coordinates.AbstractPhaseSpaceCoordinate`
        A phase-space coordinate or orbit, e.g. a `galax.dynamics.Orbit`.

    Returns
    -------
    Quantity[length] or Array, shape (...)
        Δ in ``pot.units``, with the batch shape of the input. It is a
        Quantity if any input is one, and always a Quantity for ``w``.

    Notes
    -----
    - Δ is exactly 0 everywhere, without evaluating the Hessian, for a
      potential declaring `galax.potential.Symmetry.SPHERICAL` (including a
      composite of spherical parts). A spherical potential that does not
      declare it gets 0 up to round-off, and NaN on the plane and the axis.
    - Otherwise Δ is NaN where it is undefined: where Δ² < 0; on the z axis
      (R = 0 or subnormal); on the symmetry plane of a plane-symmetric
      potential (a razor-thin disc such as Kuzmin gives its one-sided value);
      and for a separable potential.
    - Δ is also NaN, rather than finite and wrong, where the potential's
      second derivatives are inconsistent, e.g. for one evaluated near the
      bottom of the float range.
    - Use 64-bit floats (``jax_enable_x64``). In float32 a genuine
      Δ ≲ 0.02 r is rounded to 0. Δ itself is scale-free, but in float32
      SI units a mass above ~1.7e8 Msun (3.4e38 kg) does not fit, and the
      potential must not overflow on lengths ~1e20 m.
    - A potential not declaring `~galax.potential.Symmetry.SPHERICAL` whose
      own Hessian is imprecise can give a spurious Δ, 0 or NaN there, and
      which one can change with batch size or jit; no built-in does. A
      potential separable analytically but not in floating point (e.g. an
      l ≤ 2, m = 0 interior multipole) gives an arbitrary Δ instead of NaN.
    - Δ is not differentiable: the output is wrapped in
      `jax.lax.stop_gradient`.
    - The Staeckel fudge assumes a potential axisymmetric about the z axis
      through the origin. For a non-axisymmetric or off-centre one, Δ is
      only a heuristic, and it can be finite on z = 0.

    The fudge wants one Δ per orbit, but the per-point values along an orbit
    scatter. ``quaxed.numpy.nanmedian(staeckel_fudge_delta(pot, orbit),
    axis=-1, keepdims=True)`` gives one Δ per orbit, skipping the NaN points
    on the plane and the axis. Plain `jax.numpy` rejects a Quantity.

    References
    ----------
    Sanders J. L., 2012, MNRAS, 426, 128.
    gala's ``get_staeckel_fudge_delta``
    (https://github.com/adrn/gala/pull/649).

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp
    >>> from galax.actionangle import staeckel_fudge_delta
    >>> pot = gp.MiyamotoNagaiPotential(m_tot=u.Q(1e11, "Msun"),
    ...                                 a=u.Q(3.0, "kpc"), b=u.Q(0.5, "kpc"),
    ...                                 units="galactic")
    >>> delta = staeckel_fudge_delta(pot, u.Q([8.0, 0, 1.0], "kpc"), u.Q(0.0, "Myr"))
    >>> delta.shape, str(delta.unit), round(float(delta.value), 2)
    ((), 'kpc', 3.69)

    One Δ for an orbit, as the median over its points:

    >>> import quaxed.numpy as jnp
    >>> import galax.coordinates as gc
    >>> import galax.dynamics as gd
    >>> w0 = gc.PhaseSpaceCoordinate(q=u.Q([8.0, 0, 1.0], "kpc"),
    ...                              p=u.Q([0.0, 200, 20], "km/s"),
    ...                              t=u.Q(0.0, "Myr"))
    >>> orbit = gd.evaluate_orbit(pot, w0, u.Q(jnp.linspace(0, 1000, 201), "Myr"))
    >>> deltas = staeckel_fudge_delta(pot, orbit)
    >>> deltas.shape
    (201,)
    >>> delta = jnp.nanmedian(deltas, axis=-1, keepdims=True)
    >>> delta.shape, str(delta.unit), round(float(delta.value[0]), 1)
    ((1,), 'kpc', 3.9)
    """
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def staeckel_fudge(*args: Any, **kwargs: Any) -> Any:
    """Estimate actions, angles and frequencies with the Staeckel fudge.

    Overloads:

    - ``(pot, q, p, t, delta, *, phi_inf=None)``
    - ``(pot, w, delta, *, phi_inf=None)``
    """
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def isochrone_actions_angles_freqs(*args: Any, **kwargs: Any) -> Any:
    """Compute actions, angles and frequencies in an isochrone potential.

    Overloads:

    - ``(pot, q, p, t)``
    - ``(pot, w)``

    The batch shapes of ``q``, ``p`` (``shape[:-1]``) and ``t`` broadcast
    together. Angles are in ``[0, 2 pi)``; unbound points give NaN (except
    ``J_phi``). Values and gradients of bound points are finite. Enable
    64-bit floats (``jax_enable_x64``) for precision; the transform is
    scale-free, so float32 works in any unit system that can hold the mass:
    in SI, at most ~1.7e8 Msun (3.4e38 kg).
    ``J_r`` goes through ``E``: near circular orbits its absolute precision
    is ``~ eps L`` (relative ``~ eps / delta^2``, ``delta`` the fractional
    offset from the circular speed). Coordinate singularities:

    - Radial orbits (``L = 0``, numerically ``1 - e^2 <= 1e-30``): ``theta_r``
      is correct; ``theta_z`` and ``theta_phi`` (undefined) are set to 0.
      Near-radial orbits get all three angles.
    - The origin at rest (``E = Phi(0)``): all angles are set to 0.
    - Circular orbits: ``theta_r`` (undefined) is 0.
    - Planar orbits (inclination ``< 1e-10``): ``theta_z`` is undefined (it
      is conjugate to ``J_z = 0``); the value follows gala's convention
      (``psi = pi/2``) and ``theta_phi`` stays continuous.
    - On the z axis ``theta_phi`` is defined (through the longitude of the
      ascending node) whenever the orbit is not planar.
    - ``L_z = 0``: ``Omega_phi = 0`` and ``theta_phi`` is the node longitude
      (no ``theta_z`` term, ``sign(0) = 0``), consistent with the orbit.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp
    >>> from galax.actionangle import isochrone_actions_angles_freqs
    >>> pot = gp.IsochronePotential(m_tot=u.Q(1e11, "Msun"),
    ...                             r_s=u.Q(2.0, "kpc"), units="galactic")
    >>> res = isochrone_actions_angles_freqs(
    ...     pot, u.Q([8.0, 0, 0], "kpc"), u.Q([0.0, 150, 0], "km/s"), u.Q(0.0, "Myr"))
    >>> res.actions.shape, str(res.angles.unit), str(res.freqs.unit)
    ((3,), 'rad', 'rad / Myr')
    """
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def harmonic_oscillator_actions_angles_freqs(*args: Any, **kwargs: Any) -> Any:
    """Compute actions, angles and frequencies in a harmonic-oscillator potential.

    Overloads:

    - ``(pot, q, p, t)``
    - ``(pot, w)``

    Each Cartesian axis is an independent oscillator, so the results are in
    (x, y, z) order with ``J_i = (v_i^2 + omega_i^2 x_i^2) / (2 omega_i)`` and
    ``Omega_i = omega_i``. The angles follow gala's convention,
    ``x_i = sqrt(2 J_i / omega_i) cos(theta_i)`` and
    ``v_i = -sqrt(2 J_i omega_i) sin(theta_i)``, and lie in ``[0, 2 pi)``.
    A scalar ``omega`` is used for all three axes.

    The batch shapes of ``q``, ``p`` (``shape[:-1]``) and ``t`` broadcast
    together. The angle of an axis is undefined where ``J_i = 0``, a particle
    at rest at the origin of that axis, and is set to 0 there, with a
    gradient of 0. The transform is the toy for box orbits in the Sanders &
    Binney (2014) formalism.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp
    >>> from galax.actionangle import harmonic_oscillator_actions_angles_freqs
    >>> pot = gp.HarmonicOscillatorPotential(
    ...     omega=u.Q([1.0, 2.0, 3.0], "1 / Myr"), units="galactic")
    >>> res = harmonic_oscillator_actions_angles_freqs(
    ...     pot, u.Q([1.0, 0, 0], "kpc"), u.Q([0.0, 1, 0], "kpc/Myr"), u.Q(0.0, "Myr"))
    >>> res.actions.shape, str(res.angles.unit), str(res.freqs.unit)
    ((3,), 'rad', 'rad / Myr')
    """
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def o2gf_actions_angles_freqs(*args: Any, **kwargs: Any) -> Any:
    """Compute actions, angles and frequencies of orbits with O2GF.

    Overloads:

    - ``(pot, q, p, t, /, *, n_max, toy=None, family=None)``
    - ``(pot, w, /, *, n_max, toy=None, family=None)``

    The method of Sanders & Binney (2014), as in gala's
    ``find_actions_o2gf``. Each orbit is mapped through a toy potential whose
    actions and angles are known, and the true actions, angles and
    frequencies come from a least-squares fit of the generating function over
    the samples. Loop orbits use an isochrone toy, after the circulation axis
    is swapped with z. Box orbits use a harmonic-oscillator toy.

    Parameters
    ----------
    pot : `galax.potential.AbstractPotential`
        The potential the orbit was integrated in. It supplies the units of
        bare arrays and G, and seeds the isochrone fit.
    q, p : Quantity or Array, shape (..., T, 3)
        Cartesian positions and velocities along the orbits. Bare arrays are
        in ``pot.units``.
    t : Quantity[time] or Array, shape (..., T)
        The sample times, broadcast against ``q`` and ``p``.
    w : `galax.coordinates.AbstractPhaseSpaceCoordinate`
        An orbit, e.g. from `galax.dynamics.evaluate_orbit`. It must have a
        time axis, which is its last batch axis. A single phase-space point,
        or ``q`` with fewer than 2 samples, raises `ValueError`.
    n_max : int
        The largest ``|n|`` of the n-vectors in the generating function.
        Static.
    toy : `IsochroneToy` or `OscillatorToy`, optional
        Fixed toy parameters, which skip the fit. The fields are bare arrays
        in ``pot.units``; Quantity fields raise `TypeError`. They may carry
        the batch axes of the orbits. An isochrone toy always takes the loop
        path and an oscillator toy the box path. An orbit of the other family
        gets ``family_mismatch`` and is flagged.
    family : {None, "loop", "box"}, optional
        ``None`` classifies each orbit and runs both paths. A fixed family
        runs only that path, and an orbit that is not of that family gets
        ``family_mismatch`` and is flagged. A family that contradicts
        ``toy`` raises `ValueError`.

    Returns
    -------
    `O2GFResult`
        Every field except the static ``nvecs`` has the batch shape of the
        orbits in front.

        - ``aaf``: the actions, angles and frequencies, with Quantities if
          any input is one (always for ``w``). The angles are those at
          ``t = 0``, in ``[0, 2 pi)``. Loop orbits are in the aligned frame
          and ordered (R, φ, z); box orbits are ordered (x, y, z). J_phi is
          signed.
        - ``toy``: the toy each orbit used, fitted or given, in
          ``pot.units``. These are bare arrays by design, and a ``toy``
          passed in must be in ``pot.units`` too.
        - ``is_loop``: whether the orbit took the loop path.
        - ``aligned_axis``: the axis (0, 1 or 2) that was swapped with z
          before the loop path, 2 when none was. To go back, swap axis 2
          with ``aligned_axis``. The swap is a reflection, so after one
          J_phi is -L_x or -L_y, as in gala.
        - ``sn``, ``dsn_dj``: the generating function S_n for each n-vector
          and its derivative dS_n/dJ, flattened from shape (3, N). ``sn``
          has the unit of ``aaf.actions`` when that carries units, and is
          bare otherwise. ``dsn_dj`` is dimensionless and always a bare array.
        - ``nvecs``: the n-vectors, shape (N, 3). With ``family=None`` and
          no ``toy``, ``toy`` is the pair ``(IsochroneToy, OscillatorToy)``,
          ``nvecs`` is the pair ``(loop_nvecs, box_nvecs)``, and the box
          S_n and each block of the box dS_n/dJ are zero-padded to the loop
          length.
        - ``n_bad_cover``: the number of n-vectors whose angle covers less
          than 2 pi, or whose span divided by the number of kept (unmasked)
          samples exceeds pi (the mean step, as in gala). It leaves out the
          n-vectors that involve a rest axis and the resonant n-vectors.
        - ``n_nan``: the number of samples dropped for non-finite toy
          values. Samples with a non-finite input are also left out of the
          classification and the toy fit.
        - ``n_resonant``: the number of exactly resonant n-vectors left out
          of the fit (see Notes).
        - ``fit_converged``: whether the toy fit's solver converged (True
          for a given toy). It does not show that the toy is a good one. In
          float32 the fitted isochrone can be far from the best one on
          near-circular orbits with ``fit_converged`` True, and the
          oscillator fit can collapse an axis with it True (see
          ``toy_degenerate``). Coverage does not catch every bad toy either.
        - ``toy_degenerate``: whether a box orbit's oscillator toy has
          collapsed. That is, a fitted omega_i on an axis not at rest is
          below 0.02 of its seed. On many triaxial box orbits the fit drives one omega_i
          to about 0, and the actions on that axis come out wrong by orders
          of magnitude. Always False on the loop path.
        - ``thin_box``: whether a box orbit is thin, that is, the smallest
          mean toy action on an axis not at rest is below 3e-4 of the sum
          (see Notes). Always False on the loop path.
        - ``family_mismatch``: whether the path forced by ``family`` or by
          the type of ``toy`` disagrees with the orbit's classification, in
          either direction. Always False with ``family=None`` and no ``toy``.
        - ``flagged``: ``n_bad_cover > 0``, or ``n_nan > NAN_MAX`` (with
          ``NAN_MAX = 1``), or the fit did not converge, or
          ``toy_degenerate``, or ``thin_box``, or ``family_mismatch``.

    Notes
    -----
    An orbit is ``flagged`` when its angles cover the n-vectors badly (the
    orbit is too short or too coarsely sampled), when more than one sample
    has non-finite toy values, when the toy fit did not converge, when a box
    orbit's oscillator toy has collapsed, when a box orbit is thin, or when
    a forced family does not match the orbit. A
    flagged orbit keeps its values, so check ``flagged``. Enable 64-bit
    floats (``jax_enable_x64``) for precision.

    Thin box orbits are flagged. On a nearly planar box the fit modes that
    involve J_z absorb the in-plane fit error, which can move the in-plane
    actions by a few percent and J_z by up to the whole of it. A box is
    flagged as thin when its smallest mean toy action on an axis not at
    rest is below 3e-4 of the sum. In tests on triaxial NFW, triaxial
    Hernquist and LMJ09 boxes, the near-planar ones with bad actions
    reached 5.8e-5, and no random 3-D box was below 1.6e-3. The flag also
    catches thin boxes whose actions are right. A planar orbit, with J_z at
    rest, is not affected by this.

    On eccentric orbits in a spherical NFW potential, J_r can be off by
    about 1e-3 (1.05e-3 to 1.3e-3 on two orbits with J_r/L of about 4 to 7,
    measured against quadrature) without a flag. The same happens in
    spherical Hernquist loops, where J_r was off by 1.0e-3 to 2.6e-3 with
    J_r/L of 7 to 36. Eccentric orbits in other cuspy potentials are likely
    to behave the same way. A longer orbit does not reduce it, and a higher
    ``n_max`` does not reliably reduce it either. In Hernquist the error grew
    with ``n_max``, and in NFW it went up or down depending on the orbit.

    On flagged, ill-conditioned orbits (Jacobian norm around 1e17) the
    fitted toy depends on what else is in the batch, and the actions moved
    by about 3 percent between batch compositions. Unflagged orbits agree
    with single-orbit runs to 5e-14. The same batch always gives the same
    result, but the result for an ill-conditioned orbit can change when the
    other orbits in the batch change.

    In a spherical or nearly spherical potential, numerical noise in the
    samples of about 1e-7 relative or larger can keep a resonant mode just
    above the cutoff below. The orbit then stays flagged however long it is
    or however low ``n_max`` is (short of ``n_max=2``), and integrating the
    orbit more precisely removes the flag.

    An n-vector whose angle ``n . theta`` is constant along the orbit (an
    exact resonance, such as ``Omega_phi = Omega_z`` in a spherical
    potential) is left out of the fit and counted in ``n_resonant``, which
    does not flag the orbit. Constant means a span below
    ``max(1e-6, 100 eps max|n . theta|)`` rad while each toy angle it
    involves spans at least 2 pi. A mode below that threshold whose angles
    do not each span 2 pi is left out too, but counted in ``n_bad_cover``.
    Both linear systems are solved by minimum-norm least squares, which
    agrees with gala on a regular system and picks the smallest S_n where
    the system is singular.

    For an axis whose toy action is 0 along the whole orbit (e.g. z for a
    planar orbit), the n-vectors that involve it are left out, the action is
    0, and the conjugate angle and frequency are undefined and returned as
    NaN.

    Sample about 40 periods of the orbit's slowest frequency. For 30 random
    disk orbits in `galax.potential.MilkyWayPotential2022` at ``n_max=8``,
    with 2000 samples, 21 were flagged at 10 periods, 13 at 20 and 9 at 40.
    Doubling the samples at 40 periods changed none of them. In those 9 some
    ``n . Omega`` is so close to 0 that its angle still covers less than
    2 pi. For an orbit still flagged at 40 periods, first try a lower
    ``n_max``. In a second sample of 30 such orbits, 8 were flagged at 40
    periods; 5 of them were unflagged at ``n_max=6`` and 7 at ``n_max=4``,
    and their actions moved by less than 1 percent from those at
    ``n_max=8``. A longer orbit helped less: at 80 and at 160 periods, with
    the same samples per period, 6 of the 8 stayed flagged. If neither
    works, treat the result as unreliable.

    Orbits near a low-order resonance can be wrong without being flagged.
    Near the 3:2 resonance of Omega_z and Omega_r, J_z came out 10 to 22
    percent off on unflagged orbits, and a longer orbit did not fix it. The
    actions of tubes about the long axis of a triaxial potential are
    sensitive to the toy fit, and can move by up to 10 percent as ``n_max``
    changes without being flagged: on one x-tube J_r moved by 10 percent
    across ``n_max`` = 6, 8 and 10. Regular box orbits can be sensitive too:
    on an LMJ09 box J_x moved by about 4 percent across ``n_max`` = 4 to 10
    without a flag, while staying stable with orbit length.

    Very small actions carry an absolute error of a few parts in 1e7 to 1e5
    of the sum of the actions, with an arbitrary sign, on unflagged orbits
    as well. On near-planar Miyamoto-Nagai loops J_z came out at -2.45e-7
    against a true value between 3.7e-12 and 3.7e-8, and boxes just above
    the thin-box cutoff had offsets of up to 8e-6 of the sum.

    The fitted toy differs from gala's (Powell) at the fit tolerance, so
    pass ``toy`` to compare with gala exactly. At odd ``n_max`` the
    n-vectors differ from gala's.

    References
    ----------
    Sanders J. L., Binney J., 2014, MNRAS, 441, 3284.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.dynamics as gd
    >>> import galax.potential as gp
    >>> import quaxed.numpy as jnp
    >>> from galax.actionangle import o2gf_actions_angles_freqs
    >>> pot = gp.MilkyWayPotential2022()
    >>> w0 = gc.PhaseSpaceCoordinate(q=u.Q([8.0, 0, 0.5], "kpc"),
    ...                              p=u.Q([10.0, 220, 20], "km/s"),
    ...                              t=u.Q(0.0, "Myr"))
    >>> orbit = gd.evaluate_orbit(pot, w0, u.Q(jnp.linspace(0, 2000, 2000), "Myr"))
    >>> res = o2gf_actions_angles_freqs(pot, orbit, n_max=4)
    >>> res.aaf.actions.shape, str(res.aaf.actions.unit)
    ((3,), 'kpc2 / Myr')
    >>> round(float(u.ustrip("kpc2/Myr", res.aaf.actions[1])), 2)
    1.8
    >>> bool(res.is_loop), bool(res.flagged)
    (True, False)
    """
    raise NotImplementedError  # pragma: no cover
