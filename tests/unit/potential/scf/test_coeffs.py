"""Test SCF coefficient fitting from a particle snapshot."""

import numpy as np
import pytest
import scipy.special as sps

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
from galax.potential.scf import compute_coeffs_discrete

SQRT_FOURPI = 3.544907701811031
"""``sqrt(4 * pi)``, matching the literal in gala's ``bfe_helper.cpp``."""


def _hernquist_samples(n: int, r_s: float, seed: int = 42) -> np.ndarray:
    """Isotropic samples from a Hernquist profile, via its inverse CDF."""
    rng = np.random.default_rng(seed)
    # M(<r)/M_tot = r^2/(r+r_s)^2  =>  r = r_s*sqrt(f)/(1-sqrt(f))
    f = rng.uniform(0.0, 1.0, n)
    r = r_s * np.sqrt(f) / (1 - np.sqrt(f))
    costheta = rng.uniform(-1.0, 1.0, n)
    sintheta = np.sqrt(1 - costheta**2)
    phi = rng.uniform(0.0, 2 * np.pi, n)
    return np.stack(
        [r * sintheta * np.cos(phi), r * sintheta * np.sin(phi), r * costheta],
        axis=-1,
    )


def test_hernquist_samples_recover_the_monopole() -> None:
    """Sampling a Hernquist profile gives S000 ~ 1 and nothing else."""
    n, r_s, m_tot = 200_000, 10.0, 1e12
    xyz = _hernquist_samples(n, r_s)
    mass = np.full(n, m_tot / n)

    Snlm, Tnlm = compute_coeffs_discrete(
        jnp.asarray(xyz), jnp.asarray(mass), nmax=2, lmax=2, r_s=r_s
    )

    assert Snlm.shape == (3, 3, 3)
    # The monopole carries the whole mass, normalized to the scale mass.
    assert jnp.allclose(Snlm[0, 0, 0] / m_tot, 1.0, atol=0.02)
    # Every other term is consistent with zero at this sample size.
    others = Snlm.at[0, 0, 0].set(0.0) / m_tot
    assert jnp.all(jnp.abs(others) < 0.05)
    assert jnp.all(jnp.abs(Tnlm / m_tot) < 0.05)


def test_recovered_coefficients_rebuild_the_potential() -> None:
    """Feeding the fitted coefficients back reproduces Hernquist."""
    n, r_s, m_tot = 200_000, 10.0, 1e12
    xyz = _hernquist_samples(n, r_s)
    mass = np.full(n, m_tot / n)

    Snlm, Tnlm = compute_coeffs_discrete(
        jnp.asarray(xyz), jnp.asarray(mass), nmax=2, lmax=0, r_s=r_s
    )
    scf = gp.SCFPotential(
        m_tot=u.Q(m_tot, "Msun"),
        r_s=u.Q(r_s, "kpc"),
        Snlm=Snlm / m_tot,
        Tnlm=Tnlm / m_tot,
        units="galactic",
    )
    hern = gp.HernquistPotential(
        m_tot=u.Q(m_tot, "Msun"), r_s=u.Q(r_s, "kpc"), units="galactic"
    )
    q = u.Q(np.array([[5.0, 0.0, 0.0], [0.0, 12.0, 3.0]]), "kpc")
    t = u.Q(0.0, "Gyr")

    got, expect = scf.potential(q, t), hern.potential(q, t)
    assert jnp.allclose(got, expect, rtol=0.05, atol=u.Q(0.0, expect.unit))


def _ref_phi_nl(n: int, l: int, s: np.ndarray) -> np.ndarray:
    """Scalar transcription of gala's ``phi_nl`` (``bfe_helper.cpp``)."""
    xi = (s - 1) / (s + 1)
    return (
        -SQRT_FOURPI
        * s**l
        * (1 + s) ** (-2 * l - 1)
        * sps.eval_gegenbauer(n, 2 * l + 1.5, xi)
    )


def _ref_sphPlm(l: int, m: int, x: np.ndarray) -> np.ndarray:
    """Scalar transcription of gala's ``sphPlm``, via the CS-phased `lpmv`."""
    norm = np.sqrt(
        (2 * l + 1) / (4 * np.pi) * sps.factorial(l - m) / sps.factorial(l + m)
    )
    return norm * sps.lpmv(m, l, x)


def _ref_anl(n: int, l: int) -> float:
    """Scalar transcription of gala's ``A_nl`` (``coeff_helper.cpp``)."""
    knl = 0.5 * n * (n + 4 * l + 3) + (l + 1) * (2 * l + 1)
    log_ratio = (
        sps.gammaln(n + 1) + 2 * sps.gammaln(2 * l + 1.5) - sps.gammaln(n + 4 * l + 3)
    )
    return (
        -np.exp((8 * l + 6) * np.log(2.0) + log_ratio)
        / (4 * np.pi * knl)
        * (n + 2 * l + 1.5)
    )


def _ref_snlm_tnlm(
    xyz: np.ndarray, mass: np.ndarray, r_s: float, n: int, l: int, m: int
) -> tuple[float, float]:
    """Scalar transcription of gala's ``S_nlm``/``T_nlm`` sum over particles.

    Gala-free: built from `sps.lpmv`/`sps.eval_gegenbauer` directly, not from
    `phi_nl`/`compute_Ylm`, so it does not share a bug with the module under
    test.
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    s, big_x, phi = r / r_s, z / r, np.arctan2(y, x)
    km = 1.0 if m == 0 else 2.0
    term = km * _ref_anl(n, l) * mass * _ref_phi_nl(n, l, s) * _ref_sphPlm(l, m, big_x)
    return (
        float(np.sum(term * np.cos(m * phi))),
        float(np.sum(term * np.sin(m * phi))),
    )


def test_coeffs_match_scalar_reference_for_flattened_triaxial_sample() -> None:
    """A non-isotropic sample pins ``l >= 1`` coefficients against a reference.

    The isotropic Hernquist samples used elsewhere make every ``l >= 1`` term
    zero in expectation, so sampling noise hides a ``(2 - delta_m0)``
    axis-convention mistake. Flattening ``z`` and stretching ``x``/``y`` by
    different factors gives an oblate, triaxial sample where ``l = 2``
    (``m = 0, 1, 2``) is genuinely non-zero, so the weighting is exercised
    against an independent reference rather than just a noise bound.
    """
    n, r_s, m_tot = 200_000, 10.0, 1e12
    xyz = _hernquist_samples(n, r_s, seed=7) * np.array([1.3, 0.7, 0.5])
    mass = np.full(n, m_tot / n)

    Snlm, Tnlm = compute_coeffs_discrete(
        jnp.asarray(xyz), jnp.asarray(mass), nmax=2, lmax=2, r_s=r_s
    )

    triplets = [
        (0, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 2, 0),
        (1, 2, 0),
        (0, 2, 1),
        (0, 2, 2),
    ]
    for n_, l_, m_ in triplets:
        ref_s, ref_t = _ref_snlm_tnlm(xyz, mass, r_s, n_, l_, m_)
        assert Snlm[n_, l_, m_] == pytest.approx(ref_s, rel=1e-6)
        assert Tnlm[n_, l_, m_] == pytest.approx(ref_t, rel=1e-6)

    # l >= 1, m = 0 terms are genuinely large for this sample, not values a
    # factor-of-2 weighting bug could hide inside sampling noise.
    assert abs(float(Snlm[0, 1, 0]) / m_tot) > 1e-3
    assert abs(float(Snlm[0, 2, 0]) / m_tot) > 0.1


def test_particle_at_origin_gives_finite_coefficients() -> None:
    """A particle at exactly the origin must not poison the coefficients.

    The origin is a coordinate singularity (``r = 0``, ``theta`` and ``phi``
    undefined) that a snapshot can perfectly ordinarily contain -- e.g. a
    central black hole marker, or a particle-centred frame. Coefficient
    fitting must route through the guarded
    `galax.potential._src.builtin.multipole.cartesian_to_normalized_spherical`
    transform rather than recomputing ``theta``/``phi`` by hand, or this
    single particle NaNs out most of the ``Snlm``/``Tnlm`` array.
    """
    n, r_s = 500, 10.0
    xyz = np.asarray(_hernquist_samples(n, r_s))
    xyz[0] = 0.0
    mass = np.full(n, 1.0 / n)

    Snlm, Tnlm = compute_coeffs_discrete(
        jnp.asarray(xyz), jnp.asarray(mass), nmax=2, lmax=2, r_s=r_s
    )

    assert jnp.all(jnp.isfinite(Snlm))
    assert jnp.all(jnp.isfinite(Tnlm))


def test_compute_var_returns_a_covariance_block() -> None:
    """`compute_var=True` adds a (2, 2, ...) covariance array."""
    n, r_s = 5_000, 10.0
    xyz = _hernquist_samples(n, r_s)
    mass = np.full(n, 1.0 / n)

    _Snlm, _Tnlm, cov = compute_coeffs_discrete(
        jnp.asarray(xyz),
        jnp.asarray(mass),
        nmax=1,
        lmax=1,
        r_s=r_s,
        compute_var=True,
    )

    assert cov.shape == (2, 2, 2, 2, 2)
    assert jnp.all(cov[0, 0] >= 0)  # var(S) is non-negative
    assert jnp.allclose(cov[0, 1], cov[1, 0])  # symmetric
