"""Test SCF coefficient fitting from a particle snapshot."""

import numpy as np

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
from galax.potential.scf import compute_coeffs_discrete


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
