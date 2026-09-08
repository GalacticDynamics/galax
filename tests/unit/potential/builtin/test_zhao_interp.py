"""Tests for `InterpolatedZhaoPotential`.

The contract is that it is numerically indistinguishable from `ZhaoPotential`
-- it is the same model, with Zhao's Eq. 43 fitted rather than summed -- so
almost everything here is a comparison against the analytic class.

Standalone rather than built on `AbstractSinglePotential_Test`: that suite
drives the power-law indices as `ParameterField`s, and here they are static by
construction.
"""

import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp

# The Table 1 correspondences plus a generic case, as (alpha, beta, gamma).
ABG = [
    (1.0, 4.0, 1.0),  # Hernquist
    (1.0, 4.0, 2.0),  # Jaffe (p0 = 0)
    (0.5, 5.0, 0.0),  # Plummer
    (1.0, 3.0, 1.0),  # NFW (q0 = 0, infinite mass)
    (1.0, 3.0, 0.5),  # gNFW
    (0.9, 4.31, 1.2),  # generic
    (0.4, 2.5, 2.5),  # both indices negative
]
METHODS = ["potential", "gradient", "density", "laplacian", "hessian"]

M = u.Quantity(1e12, "Msun")
R_S = u.Quantity(8.0, "kpc")
T = u.Quantity(0.0, "Myr")


def _pair(abg, **kw):
    alpha, beta, gamma = abg
    fields = {
        "m": M,
        "r_s": R_S,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "units": "galactic",
    }
    return gp.InterpolatedZhaoPotential(**fields, **kw), gp.ZhaoPotential(**fields)


def _xyz(n=64, lo=1e-3, hi=1e3):
    """Positions spanning a wide range of radii, off-axis."""
    r = np.geomspace(lo, hi, n)
    direction = np.array([0.36, -0.48, 0.8])  # unit vector
    return u.Quantity(r[:, None] * direction, "kpc")


@pytest.mark.parametrize("abg", ABG)
@pytest.mark.parametrize("method", METHODS)
def test_matches_analytic(abg, method: str) -> None:
    """Must agree with `ZhaoPotential` across the radial range."""
    interp, exact = _pair(abg)
    xyz = _xyz()

    got, expect = getattr(interp, method)(xyz, T), getattr(exact, method)(xyz, T)
    got = np.asarray(u.ustrip(got.unit, got))
    expect = np.asarray(u.ustrip(expect.unit, expect))

    # `hessian` has near-cancelling entries, so compare on the tensor's scale.
    scale = np.max(np.abs(expect))
    np.testing.assert_allclose(got, expect, rtol=1e-10, atol=1e-10 * scale)


@pytest.mark.parametrize("abg", ABG)
def test_mass_enclosed_matches_analytic(abg) -> None:
    """The enclosed mass drives every derivative, so check it directly."""
    interp, exact = _pair(abg)
    xyz = _xyz()
    got = gp.spherical_mass_enclosed(interp, xyz, T)
    expect = gp.spherical_mass_enclosed(exact, xyz, T)
    np.testing.assert_allclose(
        np.asarray(u.ustrip("Msun", got)),
        np.asarray(u.ustrip("Msun", expect)),
        rtol=1e-10,
    )


def test_more_coefficients_is_not_worse() -> None:
    """Accuracy must be controllable by `n_coeffs`."""
    xyz = _xyz()
    _, exact = _pair((0.9, 4.31, 1.2))
    expect = exact.potential(xyz, T)
    expect_v = np.asarray(u.ustrip(expect.unit, expect))

    errs = []
    for n in (8, 16, 32):
        interp, _ = _pair((0.9, 4.31, 1.2), n_coeffs=n)
        got = interp.potential(xyz, T)
        got_v = np.asarray(u.ustrip(got.unit, got))
        errs.append(np.max(np.abs(got_v / expect_v - 1)))

    assert errs[0] > errs[-1]  # more coefficients, better fit
    assert errs[-1] < 1e-12


def test_time_dependent_mass_still_works() -> None:
    """``m`` and ``r_s`` stay ordinary parameters; only the indices are fixed."""

    def m_of_t(t: u.Quantity["time"]) -> u.Quantity["mass"]:
        return u.Quantity(1e12 * (1 + 0.1 * t.ustrip("Myr")), "Msun")

    fields = {"r_s": R_S, "alpha": 1.0, "beta": 4.0, "gamma": 1.0, "units": "galactic"}
    interp = gp.InterpolatedZhaoPotential(m=m_of_t, **fields)
    exact = gp.ZhaoPotential(m=m_of_t, **fields)

    xyz = u.Quantity([8.0, 0.0, 0.0], "kpc")
    for t in (0.0, 5.0, 50.0):
        tq = u.Quantity(t, "Myr")
        got, expect = interp.potential(xyz, tq), exact.potential(xyz, tq)
        assert jnp.isclose(got, expect, atol=u.Quantity(1e-12, got.unit))
    # and the mass really does vary
    assert not jnp.isclose(
        interp.potential(xyz, u.Quantity(0.0, "Myr")),
        interp.potential(xyz, u.Quantity(50.0, "Myr")),
        atol=u.Quantity(1e-6, "kpc2/Myr2"),
    )


def test_from_zhao_round_trip() -> None:
    """`from_zhao` must copy the indices and reproduce the source potential."""
    _, exact = _pair((0.9, 4.31, 1.2))
    interp = gp.InterpolatedZhaoPotential.from_zhao(exact)

    assert (interp.alpha, interp.beta, interp.gamma) == (0.9, 4.31, 1.2)
    xyz = _xyz()
    got, expect = interp.potential(xyz, T), exact.potential(xyz, T)
    assert jnp.allclose(got, expect, atol=u.Quantity(1e-12, got.unit))


def test_negative_integer_index_is_rejected() -> None:
    """A negative-integer `b` needs a log term the fit does not carry."""
    # b = alpha (beta - 3) = -1 exactly
    with pytest.raises(ValueError, match="negative integer"):
        gp.InterpolatedZhaoPotential(
            m=M, r_s=R_S, alpha=1.0, beta=2.0, gamma=1.0, units="galactic"
        )
