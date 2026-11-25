"""float32 inputs give float32 outputs, also with ``jax_enable_x64`` on.

The suite runs with x64 on. A float64 input still gives float64.
"""

import jax
import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp

FUNCS = (gp.potential, gp.gradient, gp.acceleration, gp.density, gp.hessian)
M = u.Q(1e11, "Msun")


def _hernquist() -> gp.HernquistPotential:
    return gp.HernquistPotential(m_tot=M, r_s=u.Q(5, "kpc"), units="galactic")


def _pots() -> list[gp.AbstractPotential]:
    return [
        gp.KeplerPotential(m_tot=M, units="galactic"),
        _hernquist(),
        gp.NFWPotential(m=M, r_s=u.Q(15, "kpc"), units="galactic"),
        gp.MiyamotoNagaiPotential(
            m_tot=M, a=u.Q(3, "kpc"), b=u.Q(0.3, "kpc"), units="galactic"
        ),
        gp.MN3ExponentialPotential(
            m_tot=M, h_R=u.Q(3, "kpc"), h_z=u.Q(0.3, "kpc"), units="galactic"
        ),
        gp.MilkyWayPotential2022(),
        gp.TranslatedPotential(_hernquist(), translation=u.Q([1.5, 0, 0], "kpc")),
        gp.HardCutoffNFWPotential(
            m=M, r_s=u.Q(5, "kpc"), r_t=u.Q(67, "kpc"), units="galactic"
        ),
        gp.FlattenedInThePotential(_hernquist(), q_z=2.0),
        gp.TriaxialInThePotential(_hernquist(), y_over_x=2.0),
    ]


POTS = pytest.mark.parametrize("pot", _pots(), ids=lambda p: type(p).__name__)


@POTS
@pytest.mark.parametrize("func", FUNCS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_dtype_follows_input(
    pot: gp.AbstractPotential, func: object, dtype: object
) -> None:
    """The output has the input's float dtype, as an array or a Quantity."""
    xyz = jnp.asarray([8.0, 1.0, 0.5], dtype=dtype)
    t = jnp.asarray(0.0, dtype=dtype)
    assert func(pot, xyz, t).dtype == dtype
    assert func(pot, u.Q(xyz, "kpc"), u.Q(t, "Myr")).dtype == dtype


@POTS
@pytest.mark.parametrize("func", FUNCS, ids=lambda f: f.__name__)
def test_float32_matches_float64(pot: gp.AbstractPotential, func: object) -> None:
    """The float32 result is the float64 one, to float32 precision."""
    xyz = jnp.asarray([8.0, 1.0, 0.5])
    got = func(pot, xyz.astype(jnp.float32), jnp.float32(0))
    want = func(pot, xyz, 0.0)
    atol = 1e-5 * float(jnp.max(jnp.abs(want)))
    assert jnp.allclose(got, want, rtol=1e-5, atol=atol)


@pytest.mark.parametrize("func", FUNCS, ids=lambda f: f.__name__)
def test_python_scalar_time_does_not_promote(func: object) -> None:
    """A Python int or float time does not promote a float32 position."""
    pot = _hernquist()
    xyz = jnp.asarray([8.0, 1.0, 0.5], dtype=jnp.float32)
    assert func(pot, xyz, 0).dtype == jnp.float32
    assert func(pot, xyz, 0.0).dtype == jnp.float32


@POTS
@pytest.mark.parametrize("func", FUNCS, ids=lambda f: f.__name__)
def test_integer_position_matches_float(
    pot: gp.AbstractPotential, func: object
) -> None:
    """An integer position gives the value of the same position as floats."""
    got = func(pot, jnp.asarray([8, 1, 0]), 0)
    want = func(pot, jnp.asarray([8.0, 1.0, 0.0]), 0.0)
    assert got.dtype == want.dtype
    assert jnp.array_equal(got, want, equal_nan=True)


def test_x64_off_unchanged() -> None:
    """With x64 off, everything is float32, as before."""
    with jax.enable_x64(False):  # noqa: FBT003
        # Not `M`: it was built with x64 on, so it holds a float64.
        pot = gp.HernquistPotential(
            m_tot=u.Q(1e11, "Msun"), r_s=u.Q(5, "kpc"), units="galactic"
        )
        for func in FUNCS:
            assert func(pot, jnp.asarray([8.0, 1.0, 0.5]), 0.0).dtype == jnp.float32


@pytest.mark.parametrize(
    "make",
    [
        lambda h: gp.FlattenedInThePotential(h, q_z=np.float64(2.0)),
        lambda h: gp.TriaxialInThePotential(h, y_over_x=np.float64(2.0)),
    ],
    ids=["flattened", "triaxial"],
)
def test_float64_parameter_does_not_warn(make: object) -> None:
    """A strongly typed float64 parameter is cast to the position's dtype."""
    xyz = jnp.asarray([8.0, 1.0, 0.5], dtype=jnp.float32)
    assert gp.potential(make(_hernquist()), xyz, jnp.float32(0)).dtype == jnp.float32


def test_multipole_profile_gradient_float32() -> None:
    """float32 positions work through the analytic gradient (a vjp)."""
    mp = gp.MultipoleProfilePotential.from_potential(
        _hernquist(),
        r_min=u.Q(1e-2, "kpc"),
        r_max=u.Q(1e4, "kpc"),
        n_r=64,
        l_max=0,
        symmetry="spherical",
    )
    xyz = jnp.asarray([3.0, 1.0, 2.0])
    got = gp.gradient(mp, xyz.astype(jnp.float32), jnp.float32(0.5))
    assert jnp.allclose(got, gp.gradient(mp, xyz, 0.5), rtol=1e-5)
