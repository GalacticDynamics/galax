"""Test `galax.potential.params.eval_params` and `AbstractPotential.params`."""

import subprocess
import sys

import jax
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
from galax.potential.params import eval_params

T0 = u.Q(0.0, "Gyr")


def _hernquist() -> gp.HernquistPotential:
    return gp.HernquistPotential(
        m_tot=u.Q(1e12, "Msun"), r_s=u.Q(5.0, "kpc"), units="galactic"
    )


def _zhao() -> gp.ZhaoPotential:
    return gp.ZhaoPotential(
        m=u.Q(1e12, "Msun"),
        r_s=u.Q(10.0, "kpc"),
        alpha=1.0,
        beta=3.0,
        gamma=1.0,
        units="galactic",
    )


def _shrinking_kepler() -> gp.KeplerPotential:
    m_tot = gp.params.LinearParameter(
        slope=u.Q(-1e3, "Msun/yr"),
        point_time=u.Q(0, "Myr"),
        point_value=u.Q(1e12, "Msun"),
    )
    return gp.KeplerPotential(m_tot=m_tot, units="galactic")


def _multipole() -> gp.MultipoleProfilePotential:
    return gp.MultipoleProfilePotential.from_potential(
        _hernquist(),
        r_min=u.Q(1e-2, "kpc"),
        r_max=u.Q(1e4, "kpc"),
        n_r=64,
        l_max=0,
        symmetry="spherical",
    )


def _assert_dicts_close(a: dict, b: dict) -> None:
    assert set(a) == set(b)
    for k, x in a.items():
        y = b[k]
        if isinstance(x, u.AbstractQuantity):
            assert x.unit == y.unit, k
            x, y = x.value, y.value  # noqa: PLW2901
        assert jnp.allclose(x, y), k


def test_matches_hand_written_dict() -> None:
    """It reproduces the dict the builtins build by hand."""
    pot = _hernquist()
    expected = {
        "G": pot.constants["G"].value,
        "m_tot": pot.m_tot(T0, ustrip=pot.units["mass"]),
        "r_s": pot.r_s(T0, ustrip=pot.units["length"]),
    }
    _assert_dicts_close(eval_params(pot, T0), expected)


def test_matches_zhao_private_params() -> None:
    """Dimensionless parameters strip against ``units["dimensionless"]``."""
    pot = _zhao()
    _assert_dicts_close(eval_params(pot, T0), pot._params(T0))


def test_ustrip_false_returns_quantities() -> None:
    pot = _hernquist()
    out = eval_params(pot, T0, ustrip=False)
    assert all(isinstance(v, u.AbstractQuantity) for v in out.values())
    assert out["G"] == pot.constants["G"]
    assert u.dimension_of(out["m_tot"]) == u.dimension("mass")
    assert u.dimension_of(out["r_s"]) == u.dimension("length")


def test_ustrip_true_equals_stripped_quantities() -> None:
    pot = _zhao()
    stripped = eval_params(pot, T0)
    quantities = eval_params(pot, T0, ustrip=False)
    for name, field in type(pot).param_objs.items():
        unit = pot.units[field.dimensions]
        assert jnp.allclose(stripped[name], u.ustrip(unit, quantities[name])), name


def test_accepts_float_and_quantity_time() -> None:
    pot = _hernquist()
    _assert_dicts_close(eval_params(pot, 0.0), eval_params(pot, T0))


def test_batched_time() -> None:
    """A 1-D ``t`` puts the time axis on the time-dependent values only."""
    pot = _shrinking_kepler()
    out = eval_params(pot, u.Q(jnp.asarray([0.0, 1.0, 2.0]), "Gyr"))
    assert out["m_tot"].shape == (3,)
    assert out["G"].shape == ()


def test_time_dependent_parameter_varies() -> None:
    pot = _shrinking_kepler()
    early = eval_params(pot, u.Q(0.0, "Gyr"))["m_tot"]
    late = eval_params(pot, u.Q(1.0, "Gyr"))["m_tot"]
    assert not jnp.allclose(early, late)


def test_jit_with_traced_time() -> None:
    pot = _shrinking_kepler()
    t = u.Q(0.5, "Gyr")
    jitted = jax.jit(lambda t: eval_params(pot, t))(t)
    _assert_dicts_close(jitted, eval_params(pot, t))


def test_excludes_non_parameter_fields() -> None:
    """Only `ParameterField`s (and ``G``) appear, not ``l_max`` or ``symmetry``."""
    pot = _multipole()
    out = eval_params(pot, T0)
    assert set(out) == set(type(pot).param_objs) | {"G"}
    assert "l_max" not in out
    assert "symmetry" not in out


def test_composite_nested() -> None:
    kepler, hern = _shrinking_kepler(), _hernquist()
    comp = gp.CompositePotential(kepler=kepler, hernquist=hern)
    out = eval_params(comp, T0)
    assert set(out) == {"kepler", "hernquist"}
    _assert_dicts_close(out["kepler"], eval_params(kepler, T0))
    _assert_dicts_close(out["hernquist"], eval_params(hern, T0))


def test_composite_nested_quantities() -> None:
    comp = gp.CompositePotential(kepler=_shrinking_kepler(), hernquist=_hernquist())
    out = eval_params(comp, T0, ustrip=False)
    assert all(
        isinstance(v, u.AbstractQuantity) for sub in out.values() for v in sub.values()
    )


@pytest.mark.parametrize(
    "module",
    [
        "galax.potential.params",
        "galax.potential._src.base",
        "galax.potential._src.base_multi",
    ],
)
def test_import_has_no_cycle(module: str) -> None:
    """Every module that defines or exports `eval_params` imports on its own."""
    subprocess.run(  # noqa: S603  (fixed arguments)
        [sys.executable, "-I", "-c", f"import {module}"], check=True
    )


def test_missing_G_is_omitted() -> None:
    """A potential whose ``constants`` lack ``G`` (valid) still evaluates."""
    pot = gp.HarmonicOscillatorPotential(omega=1.0, units="galactic", constants={})
    assert set(eval_params(pot, T0)) == set(type(pot).param_objs)
    assert "G" not in eval_params(pot, T0, ustrip=False)


# --- AbstractPotential.params ---


def test_method_matches_function() -> None:
    pot = _zhao()
    _assert_dicts_close(pot.params(T0), eval_params(pot, T0))
    _assert_dicts_close(
        pot.params(T0, ustrip=False), eval_params(pot, T0, ustrip=False)
    )


def test_ustrip_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        _hernquist().params(T0, False)  # type: ignore[misc]  # noqa: FBT003


def test_composite_method() -> None:
    comp = gp.CompositePotential(kepler=_shrinking_kepler(), hernquist=_hernquist())
    out, expected = comp.params(T0), eval_params(comp, T0)
    assert set(out) == set(expected)
    for name in out:
        _assert_dicts_close(out[name], expected[name])


def test_default_ustrip_true() -> None:
    assert not isinstance(_hernquist().params(T0)["m_tot"], u.AbstractQuantity)
