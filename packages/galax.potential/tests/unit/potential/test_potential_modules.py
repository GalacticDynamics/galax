"""Conformance tests for the public per-potential modules.

Each module ``galax.potential.<name>`` must follow the contract in
``docs/superpowers/specs/2026-11-20-potential-modules-design.md``: bare arrays in
the potential's unit system, ``p = pot.params(t)``, single-point kernels. To add
a module, append one instance to `INSTANCES`.
"""

import ast
import dataclasses
import importlib
import inspect
import textwrap

from types import ModuleType

import jax
import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
from galax.potential._src.utils import r_spherical

G = "galactic"
M = u.Q(1e12, "Msun")
RS = u.Q(5, "kpc")

# One instance per module. Constructor arguments follow `test_symmetry.BUILTINS`.
INSTANCES: list[gp.AbstractSinglePotential] = [
    gp.PlummerPotential(m_tot=M, r_s=RS, units=G),
    gp.ZhaoPotential(m=M, r_s=RS, alpha=0.5, beta=4.0, gamma=1.0, units=G),
    gp.BurkertPotential(m=M, r_s=RS, units=G),
    gp.GaussianPotential(m_tot=M, r_s=RS, units=G),
    gp.HardCutoffNFWPotential(m=M, r_s=RS, r_t=u.Q(20, "kpc"), units=G),
    gp.HernquistPotential(m_tot=M, r_s=RS, units=G),
    gp.IsochronePotential(m_tot=M, r_s=RS, units=G),
    gp.JaffePotential(m_tot=M, r_s=RS, units=G),
    gp.KeplerPotential(m_tot=M, units=G),
    gp.NFWPotential(m=M, r_s=RS, units=G),
    gp.PowerLawCutoffPotential(m_tot=M, alpha=1.5, r_c=RS, units=G),
    gp.StoneOstriker15Potential(m_tot=M, r_c=u.Q(1, "kpc"), r_h=RS, units=G),
    gp.gNFWPotential(m=M, r_s=RS, gamma=0.5, units=G),
]

# Modules whose `laplacian` is not the trace of the hessian of `potential`.
NO_POISSON: set[str] = {"kuzmin", "kepler"}
# Modules whose `mass_enclosed` is not the integral of `4 pi r^2 density`.
NO_MASS_QUADRATURE: set[str] = {"kepler"}
# Modules whose hessian at the origin is not representable: Phi'/r diverges
# for a cusp (Zhao, gamma = 1) and the incomplete beta series underflows at the
# smallest normal radius. The autodiff reference there is a finite, wrong 0.
NO_ORIGIN_HESSIAN: set[str] = {
    "zhao",
    # Cusps with Phi'(0) != 0 or Phi' ~ r^(1-gamma): Phi'/r diverges.
    "nfw",
    "gnfw",
    "powerlawcutoff",
    "kepler",
    # Cored, so the limit is finite (Phi''(0)), but at the smallest normal
    # radius r the shell-theorem Phi' = G M / r^2 has M ~ r^3 underflow and
    # XLA fuses Phi'/r into 0 / (r * r * r) = NaN. A `spherical_hessian` issue.
    "burkert",
    "gaussian",
    "stoneostriker15",
}
# Modules whose kernels take a trailing time argument.
TIME_EXPLICIT: set[str] = {"monarietal2016bar"}

RADIAL = (
    "potential",
    "density",
    "laplacian",
    "mass_enclosed",
    "dpotential_dr",
    "d2potential_dr2",
)
VECTOR = ("gradient", "hessian")
FULL = ("potential", "density", "laplacian", "gradient", "hessian")

XYZ = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.3, 0.2, 0.1]])


def module_name(pot: gp.AbstractSinglePotential) -> str:
    """Lowercased class name without ``Potential``."""
    return type(pot).__name__.removesuffix("Potential").lower()


def module_for(pot: gp.AbstractSinglePotential) -> ModuleType:
    """Return the public module ``galax.potential.<name>`` of ``pot``."""
    return importlib.import_module(f"galax.potential.{module_name(pot)}")


def is_radial(pot: gp.AbstractSinglePotential, name: str) -> bool:
    return pot.symmetry == gp.Symmetry.SPHERICAL and name in RADIAL


def extra(pot: gp.AbstractSinglePotential) -> tuple[float, ...]:
    """Return the trailing ``t`` argument of a time-explicit module."""
    return (0.0,) if module_name(pot) in TIME_EXPLICIT else ()


def point_fn(pot, name):
    """``module.<name>`` as a function of one point ``xyz`` of shape ``(3,)``."""
    fn, p, ex = getattr(module_for(pot), name), pot.params(0.0), extra(pot)
    if is_radial(pot, name):
        return lambda x: fn(p, jnp.linalg.norm(x), *ex)
    return lambda x: fn(p, x, *ex)


def ids(pots):
    return [module_name(pot) for pot in pots]


@pytest.fixture(params=INSTANCES, ids=ids(INSTANCES))
def pot(request):
    return request.param


# ---------------------------------------------------------------------------


def test_module_exists_and_exports_contract(pot):
    module = module_for(pot)
    assert getattr(module, type(pot).__name__) is type(pot)

    spherical = pot.symmetry == gp.Symmetry.SPHERICAL
    radial = RADIAL if spherical else ()
    names = (*radial, *VECTOR) if spherical else FULL
    assert set(names) <= set(module.__all__)

    # The helper-built functions carry generic parameter names for now; the exact
    # `r` / `xyz` names are checked once the factories keep the source signature.
    # A trailing `*rest` (the helpers' spelling of `t`) counts for timed modules.
    timed = module_name(pot) in TIME_EXPLICIT
    for name in names:
        P = inspect.Parameter
        params = inspect.signature(getattr(module, name)).parameters.values()
        kinds = [q.kind for q in params]
        n_pos = sum(k in (P.POSITIONAL_ONLY, P.POSITIONAL_OR_KEYWORD) for k in kinds)
        has_rest = P.VAR_POSITIONAL in kinds
        assert next(iter(params)).name == "p", name
        assert n_pos + (has_rest and timed) == (3 if timed else 2), name


@pytest.mark.parametrize("shape", [(3,), (2, 3, 3), (0, 3)], ids=str)
@pytest.mark.parametrize("name", FULL)
def test_class_matches_vmap(pot, name, shape):
    xyz = jnp.resize(XYZ, shape) if 0 not in shape else jnp.zeros(shape)
    got = getattr(pot, name)(xyz, 0.0)
    batch = xyz.shape[:-1]
    flat = xyz.reshape(-1, 3)
    want = jax.vmap(point_fn(pot, name))(flat)
    want = want.reshape(*batch, *want.shape[1:])
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=1e-10)


def test_autodiff_reference(pot):
    module, p, ex = module_for(pot), pot.params(0.0), extra(pot)
    spherical = pot.symmetry == gp.Symmetry.SPHERICAL
    phi = point_fn(pot, "potential")  # a function of xyz, through the norm

    for xyz in XYZ:
        np.testing.assert_allclose(
            point_fn(pot, "gradient")(xyz), jax.grad(phi)(xyz), rtol=1e-8
        )
        hess = jax.hessian(phi)(xyz)
        np.testing.assert_allclose(point_fn(pot, "hessian")(xyz), hess, rtol=1e-8)
        if module_name(pot) not in NO_POISSON:
            np.testing.assert_allclose(
                point_fn(pot, "laplacian")(xyz), jnp.trace(hess), rtol=1e-8
            )

        point = jnp.linalg.norm(xyz) if spherical else xyz
        g = jax.jit(jax.grad(module.potential, 1))(p, point, *ex)
        h = jax.jit(jax.hessian(module.potential, 1))(p, point, *ex)
        assert jnp.all(jnp.isfinite(g))
        assert jnp.all(jnp.isfinite(h))

    if not spherical:
        return

    nodes, weights = np.polynomial.legendre.leggauss(64)
    for r in jnp.linalg.norm(XYZ, axis=-1):
        if module_name(pot) not in NO_MASS_QUADRATURE:
            # s = r u^2: a density like r^-1.5 gives an integrand s^0.5, which
            # Gauss-Legendre in s converges slowly on; in u it is analytic.
            u_ = (nodes + 1) / 2
            s = r * u_**2
            integral = jnp.sum(
                weights * 4 * jnp.pi * s**2 * module.density(p, s) * r * u_
            )
            np.testing.assert_allclose(module.mass_enclosed(p, r), integral, rtol=1e-8)
        np.testing.assert_allclose(
            module.dpotential_dr(p, r),
            p["G"] * module.mass_enclosed(p, r) / r**2,
            rtol=1e-10,
        )


def test_origin_matches_autodiff_where_finite(pot):
    if pot.symmetry != gp.Symmetry.SPHERICAL:
        pytest.skip("origin handling is a spherical-kernel contract")
    module, p, x0 = module_for(pot), pot.params(0.0), jnp.zeros(3)
    unit = pot.units["length"]

    def phi(x):
        return module.potential(p, r_spherical(x, unit))

    pairs = [(module.gradient(p, x0), jax.grad(phi)(x0))]
    if module_name(pot) not in NO_ORIGIN_HESSIAN:
        pairs.append((module.hessian(p, x0), jax.hessian(phi)(x0)))
    for got, ref in pairs:
        assert jnp.all(jnp.isfinite(got))
        ok = np.isfinite(ref)
        np.testing.assert_allclose(np.asarray(got)[ok], np.asarray(ref)[ok], rtol=1e-8)


def test_param_gradient(pot):
    p, ex = pot.params(0.0), extra(pot)
    if "m_tot" not in p:
        pytest.skip("no m_tot parameter")
    module, xyz = module_for(pot), XYZ[0]

    def via_module(m):
        return module.gradient({**p, "m_tot": m}, xyz, *ex).sum()

    def via_class(m):
        return dataclasses.replace(pot, m_tot=m).gradient(xyz, 0.0).sum()

    got = jax.grad(via_module)(p["m_tot"])
    assert jnp.isfinite(got)
    np.testing.assert_allclose(got, jax.grad(via_class)(p["m_tot"]), rtol=1e-10)


@pytest.mark.parametrize("name", ["potential", "density"])
def test_class_methods_call_module_kernels(pot, name):
    """``_<name>`` of the class calls the module-level ``<name>``: no formula twice."""
    module = module_for(pot)
    method = inspect.unwrap(getattr(type(pot), f"_{name}"))
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert name in called
    assert method.__globals__[name] is getattr(module, name)
