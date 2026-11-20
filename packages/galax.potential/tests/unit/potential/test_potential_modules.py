"""Conformance tests for the public per-potential modules.

Each module ``galax.potential.<name>`` must follow the contract in
``docs/potential_modules.rst``: bare arrays in the potential's unit
system, ``p = pot.params(t)``, single-point kernels. To add
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
    gp.KuzminPotential(m_tot=M, r_s=RS, units=G),
    gp.KuzminKutuzovPotential(m_tot=M, a=RS, c=u.Q(1, "kpc"), units=G),
    gp.LongMuraliBarPotential(
        m_tot=M, a=RS, b=u.Q(1, "kpc"), c=u.Q(1, "kpc"), alpha=u.Q(0.3, "rad"), units=G
    ),
    gp.MiyamotoNagaiPotential(m_tot=M, a=RS, b=u.Q(1, "kpc"), units=G),
    gp.MonariEtAl2016BarPotential(
        alpha=u.Q(0.01, ""),
        R0=u.Q(8.0, "kpc"),
        v0=u.Q(220.0, "km/s"),
        phi_b=u.Q(25, "deg"),
        Omega=u.Q(52.2, "km/(s kpc)"),
        units=G,
    ),
    gp.SatohPotential(m_tot=M, a=RS, b=u.Q(1, "kpc"), units=G),
    gp.LeeSutoTriaxialNFWPotential(m=M, r_s=RS, a1=1.0, a2=0.8, a3=0.6, units=G),
]

# Modules whose `mass_enclosed` is not the integral of `4 pi r^2 density`.
# Kepler: point mass, so the density is a delta.
NO_MASS_QUADRATURE: set[str] = {"kepler"}
# Modules whose kernels take a trailing time argument.
TIME_EXPLICIT: set[str] = {"monarietal2016bar"}
# A time at which Omega * t != 0 for the rotating bar.
T = 37.0
# Modules in `galax.potential.__all__` that are not a per-potential module.
NOT_PER_POTENTIAL = {"harmonic", "io", "multipole_profile", "params", "plot", "scf"}

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
    return (T,) if module_name(pot) in TIME_EXPLICIT else ()


def point_fn(pot, name):
    """``module.<name>`` as a function of one point ``xyz`` of shape ``(3,)``."""
    fn, p, ex = getattr(module_for(pot), name), pot.params(T), extra(pot)
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

    timed = module_name(pot) in TIME_EXPLICIT
    for name in names:
        point = "r" if is_radial(pot, name) else "xyz"
        want = ("p", point, *(("t",) if timed else ()))
        params = inspect.signature(getattr(module, name)).parameters.values()
        assert tuple(x.name for x in params) == want, name
        assert all(
            x.kind in (x.POSITIONAL_ONLY, x.POSITIONAL_OR_KEYWORD) for x in params
        ), name


def test_instances_cover_per_potential_modules():
    """Every per-potential module in `galax.potential.__all__` has an instance."""
    public = {
        n
        for n in gp.__all__
        if isinstance(getattr(gp, n), ModuleType) and n not in NOT_PER_POTENTIAL
    }
    assert public == {module_name(pot) for pot in INSTANCES}


@pytest.mark.parametrize("shape", [(3,), (2, 3, 3), (0, 3)], ids=str)
@pytest.mark.parametrize("name", FULL)
def test_class_matches_vmap(pot, name, shape):
    xyz = jnp.resize(XYZ, shape) if 0 not in shape else jnp.zeros(shape)
    got = getattr(pot, name)(xyz, T)
    batch = xyz.shape[:-1]
    flat = xyz.reshape(-1, 3)
    want = jax.vmap(point_fn(pot, name))(flat)
    want = want.reshape(*batch, *want.shape[1:])
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=1e-10)


def test_autodiff_reference(pot):
    module, p, ex = module_for(pot), pot.params(T), extra(pot)
    spherical = pot.symmetry == gp.Symmetry.SPHERICAL
    phi = point_fn(pot, "potential")  # a function of xyz, through the norm

    for xyz in XYZ:
        np.testing.assert_allclose(
            point_fn(pot, "gradient")(xyz), jax.grad(phi)(xyz), rtol=1e-8
        )
        hess = jax.hessian(phi)(xyz)
        np.testing.assert_allclose(point_fn(pot, "hessian")(xyz), hess, rtol=1e-8)
        # Where the laplacian is exactly 0 (a thin disk, a point mass, outside a
        # bar) the hessian trace is round-off, a few 1e-16 of the largest entry.
        np.testing.assert_allclose(
            point_fn(pot, "laplacian")(xyz),
            jnp.trace(hess),
            rtol=1e-8,
            atol=1e-12 * np.abs(hess).max(),
        )

        point = jnp.linalg.norm(xyz) if spherical else xyz
        g = jax.jit(jax.grad(module.potential, 1))(p, point, *ex)
        h = jax.jit(jax.hessian(module.potential, 1))(p, point, *ex)
        assert jnp.all(jnp.isfinite(g))
        assert jnp.all(jnp.isfinite(h))

    if not spherical:
        return

    radii = jnp.linalg.norm(XYZ, axis=-1)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    if module_name(pot) not in NO_MASS_QUADRATURE:
        for r in radii:
            # s = r u^2: a density like r^-1.5 gives an integrand s^0.5, which
            # Gauss-Legendre in s converges slowly on; in u it is analytic.
            u_ = (nodes + 1) / 2
            s = r * u_**2
            integral = jnp.sum(
                weights * 4 * jnp.pi * s**2 * module.density(p, s) * r * u_
            )
            np.testing.assert_allclose(module.mass_enclosed(p, r), integral, rtol=1e-8)

    # r = 30 kpc is outside the cutoff radius of 20 kpc. Not in the quadrature
    # above: the density jump at r_t breaks Gauss-Legendre convergence.
    if module_name(pot) == "hardcutoffnfw":
        radii = jnp.concat([radii, jnp.array([30.0])])
    for r in radii:
        np.testing.assert_allclose(
            module.dpotential_dr(p, r),
            p["G"] * module.mass_enclosed(p, r) / r**2,
            rtol=1e-10,
        )


def test_origin(pot):
    """Spherical origin rule: gradient 0, hessian `laplacian(p, 0) / 3 * I`."""
    if pot.symmetry != gp.Symmetry.SPHERICAL:
        pytest.skip("origin handling is a spherical-kernel contract")
    module, p, x0 = module_for(pot), pot.params(0.0), jnp.zeros(3)

    assert np.array_equal(module.gradient(p, x0), np.zeros(3))

    h0 = np.asarray(module.hessian(p, x0))
    assert not np.any(np.isnan(h0))
    np.testing.assert_array_equal(h0[~np.eye(3, dtype=bool)], 0)

    if np.isfinite(module.density(p, jnp.zeros(()))):
        # A core: Phi'/r -> Phi''(0), read off the transverse entry nearby.
        near = module.hessian(p, jnp.array([1e-6, 0.0, 0.0]))[1, 1]
        np.testing.assert_allclose(np.diag(h0), near, rtol=1e-6)
    else:
        assert np.all(np.diag(h0) == np.inf)


def test_param_gradient(pot):
    p, ex = pot.params(T), extra(pot)
    if "m_tot" not in p and "m" not in p:
        pytest.skip("no mass parameter")
    key = "m_tot" if "m_tot" in p else "m"
    module, xyz = module_for(pot), XYZ[0]

    def via_module(m):
        return module.gradient({**p, key: m}, xyz, *ex).sum()

    def via_class(m):
        return dataclasses.replace(pot, **{key: m}).gradient(xyz, T).sum()

    got = jax.grad(via_module)(p[key])
    assert jnp.isfinite(got)
    np.testing.assert_allclose(got, jax.grad(via_class)(p[key]), rtol=1e-10)


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


def test_kuzmin_density_is_zero():
    """The razor-thin disk has no volume density off the plane."""
    pot = gp.KuzminPotential(m_tot=M, r_s=RS, units=G)
    got = gp.kuzmin.density(pot.params(0.0), XYZ[0])
    assert got.shape == ()
    assert got == 0.0
    assert jnp.all(pot.density(XYZ, 0.0) == 0.0)
