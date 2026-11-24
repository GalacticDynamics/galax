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

# Multipole coefficients, of shape (l_max + 1, l_max + 1) with l_max = 2, with
# terms in m = 0, 1 and 2 and in both the cos and the sin coefficients.
SLM = jnp.zeros((3, 3)).at[0, 0].set(1.0).at[1, 0].set(0.2).at[1, 1].set(0.3)
SLM = SLM.at[2, 1].set(0.3).at[2, 2].set(-0.1)
TLM = jnp.zeros((3, 3)).at[1, 1].set(0.25).at[2, 2].set(0.15)

# At least one instance per module. Constructor arguments follow
# `test_symmetry.BUILTINS`. Length parameters avoid 1 kpc, where an exponent
# error (b**2 -> b**2.002) would not show.
INSTANCES: list[gp.AbstractSinglePotential] = [
    gp.PlummerPotential(m_tot=M, r_s=RS, units=G),
    gp.ZhaoPotential(m=M, r_s=RS, alpha=0.5, beta=4.0, gamma=1.0, units=G),
    # Cored: its origin hessian is the finite `laplacian(p, 0) / 3`.
    gp.ZhaoPotential(m=M, r_s=RS, alpha=0.5, beta=4.0, gamma=0.0, units=G),
    gp.BurkertPotential(m=M, r_s=RS, units=G),
    gp.GaussianPotential(m_tot=M, r_s=RS, units=G),
    gp.HardCutoffNFWPotential(m=M, r_s=RS, r_t=u.Q(20, "kpc"), units=G),
    gp.HernquistPotential(m_tot=M, r_s=RS, units=G),
    gp.IsochronePotential(m_tot=M, r_s=RS, units=G),
    gp.JaffePotential(m_tot=M, r_s=RS, units=G),
    gp.KeplerPotential(m_tot=M, units=G),
    gp.NFWPotential(m=M, r_s=RS, units=G),
    gp.PowerLawCutoffPotential(m_tot=M, alpha=1.5, r_c=RS, units=G),
    gp.StoneOstriker15Potential(m_tot=M, r_c=u.Q(0.8, "kpc"), r_h=RS, units=G),
    gp.gNFWPotential(m=M, r_s=RS, gamma=0.5, units=G),
    # gamma > 1: the gradient's jvp at the origin must still be 0.
    gp.gNFWPotential(m=M, r_s=RS, gamma=1.5, units=G),
    gp.KuzminPotential(m_tot=M, r_s=RS, units=G),
    gp.KuzminKutuzovPotential(m_tot=M, a=RS, c=u.Q(1, "kpc"), units=G),
    gp.LongMuraliBarPotential(
        m_tot=M,
        a=RS,
        b=u.Q(1.2, "kpc"),
        c=u.Q(0.7, "kpc"),  # b != c, so that a swap of the two shows
        alpha=u.Q(0.3, "rad"),
        units=G,
    ),
    gp.MiyamotoNagaiPotential(m_tot=M, a=RS, b=u.Q(1.3, "kpc"), units=G),
    gp.MonariEtAl2016BarPotential(
        alpha=u.Q(0.01, ""),
        R0=u.Q(8.0, "kpc"),
        v0=u.Q(220.0, "km/s"),
        phi_b=u.Q(25, "deg"),
        Omega=u.Q(52.2, "km/(s kpc)"),
        units=G,
    ),
    gp.SatohPotential(m_tot=M, a=RS, b=u.Q(1.3, "kpc"), units=G),
    gp.LeeSutoTriaxialNFWPotential(m=M, r_s=RS, a1=1.0, a2=0.8, a3=0.6, units=G),
    gp.MultipoleInnerPotential(m_tot=M, r_s=RS, l_max=2, Slm=SLM, Tlm=TLM, units=G),
    gp.MultipoleOuterPotential(m_tot=M, r_s=RS, l_max=2, Slm=SLM, Tlm=TLM, units=G),
    gp.MultipolePotential(
        m_tot=M, r_s=RS, l_max=2, ISlm=SLM, ITlm=TLM, OSlm=0.5 * SLM, OTlm=-TLM, units=G
    ),
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
    """Return module names, a count appended to the second and later of one."""
    names = [module_name(pot) for pot in pots]
    return [f"{n}{names[:i].count(n) or ''}" for i, n in enumerate(names)]


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
    # Entries that are zero in exact arithmetic differ at round-off.
    np.testing.assert_allclose(
        got, want, rtol=1e-10, atol=1e-13 * np.abs(want).max(initial=0.0)
    )


def test_autodiff_reference(pot):
    module, p, ex = module_for(pot), pot.params(T), extra(pot)
    spherical = pot.symmetry == gp.Symmetry.SPHERICAL
    phi = point_fn(pot, "potential")  # a function of xyz, through the norm

    for xyz in XYZ:
        # An entry that is zero in exact arithmetic can come out as round-off
        # that differs between the two paths, so also allow a few 1e-12 of the
        # largest entry.
        grad = jax.grad(phi)(xyz)
        np.testing.assert_allclose(
            point_fn(pot, "gradient")(xyz),
            grad,
            rtol=1e-8,
            atol=1e-12 * np.abs(grad).max(),
        )
        hess = jax.hessian(phi)(xyz)
        np.testing.assert_allclose(
            point_fn(pot, "hessian")(xyz),
            hess,
            rtol=1e-8,
            atol=1e-12 * np.abs(hess).max(),
        )
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


# Off-centre radii (kpc), >= 0.1 r_s and away from HardCutoffNFW's r_t = 20.
RADII = (0.5, 1.7, 3.0, 7.0, 30.0)


def test_d2potential_dr2_poisson_identity(pot):
    """``Phi'' = 4 pi G rho - 2 Phi'/r`` (Poisson in radial form), and ``(Phi')'``."""
    if pot.symmetry != gp.Symmetry.SPHERICAL:
        pytest.skip("radial kernels only")
    module, p = module_for(pot), pot.params(0.0)
    d1, d2 = module.dpotential_dr, module.d2potential_dr2
    for r in map(jnp.asarray, RADII):
        poisson = 4 * jnp.pi * p["G"] * module.density(p, r) - 2 * d1(p, r) / r
        # Measured error <= 2.5e-14 on all 13 spherical modules; 1e-9 leaves
        # a margin of 4e4.
        np.testing.assert_allclose(d2(p, r), poisson, rtol=1e-9, err_msg=f"r={r}")
        # Central difference: truncation and roundoff, <= 1.2e-9 measured.
        h = 1e-5 * r
        fd = (d1(p, r + h) - d1(p, r - h)) / (2 * h)
        np.testing.assert_allclose(d2(p, r), fd, rtol=1e-6, err_msg=f"r={r}")


def test_origin(pot):
    """Spherical origin rule: gradient 0, hessian `laplacian(p, 0) / 3 * I`."""
    if pot.symmetry != gp.Symmetry.SPHERICAL:
        pytest.skip("origin handling is a spherical-kernel contract")
    module, p, x0 = module_for(pot), pot.params(0.0), jnp.zeros(3)
    key = "m_tot" if "m_tot" in p else "m"

    def at_origin(fn):
        """``fn(p, 0)`` as a function of the mass parameter."""
        return lambda m: fn({**p, key: m}, x0)

    assert np.array_equal(module.gradient(p, x0), np.zeros(3))
    # The outer `where` discards the tangent; the inner one keeps NaN out of
    # reverse mode and the parameter derivative.
    assert np.array_equal(jax.jacfwd(module.gradient, 1)(p, x0), np.zeros((3, 3)))
    assert np.all(np.isfinite(jax.jacrev(module.gradient, 1)(p, x0)))
    assert np.all(np.isfinite(jax.jacfwd(at_origin(module.gradient))(p[key])))

    h0 = np.asarray(module.hessian(p, x0))
    assert not np.any(np.isnan(h0))
    np.testing.assert_array_equal(h0[~np.eye(3, dtype=bool)], 0)

    if np.isfinite(module.density(p, jnp.zeros(()))):
        # A core: Phi'/r -> Phi''(0), read off the transverse entry nearby.
        near = module.hessian(p, jnp.array([1e-6, 0.0, 0.0]))[1, 1]
        np.testing.assert_allclose(np.diag(h0), near, rtol=1e-6)
        assert np.all(np.isfinite(jax.jacrev(module.hessian, 1)(p, x0)))
        assert np.all(np.isfinite(jax.jacfwd(at_origin(module.hessian))(p[key])))
    else:
        assert np.all(np.diag(h0) == np.inf)


def test_origin_hessian_without_jit(pot):
    """No NaN at the origin even when nothing rewrites `inf * 0` (`disable_jit`)."""
    if pot.symmetry != gp.Symmetry.SPHERICAL:
        pytest.skip("origin handling is a spherical-kernel contract")
    module, p = module_for(pot), pot.params(0.0)

    with jax.disable_jit():
        h0 = np.asarray(module.hessian(p, jnp.zeros(3)))

    assert not np.any(np.isnan(h0))
    np.testing.assert_array_equal(h0[~np.eye(3, dtype=bool)], 0)


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


def as_float32(tree):
    """Cast every floating leaf of ``tree`` to float32."""

    def cast(a):
        return a.astype(jnp.float32) if jnp.issubdtype(a.dtype, jnp.floating) else a

    return jax.tree.map(cast, tree)


def to_si(pot, m):
    """``pot`` in SI units with mass parameter ``m``."""
    kw = {
        f.name: getattr(pot, f.name)
        for f in dataclasses.fields(pot)
        if f.name not in ("units", "constants")
    }
    key = "m_tot" if "m_tot" in kw else "m"
    return type(pot)(**{**kw, key: m}, units="si")


# float32 in SI units: r ~ 1e19 m, so r**3 overflows float32 (max ~3e38).
# 1e7 Msun, since 1e12 Msun in kg overflows float32 too.
M_SI = u.Q(1e7, "Msun")
F32_SI_RADII = (1e16, 3e17, 3.0857e18, 1e19)  # m
# Wrong in float32 SI before the spherical kernels see them, and the class
# methods equally so: their radial derivatives are autodiff of a closed form.
_PHI1_SUM = "Phi' forms r**2 + r_s**2 ~ 2e40"
_PHI2_INV = "Phi'' forms r**-3, which underflows float32 beyond 4.4e12 m"
F32_SI_EXCLUDED = {
    "gradient": {
        "plummer": _PHI1_SUM,
        "isochrone": _PHI1_SUM,
        "hernquist": "Phi' forms (r + r_s)**2 ~ 2e40",
    },
    "hessian": {
        "plummer": _PHI1_SUM,
        "isochrone": _PHI1_SUM,
        "hernquist": "Phi' forms (r + r_s)**2 ~ 2e40",
        "jaffe": _PHI2_INV,
        "kepler": _PHI2_INV,
    },
}
# Their Phi' forms r**-2, which underflows float32 beyond 9.2e18 m.
F32_SI_MAX_R = {"jaffe": 9.2e18, "kepler": 9.2e18}


@pytest.mark.parametrize("name", VECTOR)
def test_float32_si_gradient_hessian(pot, name):
    """The jitted float32 ``gradient`` and ``hessian`` in SI units match float64."""
    if pot.symmetry != gp.Symmetry.SPHERICAL:
        pytest.skip("radial kernels only")
    if module_name(pot) in F32_SI_EXCLUDED[name]:
        pytest.skip(F32_SI_EXCLUDED[name][module_name(pot)])
    r_max = F32_SI_MAX_R.get(module_name(pot), np.inf)
    fn = getattr(module_for(pot), name)
    p64 = to_si(pot, M_SI).params(0.0)
    p32 = as_float32(p64)
    for r in (r for r in F32_SI_RADII if r < r_max):
        x64 = r * jnp.array([2.0, 3.0, 6.0]) / 7
        got, want = jax.jit(fn)(p32, x64.astype(jnp.float32)), fn(p64, x64)
        assert got.dtype == jnp.float32
        # Relative to the largest entry: near a core's centre the outer term
        # Phi'' - Phi'/r cancels to below float32 resolution.
        atol = 1e-4 * np.abs(want).max()
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=atol, err_msg=f"r={r}")


def test_float32_si_zhao_class_gradient():
    """Zhao's class gradient in float32 SI keeps its pre-PR (32642250) value."""
    pot = gp.ZhaoPotential(m=M_SI, r_s=RS, alpha=0.5, beta=4.0, gamma=1.0, units="si")
    xyz = jnp.array([3.0857e18, 0.0, 0.0], dtype=jnp.float32)  # 0.1 kpc
    # `_gradient`: the public `gradient` promotes to float64 under x64.
    got = as_float32(pot)._gradient(xyz, jnp.float32(0.0))
    assert got.dtype == jnp.float32
    np.testing.assert_allclose(got, [9.5148315e-14, 0.0, 0.0], rtol=1e-4)


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
