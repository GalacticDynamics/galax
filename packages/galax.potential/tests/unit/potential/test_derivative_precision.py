r"""Gradients and Hessians of the built-in potentials, to a few ulp.

Each is compared, from $10^{-8}$ to $10^8$ scale lengths, with a 50-digit
mpmath reference, in units of ``eps * max|reference|``: an error relative to
the magnitude of the gradient or Hessian, so a component that is small because
two large ones nearly cancel is held to the precision of the large ones.

The references are independent of the code under test:

- spherical potentials from the density alone, by Newton's shell theorem,
  $\Phi' = G M(r)/r^2$ and $\Phi'' = 4\pi G\rho - 2 G M(r)/r^3$, with $M(r)$
  in closed form or by quadrature;
- the rest by differentiating the transcribed $\Phi$ with `mpmath.diff`.

The Hessian is checked both as `_hessian` and as `jax.jacfwd` / `jax.jacrev`
of `_gradient` -- the first is how the Staeckel fudge forms it.
"""

import functools as ft

from collections.abc import Callable
from typing import Any, NamedTuple

import equinox as eqx
import jax
import mpmath as mp
import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp

mp.mp.dps = 50
EPS = float(np.finfo(np.float64).eps)
TOL = 16
r"""Allowed error, in ulp of the magnitude.

Most potentials stay below 6. What is left above that is not cancellation in
the potential's formulas: in the Burkert core $\Phi'' = 4\pi G\rho - 2GM/r^3$
cancels 3:2 (~9). See `TOL_OVERRIDE` for the one case above this.
"""

TOL_OVERRIDE = {"Zhao-c": 40}
r"""The Zhao model with $(\alpha, \beta, \gamma) = (2, 5, 0)$ reaches ~32.

Its mass is $B(6, 4, \chi)$, whose series terminate and alternate with
coefficients up to 10, so $B$ itself is good to only ~15 ulp, however it is
summed. Then $\Phi'' = 4\pi G\rho - 2GM/r^3$ cancels at 2-3 $r_s$, which
doubles that, and powers like $(1 + u^{1/2})^{10}$ multiply rounding by their
exponent.
"""

T0 = u.Q(0.0, "Myr")
# Every decade, plus the switch points of the piecewise forms.
SCALES = (1e-8, 1e-6, 1e-4, 1e-2, 0.5, 0.8, 1.0, 2.0, 3.0, 1e2, 1e4, 1e6, 1e8)
DIRECTIONS = ((0.48, -0.6, 0.64), (0.1, 0.2, -0.9746794344808963))

M, RS, KPC = u.Q(1e12, "Msun"), u.Q(10.0, "kpc"), "kpc"
Fn = Callable[..., mp.mpf]


def _mp(pot: Any, name: str) -> mp.mpf:
    """Return a parameter of ``pot`` at t = 0, in its unit system, as an mpf."""
    q = getattr(pot, name)(T0)
    return mp.mpf(float(u.ustrip(pot.units[u.dimension_of(q)], q)))


def _G(pot: Any) -> mp.mpf:
    return mp.mpf(float(pot.constants["G"].value))


def _mass_from_rho(rho: Fn, r: mp.mpf, a: mp.mpf) -> mp.mpf:
    # Breakpoints every decade of the scale length keep tanh-sinh accurate.
    brk = [a * mp.mpf(10) ** k for k in range(-9, 10) if a * mp.mpf(10) ** k < r]
    return 4 * mp.pi * mp.quad(lambda s: rho(s) * s**2, [mp.mpf(0), *brk, r])


class Spherical(NamedTuple):
    pot: Any
    rho: Fn
    mass: Fn | None  # None: by quadrature of rho
    scale: mp.mpf


class General(NamedTuple):
    pot: Any
    phi: Fn  # Phi(x, y, z)
    scale: mp.mpf


CASES: dict[str, Callable[[], Spherical | General]] = {}


def case(name: str) -> Callable[[Callable[[], Any]], Callable[[], Any]]:
    def register(build: Callable[[], Any]) -> Callable[[], Any]:
        CASES[name] = build
        return build

    return register


# -------------------------------------------------------------------
# Spherical: density (and enclosed mass, where closed-form)


@case("Kepler")
def _kepler() -> Spherical:
    pot = gp.KeplerPotential(m_tot=M, units="galactic")
    m = _mp(pot, "m_tot")
    return Spherical(pot, lambda _: mp.mpf(0), lambda _: m, mp.mpf(10))


@case("Plummer")
def _plummer() -> Spherical:
    pot = gp.PlummerPotential(m_tot=M, r_s=RS, units="galactic")
    m, b = _mp(pot, "m_tot"), _mp(pot, "r_s")

    def rho(r: mp.mpf) -> mp.mpf:
        return 3 * m / (4 * mp.pi * b**3) * (1 + (r / b) ** 2) ** -2.5

    return Spherical(pot, rho, None, b)


@case("Hernquist")
def _hernquist() -> Spherical:
    pot = gp.HernquistPotential(m_tot=M, r_s=RS, units="galactic")
    m, a = _mp(pot, "m_tot"), _mp(pot, "r_s")
    return Spherical(
        pot,
        lambda r: m * a / (2 * mp.pi * r * (r + a) ** 3),
        lambda r: m * r**2 / (r + a) ** 2,
        a,
    )


@case("Jaffe")
def _jaffe() -> Spherical:
    pot = gp.JaffePotential(m_tot=M, r_s=RS, units="galactic")
    m, a = _mp(pot, "m_tot"), _mp(pot, "r_s")
    return Spherical(
        pot,
        lambda r: m * a / (4 * mp.pi * r**2 * (r + a) ** 2),
        lambda r: m * r / (r + a),
        a,
    )


@case("Isochrone")
def _isochrone() -> Spherical:
    pot = gp.IsochronePotential(m_tot=M, r_s=RS, units="galactic")
    m, b = _mp(pot, "m_tot"), _mp(pot, "r_s")

    def mass(r: mp.mpf) -> mp.mpf:
        s = mp.sqrt(b**2 + r**2)
        return m * r**3 / (s * (b + s) ** 2)

    def rho(r: mp.mpf) -> mp.mpf:
        s = mp.sqrt(b**2 + r**2)
        num = 3 * (b + s) * s**2 - r**2 * (b + 3 * s)
        return m * num / (4 * mp.pi * (b + s) ** 3 * s**3)

    return Spherical(pot, rho, mass, b)


def _nfw(pot: Any, r_t: mp.mpf) -> Spherical:
    m, a = _mp(pot, "m"), _mp(pot, "r_s")

    def rho(r: mp.mpf) -> mp.mpf:
        if r > r_t:
            return mp.mpf(0)
        return m / (4 * mp.pi * a**3) / ((r / a) * (1 + r / a) ** 2)

    def mass(r: mp.mpf) -> mp.mpf:
        x = min(r, r_t) / a
        return m * (mp.log1p(x) - x / (1 + x))

    return Spherical(pot, rho, mass, a)


@case("NFW")
def _nfw_plain() -> Spherical:
    return _nfw(gp.NFWPotential(m=M, r_s=RS, units="galactic"), mp.inf)


@case("HardCutoffNFW")
def _nfw_cut() -> Spherical:
    # r_t off the sampled radii: Phi'' jumps there.
    r_t = u.Q(123.0, KPC)
    pot = gp.HardCutoffNFWPotential(m=M, r_s=RS, r_t=r_t, units="galactic")
    return _nfw(pot, _mp(pot, "r_t"))


@case("Burkert")
def _burkert() -> Spherical:
    pot = gp.BurkertPotential(m=M, r_s=RS, units="galactic")
    m, a = _mp(pot, "m"), _mp(pot, "r_s")
    C = 3 * mp.log(2) - mp.pi / 2

    def mass(r: mp.mpf) -> mp.mpf:
        x = r / a
        return m / C * (2 * mp.log1p(x) + mp.log1p(x**2) - 2 * mp.atan(x))

    return Spherical(
        pot, lambda r: m / (mp.pi * C) / ((r + a) * (r**2 + a**2)), mass, a
    )


def _gnfw(gamma: float) -> Spherical:
    pot = gp.gNFWPotential(m=M, r_s=RS, gamma=gamma, units="galactic")
    m, a, g = _mp(pot, "m"), _mp(pot, "r_s"), mp.mpf(gamma)

    def rho(r: mp.mpf) -> mp.mpf:
        return m / (4 * mp.pi * a**3) * (r / a) ** -g * (1 + r / a) ** (g - 3)

    return Spherical(pot, rho, None, a)


for _g in (0.0, 0.5, 1.5):
    case(f"gNFW{_g}")(ft.partial(_gnfw, _g))


def _zhao(alpha: float, beta: float, gamma: float) -> Spherical:
    pot = gp.ZhaoPotential(
        m=M, r_s=RS, alpha=alpha, beta=beta, gamma=gamma, units="galactic"
    )
    m, a = _mp(pot, "m"), _mp(pot, "r_s")
    al, be, ga = mp.mpf(alpha), mp.mpf(beta), mp.mpf(gamma)

    def shape(r: mp.mpf) -> mp.mpf:
        return (r / a) ** -ga * (1 + (r / a) ** (1 / al)) ** (-(be - ga) * al)

    norm = m / _mass_from_rho(shape, a, a)  # m is the mass within r_s
    return Spherical(pot, lambda r: norm * shape(r), None, a)


for _n, _abg in {
    "a": (0.5, 4.0, 1.0),
    "b": (1.0, 3.0, 0.5),
    "c": (2.0, 5.0, 0.0),
}.items():
    case(f"Zhao-{_n}")(ft.partial(_zhao, *_abg))


@case("PowerLawCutoff")
def _plcut() -> Spherical:
    pot = gp.PowerLawCutoffPotential(m_tot=M, alpha=1.8, r_c=RS, units="galactic")
    m, al, rc = _mp(pot, "m_tot"), _mp(pot, "alpha"), _mp(pot, "r_c")
    rho0 = m / (2 * mp.pi * mp.gamma(mp.mpf(1.5) - al / 2) * rc**3)
    return Spherical(
        pot, lambda r: rho0 * (rc / r) ** al * mp.exp(-((r / rc) ** 2)), None, rc
    )


@case("StoneOstriker15")
def _so15() -> Spherical:
    pot = gp.StoneOstriker15Potential(
        m_tot=M, r_c=u.Q(1.0, KPC), r_h=RS, units="galactic"
    )
    m, rc, rh = _mp(pot, "m_tot"), _mp(pot, "r_c"), _mp(pot, "r_h")
    rhoc = m * (rh + rc) / (2 * mp.pi**2 * rc**2 * rh**2)

    def mass(r: mp.mpf) -> mp.mpf:
        return (
            2 * m / (mp.pi * (rh - rc)) * (rh * mp.atan(r / rh) - rc * mp.atan(r / rc))
        )

    def rho(r: mp.mpf) -> mp.mpf:
        return rhoc / ((1 + (r / rc) ** 2) * (1 + (r / rh) ** 2))

    return Spherical(pot, rho, mass, rc)


# -------------------------------------------------------------------
# Non-spherical: Phi transcribed


@case("Vogelsberger08TriaxialNFW")
def _vogelsberger() -> General:
    pot = gp.Vogelsberger08TriaxialNFWPotential(
        m=M, r_s=RS, q1=0.8, a_r=1.5, units="galactic"
    )
    Gm, a, q1 = _G(pot) * _mp(pot, "m"), _mp(pot, "r_s"), _mp(pot, "q1")
    ra = _mp(pot, "a_r") * a

    def phi(x: mp.mpf, y: mp.mpf, z: mp.mpf) -> mp.mpf:
        re = mp.sqrt(x**2 + y**2 / q1**2 + z**2 / (3 - q1**2))
        rt = (ra + mp.sqrt(x**2 + y**2 + z**2)) * re / (ra + re)
        return -Gm * mp.log1p(rt / a) / rt

    return General(pot, phi, a)


@case("LeeSutoTriaxialNFW")
def _leesuto() -> General:
    pot = gp.LeeSutoTriaxialNFWPotential(
        m=M, r_s=RS, a1=1.0, a2=0.9, a3=0.8, units="galactic"
    )
    a, a1 = _mp(pot, "r_s"), _mp(pot, "a1")
    eb, ec = 1 - (_mp(pot, "a2") / a1) ** 2, 1 - (_mp(pot, "a3") / a1) ** 2
    vc2 = _G(pot) * _mp(pot, "m") / a
    phi0 = vc2 / (mp.log(2) - 0.5 + (mp.log(2) - 0.75) * (eb + ec))

    def phi(x: mp.mpf, y: mp.mpf, z: mp.mpf) -> mp.mpf:
        r2 = x**2 + y**2 + z**2
        s = mp.sqrt(r2) / a
        L = mp.log1p(s)
        F1 = -L / s
        F2 = -mp.mpf(1) / 3 + (2 * s**2 - 3 * s + 6) / (6 * s**2) + (1 / s - s**-3) * L
        F3 = (s**2 - 3 * s - 6) / (2 * s**2 * (1 + s)) + 3 * s**-3 * L
        angular = (eb * y**2 + ec * z**2) / r2
        return phi0 * (F1 + (eb + ec) / 2 * F2 + angular / 2 * F3)

    return General(pot, phi, a)


@case("LongMuraliBar")
def _longmurali() -> General:
    pot = gp.LongMuraliBarPotential(
        m_tot=M,
        a=u.Q(4.0, KPC),
        b=u.Q(1.0, KPC),
        c=u.Q(0.5, KPC),
        alpha=u.Q(0.3, "rad"),
        units="galactic",
    )
    Gm = _G(pot) * _mp(pot, "m_tot")
    a, b, c, al = _mp(pot, "a"), _mp(pot, "b"), _mp(pot, "c"), _mp(pot, "alpha")

    def phi(x: mp.mpf, y: mp.mpf, z: mp.mpf) -> mp.mpf:
        xp = x * mp.cos(al) + y * mp.sin(al)
        yp = -x * mp.sin(al) + y * mp.cos(al)
        eta2 = yp**2 + (b + mp.sqrt(c**2 + z**2)) ** 2
        Tp, Tm = mp.sqrt((a + xp) ** 2 + eta2), mp.sqrt((a - xp) ** 2 + eta2)
        return Gm / (2 * a) * mp.log((xp - a + Tm) / (xp + a + Tp))

    return General(pot, phi, a)


@case("Logarithmic")
def _log() -> General:
    pot = gp.LogarithmicPotential(v_c=u.Q(200, "km/s"), r_s=RS, units="galactic")
    v, a = _mp(pot, "v_c"), _mp(pot, "r_s")
    return General(pot, lambda x, y, z: v**2 / 2 * mp.log(a**2 + x**2 + y**2 + z**2), a)


@case("LMJ09Logarithmic")
def _lmj09() -> General:
    pot = gp.LMJ09LogarithmicPotential(
        v_c=u.Q(200, "km/s"),
        r_s=RS,
        q1=1.0,
        q2=0.9,
        q3=0.8,
        phi=u.Q(0.4, "rad"),
        units="galactic",
    )
    v, a, ph = _mp(pot, "v_c"), _mp(pot, "r_s"), _mp(pot, "phi")
    q1, q2, q3 = _mp(pot, "q1"), _mp(pot, "q2"), _mp(pot, "q3")

    def phi(x: mp.mpf, y: mp.mpf, z: mp.mpf) -> mp.mpf:
        X, Y = x * mp.cos(ph) + y * mp.sin(ph), -x * mp.sin(ph) + y * mp.cos(ph)
        return v**2 / 2 * mp.log(a**2 + (X / q1) ** 2 + (Y / q2) ** 2 + (z / q3) ** 2)

    return General(pot, phi, a)


def _disc(cls: Any, form: Callable[..., mp.mpf]) -> General:
    pot = cls(m_tot=M, a=u.Q(3.0, KPC), b=u.Q(0.3, KPC), units="galactic")
    Gm, a, b = _G(pot) * _mp(pot, "m_tot"), _mp(pot, "a"), _mp(pot, "b")
    return General(pot, lambda x, y, z: -Gm / mp.sqrt(form(x, y, z, a, b)), a)


case("MiyamotoNagai")(
    ft.partial(
        _disc,
        gp.MiyamotoNagaiPotential,
        lambda x, y, z, a, b: x**2 + y**2 + (a + mp.sqrt(z**2 + b**2)) ** 2,
    )
)
case("Satoh")(
    ft.partial(
        _disc,
        gp.SatohPotential,
        lambda x, y, z, a, b: x**2 + y**2 + z**2 + a * (a + 2 * mp.sqrt(z**2 + b**2)),
    )
)


@case("Kuzmin")
def _kuzmin() -> General:  # the directions all have z != 0, off its kink
    pot = gp.KuzminPotential(m_tot=M, r_s=u.Q(3.0, KPC), units="galactic")
    Gm, a = _G(pot) * _mp(pot, "m_tot"), _mp(pot, "r_s")
    return General(
        pot, lambda x, y, z: -Gm / mp.sqrt(x**2 + y**2 + (abs(z) + a) ** 2), a
    )


@case("TriaxialHernquist")
def _thern() -> General:
    pot = gp.TriaxialHernquistPotential(
        m_tot=M, r_s=RS, q1=0.9, q2=0.8, units="galactic"
    )
    Gm, a = _G(pot) * _mp(pot, "m_tot"), _mp(pot, "r_s")
    q1, q2 = _mp(pot, "q1"), _mp(pot, "q2")
    return General(
        pot,
        lambda x, y, z: -Gm / (mp.sqrt(x**2 + (y / q1) ** 2 + (z / q2) ** 2) + a),
        a,
    )


# -------------------------------------------------------------------


def _reference(c: Spherical | General, x: list[mp.mpf]) -> tuple[Any, Any]:
    """Return the gradient and Hessian at ``x``, as float arrays."""
    if isinstance(c, Spherical):
        G = _G(c.pot)
        r = mp.sqrt(sum(v**2 for v in x))
        mass = c.mass(r) if c.mass is not None else _mass_from_rho(c.rho, r, c.scale)
        d1 = G * mass / r**2
        d2 = 4 * mp.pi * G * c.rho(r) - 2 * G * mass / r**3
        n = [v / r for v in x]
        g = [d1 * ni for ni in n]
        H = [
            [d1 / r * (i == j) + (d2 - d1 / r) * n[i] * n[j] for j in range(3)]
            for i in range(3)
        ]
    else:  # extra digits: Lee & Suto's F2, F3 as written cancel ~1/s^2
        with mp.workdps(90):
            g = [
                mp.diff(c.phi, x, tuple(int(i == k) for i in range(3)))
                for k in range(3)
            ]
            H = [
                [
                    mp.diff(
                        c.phi, x, tuple(int(i == k) + int(j == k) for k in range(3))
                    )
                    for j in range(3)
                ]
                for i in range(3)
            ]
    return np.asarray(g, dtype=float), np.asarray(H, dtype=float)


@ft.cache
def _references(name: str) -> tuple[Any, list[tuple[Any, Any, Any]]]:
    """Return the potential and its ``[(q, gradient, Hessian)]`` references."""
    c = CASES[name]()
    refs = []
    for scale in SCALES:
        for d in DIRECTIONS:
            q = np.asarray(d) * float(c.scale) * scale
            refs.append((q, *_reference(c, [mp.mpf(float(v)) for v in q])))
    return c.pot, refs


def _ulp(got: Any, ref: Any) -> float:
    return float(np.max(np.abs(np.asarray(got) - ref)) / (EPS * np.max(np.abs(ref))))


@pytest.mark.parametrize("name", CASES)
def test_gradient_and_hessian_to_a_few_ulp(name: str) -> None:
    """`_gradient`, `_hessian` and both jacobians of `_gradient`, to `TOL` ulp."""
    pot, refs = _references(name)
    t = jnp.asarray(0.0)

    def grad(q: Any) -> Any:
        return pot._gradient(q, t)

    fns = {
        "_gradient": grad,
        "_hessian": lambda q: pot._hessian(q, t),
        "jacfwd(_gradient)": jax.jacfwd(grad),
        "jacrev(_gradient)": jax.jacrev(grad),
    }
    qs = jnp.asarray(np.stack([q for q, _, _ in refs]))
    worst = {}
    for k, f in fns.items():
        out = np.asarray(jax.jit(jax.vmap(f))(qs))
        i_ref = 1 if k == "_gradient" else 2
        worst[k] = max(_ulp(out[i], ref[i_ref]) for i, ref in enumerate(refs))
    tol = TOL_OVERRIDE.get(name, TOL)
    assert all(w <= tol for w in worst.values()), worst


# The potentials whose derivatives come from a `jax.custom_jvp`, with the
# length parameter to differentiate by -- the custom rules' parameter path.
CUSTOM_JVP = {
    "NFW": "r_s",
    "HardCutoffNFW": "r_s",
    "Burkert": "r_s",
    "gNFW0.5": "r_s",
    "Zhao-b": "r_s",
    "PowerLawCutoff": "r_c",
    "StoneOstriker15": "r_h",
    "Vogelsberger08TriaxialNFW": "r_s",
    "LeeSutoTriaxialNFW": "r_s",
    "LongMuraliBar": "a",
}


@pytest.mark.parametrize(("name", "param"), CUSTOM_JVP.items())
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_parameter_derivative_matches_finite_difference(
    name: str, param: str, scale: float
) -> None:
    """`jax.jacfwd` and `jax.jacrev` of `_gradient` by a length parameter."""
    pot = CASES[name]().pot
    p0 = float(getattr(pot, param)(T0).ustrip(KPC))
    q = jnp.asarray(DIRECTIONS[0]) * p0 * scale
    t = jnp.asarray(0.0)

    def grad(p: Any) -> Any:
        new = eqx.tree_at(lambda x: getattr(x, param).value, pot, u.Q(p, KPC))
        return new._gradient(q, t)

    # A wide step: 1000 lengths out the gradient moves by only ~1e-9 of itself
    # per 1e-3 of the parameter, so a narrow one is all rounding. Truncation
    # is then ~(1e-3)^2 relative, hence the tolerance.
    h = 1e-3 * p0
    fd = (grad(p0 + h) - grad(p0 - h)) / (2 * h)
    atol = 1e-5 * float(jnp.max(jnp.abs(fd)))
    assert jnp.allclose(jax.jacfwd(grad)(p0), fd, rtol=1e-5, atol=atol)
    assert jnp.allclose(jax.jacrev(grad)(p0), fd, rtol=1e-5, atol=atol)
