"""galax: Galactic Dynamix in Jax."""

__all__: tuple[str, ...] = ()

import functools as ft
import math

from collections.abc import Callable
from jaxtyping import Array, Bool, Shaped
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import equinox as eqx
import jax
import numpy as np
import optype as op
from jax.custom_derivatives import SymbolicZero
from jax.dtypes import canonicalize_dtype
from plum import Dispatcher, convert

import coordinax as cx
import coordinax.frames as cxf
import coordinax.vecs as cxv
import quaxed.numpy as jnp
import unxt as u
from spexial import incomplete_beta
from unxt.quantity import AllowValue, Quantity

import galax.coordinates as gc
import galax.potential.custom_types as gt
from .symmetry import Symmetry

if TYPE_CHECKING:
    from .base import AbstractPotential

OptUSys: TypeAlias = u.AbstractUnitSystem | None

speed_of_light = u.quantity.Quantity(299_792_458, "m/s")


def parse_dtypes(dtype2: np.dtype, dtype1: Any, /) -> np.dtype | None:
    return (
        dtype2
        if dtype1 is None
        else jnp.promote_types(dtype2, canonicalize_dtype(dtype1))
    )


# ==============================================================================


@ft.partial(jax.jit, inline=True)
def safe_sqrt(q2: gt.BBtFloatSz0, /) -> gt.BBtFloatSz0:
    """Square root that stays differentiable where its argument vanishes.

    ``sqrt`` has an infinite derivative at 0, so a radius built as
    ``sqrt(x**2 + y**2 + z**2)`` makes `jax.grad` and `jax.hessian` of every
    potential defined in terms of it NaN at the origin. Offsetting by the
    smallest normal float keeps the derivative finite. The offset is ~1e-308
    (float64), so the value is unchanged for any physically meaningful
    position.

    The offset must be applied to the primal, not just the tangent: recovering
    the correct Hessian of a cored profile at the origin needs ``Phi'(r) / r``
    evaluated at a genuinely non-zero ``r``.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from galax.potential._src.utils import safe_sqrt

    The value matches `jnp.sqrt`:

    >>> safe_sqrt(jnp.asarray(4.0))
    Array(2., dtype=float64)

    but the derivative at zero is finite rather than infinite:

    >>> import jax
    >>> jnp.isfinite(jax.grad(safe_sqrt)(jnp.asarray(0.0)))
    Array(True, dtype=bool, weak_type=True)
    >>> jax.grad(jnp.sqrt)(jnp.asarray(0.0))
    Array(inf, dtype=float64...)

    """
    tiny = jnp.finfo(jnp.promote_types(q2.dtype, float)).tiny
    return jnp.sqrt(q2 + tiny)  # type: ignore[no-any-return]


@ft.partial(jax.jit, inline=True)
def safe_vector_norm(x: gt.BBtSz3, /) -> gt.BBtFloatSz0:
    """`jnp.linalg.vector_norm` over the last axis, differentiable at the origin.

    Values match `jnp.linalg.vector_norm` to within a rounding ulp -- the
    offset is a no-op at any meaningful magnitude, but XLA fuses
    ``sum(square(x))`` differently from `vector_norm`'s scaled algorithm. The
    derivative at the origin is finite rather than NaN. See `safe_sqrt`.

    Unlike `vector_norm`, this overflows for ``|x| > ~1e154`` (float64), which
    no physical position reaches.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from galax.potential._src.utils import safe_vector_norm

    >>> xyz = jnp.asarray([3.0, 4.0, 0.0])
    >>> safe_vector_norm(xyz)
    Array(5., dtype=float64)

    At the origin the gradient is zero -- the value `jnp.linalg.vector_norm`
    cannot give -- rather than NaN:

    >>> import jax
    >>> jax.grad(safe_vector_norm)(jnp.zeros(3))
    Array([0., 0., 0.], dtype=float64)
    >>> jax.grad(jnp.linalg.vector_norm)(jnp.zeros(3))
    Array([nan, nan, nan], dtype=float64)

    """
    return safe_sqrt(jnp.sum(jnp.square(x), axis=-1))  # type: ignore[no-any-return]


@ft.partial(jax.jit, inline=True, static_argnames=("unit",))
def r_spherical(xyz: gt.BBtQorVSz3, unit: Any) -> gt.BBtFloatSz0:
    """Spherical radius.

    Uses `safe_vector_norm` so that the gradient and hessian of potentials
    written in terms of ``r`` are finite at the origin rather than NaN.
    """
    xyz = u.ustrip(AllowValue, unit, xyz)
    r = safe_vector_norm(xyz)
    return r  # type: ignore[no-any-return]


# ==============================================================================

RadialFn: TypeAlias = Callable[[gt.Params, gt.BBtSz0], gt.BBtSz0]


def _zero_tangent(x: Any, /) -> Any:
    """Return a zero tangent for ``x``: `float0` for an integer or bool one."""
    x = jnp.asarray(x)
    if jnp.issubdtype(x.dtype, jnp.inexact):
        return jnp.zeros_like(x)
    return np.zeros(x.shape, dtype=jax.dtypes.float0)


def with_r_derivative(
    f: RadialFn,
    df: RadialFn,
    /,
    contract: Callable[[Any, Any], Any] = jnp.multiply,
) -> RadialFn:
    """``f(p, r)`` as a `jax.custom_jvp` whose ``r``-derivative is ``df(p, r)``.

    The tangent is ``contract(df(p, r), r_dot)``: a product for a scalar
    ``r``, a dot product over the last axis when ``r`` is a position and
    ``df`` the gradient. The ``p``-derivative is autodiff of ``f``, skipped
    when ``p`` carries no tangent -- which is every position derivative.
    """
    f_jvp = jax.custom_jvp(f)

    def is_zero(x: Any) -> bool:
        # float0 is the tangent of an integer or bool primal: zero as well.
        return isinstance(x, SymbolicZero) or (
            getattr(x, "dtype", None) == jax.dtypes.float0
        )

    @ft.partial(f_jvp.defjvp, symbolic_zeros=True)
    def _(primals: tuple[Any, Any], tangents: tuple[Any, Any]) -> tuple[Any, Any]:
        (p, r), (p_dot, r_dot) = primals, tangents
        # The wrapped function, not `f`: a higher derivative differentiates
        # this rule, and must again reach `df` rather than autodiff of `f`.
        out = f_jvp(p, r)
        tangent = jnp.zeros_like(out)
        if not all(map(is_zero, jax.tree.leaves(p_dot, is_leaf=is_zero))):
            p_dot = jax.tree.map(
                lambda d, x: _zero_tangent(x) if is_zero(d) else d,
                p_dot,
                p,
                is_leaf=is_zero,
            )
            # Plain autodiff of `f`: an r-derivative taken *after* this p one
            # does not go through `df`. The usual order (r first) does.
            tangent = jax.jvp(lambda q: f(q, r), (p,), (p_dot,))[1]
        if not is_zero(r_dot):
            tangent = tangent + contract(df(p, r), r_dot)
        return out, tangent

    return f_jvp


def shell_d2phi_dr2(mass_enclosed: RadialFn, dmass_dr: RadialFn, /) -> RadialFn:
    r"""$\Phi'' = G M'/r^2 - 2GM/r^3$; no $r_s^3$, $r^3$ (float32 SI overflow)."""

    def d2phi(p: gt.Params, r: gt.BBtSz0) -> gt.BBtSz0:
        G = p["G"]
        return (G * dmass_dr(p, r) / r - 2 * G * mass_enclosed(p, r) / r / r) / r

    return d2phi


def shell_dphi_dr(mass_enclosed: RadialFn, dmass_dr: RadialFn, /) -> RadialFn:
    r"""$\Phi'(r) = G M(r) / r^2$, as a `jax.custom_jvp` with an exact derivative.

    Its ``r``-derivative is $\Phi''(r) = G M'(r)/r^2 - 2 G M(r)/r^3$ rather than
    autodiff of ``mass_enclosed``, so a gradient built on it linearizes to the
    exact Hessian. See `radial_derivatives`, which gives a potential this as
    its own derivative.
    """
    d2phi = shell_d2phi_dr2(mass_enclosed, dmass_dr)
    return with_r_derivative(lambda p, r: p["G"] * mass_enclosed(p, r) / r / r, d2phi)


def radial_derivatives(
    mass_enclosed: RadialFn, dmass_dr: RadialFn, /
) -> Callable[[RadialFn], RadialFn]:
    r"""Give a spherical ``potential(p, r)`` cancellation-free ``r``-derivatives.

    Autodiff differentiates a closed form term by term, and the terms of the
    derivative can cancel where the closed form itself does not: NFW's
    $\ln(1+x)/x$ is accurate, but its autodiff derivative
    $1/(x(1+x)) - \ln(1+x)/x^2$ loses every digit as $x \to 0$. This
    decorator replaces the ``r``-derivatives with the ones Newton's shell
    theorem gives,

    .. math::

        \Phi'(r) = \frac{G M(r)}{r^2}, \qquad
        \Phi''(r) = \frac{G M'(r)}{r^2} - \frac{2 G M(r)}{r^3},

    through two nested `jax.custom_jvp`, so `jax.grad`, `jax.hessian` and
    `jax.linearize` of anything built on ``potential`` are as accurate as
    ``mass_enclosed`` and ``dmass_dr`` (both ``(p, r) -> array``, with
    ``p["G"]`` the gravitational constant). ``dmass_dr`` is $4\pi r^2\rho$,
    taken rather than $\rho$ because it needs no $r_s^3$, which overflows
    float32 in SI units. Each division by $r$ is separate for the same reason.
    Third derivatives are autodiff of $\Phi''$.
    """
    dphi = shell_dphi_dr(mass_enclosed, dmass_dr)

    def decorator(potential: RadialFn) -> RadialFn:
        return ft.wraps(potential)(with_r_derivative(potential, dphi))

    return decorator


_POLE_BAND = 1e-3
"""Half-width of the band around ``b + m == 0`` where `incomplete_beta_zw`
expands; `spexial`'s value, for the same reason (its ``_large_z``)."""


def incomplete_beta_zw(
    a: gt.Sz0, b: gt.Sz0, z: gt.BBtSz0, w: gt.BBtSz0, /
) -> gt.BBtSz0:
    r"""$B(a, b, z)$ from both $z$ and $w = 1 - z$, each to full precision.

    `spexial.incomplete_beta` takes $z$ alone and forms $1 - z$ itself, so for
    $z \to 1$ it gets only the digits of $1 - z$ that $z$ can hold: for NFW's
    $z = x/(1+x)$ at $x = 10^8$, half of them. These are spexial's two series:
    for $z \leq 1/2$ the defining one (its ``_small_z``), and above its
    reflection about $t = 1/2$ (its ``_large_z``), evaluated from $w$:

    .. math::

        B(a, b, z) = B(a, b, 1/2)
            + \sum_{m \geq 0} \frac{(1-a)_m}{m!}
              \frac{(1/2)^{b+m} - w^{b+m}}{b+m},

    with the $b + m \to 0$ term $\ln(1/(2w))$, expanded within `_POLE_BAND`
    of the pole. The terms fall off like $2^{-m}$, so 64 reach $10^{-19}$.
    """
    zc = jnp.where(z <= 0.5, z, 0.5)
    # w = 0 (z = 1) only by underflow, at r ~ 1e154 r_s or beyond: the tiny
    # normal float keeps ln(w) finite, and the result ~ -ln(w) (b = 0) or the
    # complete beta function (b > 0) rather than NaN.
    tiny = jnp.finfo(jnp.result_type(w, float)).tiny
    wc = jnp.where(w < 0.5, jnp.where(w > tiny, w, tiny), 0.5)
    # (L1^k - L2^k) / k! for L1 = ln(1/2), L2 = ln(w), for the near-pole term;
    # 8 of them, where spexial stops at 4: its O(s^4 L^5) truncation is ~1e-8
    # relative at s = 1e-3 for w ~ 1e-16 (L ~ 37), and 8 bring it to ~1e-16.
    l1, l2 = jnp.log(0.5), jnp.log(wc)
    d = [(l1**k - l2**k) / math.factorial(k) for k in range(1, 9)]

    # Both series, unrolled in Python rather than a `jax.lax.scan`: a scan
    # step is a separate pass over the batch, so it is memory-bound and ~7x
    # slower than the one fused loop this compiles to.
    lower = up = pole_coeff = pole_s = jnp.zeros_like(zc)
    z_pow, c_lo = jnp.ones_like(zc), jnp.ones_like(b)  # c_lo = (1-b)_m / m!
    w_pow, half_pow, c_up = wc**b, 0.5**b, jnp.ones_like(a)  # c_up = (1-a)_m / m!
    for m in range(64):
        lower = lower + c_lo / (a + m) * z_pow
        # At most one m is near the pole b + m = 0: its term is set aside here
        # and expanded once below, rather than at every m.
        s = b + m
        near = jnp.abs(s) < _POLE_BAND
        direct = (half_pow - w_pow) / jnp.where(near, 1.0, s)
        up = up + c_up * jnp.where(near, 0.0, direct)
        pole_coeff = pole_coeff + jnp.where(near, c_up, 0.0)
        pole_s = pole_s + jnp.where(near, s, 0.0)
        z_pow, c_lo = z_pow * zc, c_lo * (m + 1.0 - b) / (m + 1.0)
        w_pow, half_pow, c_up = (
            w_pow * wc,
            half_pow * 0.5,
            c_up * (m + 1.0 - a) / (m + 1.0),
        )

    expansion = d[-1]
    for d_k in reversed(d[:-1]):
        expansion = d_k + pole_s * expansion
    b_half = incomplete_beta(a, b, jnp.asarray(0.5, dtype=zc.dtype))
    upper = b_half + up + pole_coeff * expansion
    return jnp.where(z <= 0.5, zc**a * lower, upper)  # type: ignore[no-any-return]


# ==============================================================================


@ft.lru_cache(maxsize=128)
def _nodes_for_dtype(
    order: int, interval: tuple[float, float], dtype: Any, /
) -> tuple[Shaped[Array, "O"], Shaped[Array, "O"]]:
    """Build nodes and weights, cached on the working dtype as well.

    The dtype has to be part of the key. Without it the first caller fixed it
    for every later one: under x64 the entry held float64 arrays, and a
    float32 caller got float64 back -- while the same call at an uncached
    order correctly gave float32. Same function, same config, two dtypes,
    decided by call order. That is what made float32 tests pass alone and
    fail in a file, since JAX then warns about truncation and
    ``filterwarnings = ["error"]`` turns the warning into a failure.

    Keying on it costs one extra `leggauss` per order per dtype, and there
    are only two dtypes.
    """
    x, w = np.polynomial.legendre.leggauss(order)

    # The affine map is not value-neutral, so the identity case is skipped.
    a, b = interval
    if (a, b) != (-1.0, 1.0):
        half = 0.5 * (b - a)
        x, w = half * (x + 1) + a, half * w

    # These are constants, so they must not capture whatever jit trace
    # happens to call for them first.
    with jax.ensure_compile_time_eval():
        return jnp.asarray(x, dtype=dtype), jnp.asarray(w, dtype=dtype)


def gauss_legendre_nodes(
    order: int, interval: tuple[float, float] = (0.0, 1.0), /
) -> tuple[Shaped[Array, "O"], Shaped[Array, "O"]]:
    """Gauss-Legendre nodes & weights, mapped onto ``interval``.

    The nodes and weights are constants, so they are cached: computing them
    (`numpy.polynomial.legendre.leggauss`) costs ~0.6 ms at order 50, which is
    a significant fraction of the construction time of a potential that
    integrates.

    Examples
    --------
    >>> from galax.potential._src.utils import gauss_legendre_nodes

    >>> x, w = gauss_legendre_nodes(3, (-1.0, 1.0))
    >>> x
    Array([-0.77459667,  0.        ,  0.77459667], dtype=float64)
    >>> w
    Array([0.55555556, 0.88888889, 0.55555556], dtype=float64)

    The reference interval is exact: the affine map is skipped entirely, since
    e.g. ``(x + 1) - 1`` perturbs every node by an ulp.

    >>> import numpy as np
    >>> np.array_equal(x, np.polynomial.legendre.leggauss(3)[0])
    True

    """
    return _nodes_for_dtype(order, interval, canonicalize_dtype(float))


def gauss_legendre(
    f: Callable[
        [Shaped[Array, "N"]],
        Shaped[Array, "N *batch"] | Shaped[u.AbstractQuantity, "N *batch"],
    ],
    order: int,
    /,
    *,
    interval: tuple[float, float] = (0.0, 1.0),
) -> Shaped[Array, "*batch"] | Shaped[u.AbstractQuantity, "*batch"]:
    """Integrate ``f`` over ``interval`` by Gauss-Legendre quadrature.

    ``f`` maps the ``(N,)`` nodes to ``(N, *batch)``; the sum is over the node
    axis, giving ``(*batch,)``.

    This is deliberately not `jax.jit`-ed: ``f`` would have to be static, and a
    freshly-built closure -- which is what every caller passes -- then misses
    the cache on every call. Callers jit their own enclosing function instead.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from galax.potential._src.utils import gauss_legendre

    >>> gauss_legendre(lambda x: x**2, 10).round(12)
    Array(0.33333333, dtype=float64)

    >>> gauss_legendre(jnp.sin, 10, interval=(0.0, jnp.pi)).round(12)
    Array(2., dtype=float64)

    """
    x, w = gauss_legendre_nodes(order, (float(interval[0]), float(interval[1])))
    y = f(x)
    w = w.reshape(w.shape + (1,) * (y.ndim - 1))
    return jnp.sum(y * w, axis=0)


# ==============================================================================

coord_dispatcher = Dispatcher(warn_redefinition=True)


@coord_dispatcher.abstract
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None, *args: Any, **kwargs: Any
) -> tuple[Any, Any]:  # pos, time
    """Parse input arguments to position & time.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc
    >>> from galax.potential._src.utils import parse_to_xyz_t

    - `jax.Array`-like:

    >>> xyz = [1, 0, 0]
    >>> t = 0
    >>> parse_to_xyz_t(None, xyz, t, dtype=float)
    (Array([1., 0., 0.], dtype=float64),
     Array(0., dtype=float64))

    ``t=None`` is the default time, ``t=0``:

    >>> parse_to_xyz_t(None, xyz, None, dtype=float)
    (Array([1., 0., 0.], dtype=float64),
     Array(0., dtype=float64))

    - `jax.Array`:

    >>> xyz = jnp.array([1, 0, 0])
    >>> t = jnp.array(0)
    >>> parse_to_xyz_t(None, xyz, t, dtype=float)
    (Array([1., 0., 0.], dtype=float64),
     Array(0., dtype=float64))

    >>> txyz = jnp.array([0, 1, 0, 0])
    >>> parse_to_xyz_t(None, txyz, dtype=float)
    (Array([1., 0., 0.], dtype=float64),
     Array(0., dtype=float64))

    >>> parse_to_xyz_t(None, txyz, t, dtype=float)
    (Array([1., 0., 0.], dtype=float64),
     Array(0., dtype=float64))

    - `unxt.Quantity`:

    >>> q = u.Q([1, 0, 0], "kpc")
    >>> t = u.Q(1, "Gyr")
    >>> parse_to_xyz_t(None, q, t)
    (Q([1, 0, 0], 'kpc'), Q(1, 'Gyr'))

    >>> parse_to_xyz_t(None, q, t, ustrip=u.unitsystems.galactic)
    (Array([1, 0, 0], dtype=int64),
     Array(1000., dtype=float64, weak_type=True))

    >>> parse_to_xyz_t(None, q, None)
    (Q([1, 0, 0], 'kpc'), Q(0., 'Myr'))

    >>> tq = u.Q([0, 1, 0, 0], "kpc")
    >>> parse_to_xyz_t(None, tq)
    (Q([1, 0, 0], 'kpc'), Q(0., 'kpc s / m'))

    >>> parse_to_xyz_t(None, tq, u.Q(0, "Gyr"))
    (Q([1, 0, 0], 'kpc'), Q(0., 'kpc s / m'))

    - `coordinax.AbstractVector` objects:

    >>> q = cx.vecs.CartesianPos3D.from_([1, 0, 0], "kpc")
    >>> parse_to_xyz_t(None, q, t)
    (Q([1, 0, 0], 'kpc'), Q(1, 'Gyr'))

    >>> parse_to_xyz_t(None, q, t, ustrip=u.unitsystems.galactic)
    (Array([1, 0, 0], dtype=int64),
     Array(1000., dtype=float64, weak_type=True))

    >>> tq = cx.vecs.FourVector(q=q, t=t)
    >>> parse_to_xyz_t(None, tq)
    (Q([1, 0, 0], 'kpc'), Q(1, 'Gyr'))

    >>> parse_to_xyz_t(None, tq, t)
    (Q([1, 0, 0], 'kpc'), Q(1, 'Gyr'))

    - `coordinax.KinematicSpace` objects:

    >>> space = cx.KinematicSpace(length=q)
    >>> parse_to_xyz_t(None, space, t)
    (Q([1, 0, 0], 'kpc'), Q(1, 'Gyr'))

    >>> parse_to_xyz_t(None, space, t, ustrip=u.unitsystems.galactic)
    (Array([1, 0, 0], dtype=int64),
     Array(1000., dtype=float64, weak_type=True))

    >>> space = cx.KinematicSpace(length=tq)
    >>> parse_to_xyz_t(None, space)
    (Q([1, 0, 0], 'kpc'), Q(1, 'Gyr'))

    >>> parse_to_xyz_t(None, space, t)
    (Q([1, 0, 0], 'kpc'), Q(1, 'Gyr'))

    - `coordinax.AbstractCoordinate` objects:

    >>> coord = cx.Coordinate(cx.KinematicSpace(length=q),
    ...                       frame=gc.frames.simulation_frame)
    >>> parse_to_xyz_t(None, coord, t)
    (Q([1, 0, 0], 'kpc'), Q(1, 'Gyr'))

    >>> parse_to_xyz_t(None, coord, t, ustrip=u.unitsystems.galactic)
    (Array([1, 0, 0], dtype=int64),
     Array(1000., dtype=float64, weak_type=True))

    >>> coord = cx.Coordinate(cx.KinematicSpace(length=tq),
    ...                       frame=gc.frames.simulation_frame)
    >>> parse_to_xyz_t(None, coord)
    (Q([1, 0, 0], 'kpc'), Q(1, 'Gyr'))

    >>> parse_to_xyz_t(None, coord, t)
    (Q([1, 0, 0], 'kpc'), Q(1, 'Gyr'))

    - `galax.coordinates.PhaseSpacePosition` objects:

    >>> p = cx.vecs.CartesianVel3D.from_([0, 0, 0], "km/s")
    >>> w = gc.PhaseSpacePosition(q=q, p=p)
    >>> parse_to_xyz_t(None, w, t)
    (Q([1, 0, 0], 'kpc'), Q(1, 'Gyr'))

    >>> parse_to_xyz_t(None, w, t, ustrip=u.unitsystems.galactic)
    (Array([1, 0, 0], dtype=int64),
     Array(1000., dtype=float64, weak_type=True))

    - `galax.coordinates.PhaseSpaceCoordinate` objects:

    >>> wt = gc.PhaseSpaceCoordinate(q=q, p=p, t=t)

    >>> parse_to_xyz_t(None, wt)
    (Q([1, 0, 0], 'kpc'), Q(1, 'Gyr'))

    >>> parse_to_xyz_t(None, wt, ustrip=u.unitsystems.galactic)
    (Array([1, 0, 0], dtype=int64),
     Array(1000., dtype=float64, weak_type=True))

    >>> parse_to_xyz_t(None, wt, t)
    (Q([1, 0, 0], 'kpc'), Q(1, 'Gyr'))

    """


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    xyz: gt.XYZArrayLike,
    t: gt.BBtLikeSz0 | None,  # TODO: consider also "*#batch 1"
    /,
    *,
    dtype: Any = None,
    ustrip: OptUSys = None,  # noqa: ARG001
) -> tuple[gt.BBtSz3, gt.BBtSz0]:
    """Parse input arguments to position & time.

    ``t=None`` means the default time, ``t=0``.
    """
    # Process the input arguments into arrays
    xyz = jnp.asarray(xyz, dtype=dtype)
    t = jnp.asarray(0.0 if t is None else t, dtype=dtype)

    # The coordinates are assumed to be in the simulation frame and may need to
    # be transformed to the target frame.
    if to_frame is not None:
        op = cxf.frame_transform_op(gc.frames.simulation_frame, to_frame)
        xyz, t = op(xyz, t)

    return xyz, t


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    txyz: gt.BBtLikeSz4,  # Cartesian, in the reference frame
    /,
    *,
    dtype: Any = None,
    ustrip: OptUSys = None,
) -> tuple[gt.BBtSz3, gt.BBtSz0]:
    """Parse input argument to position & time."""
    txyz = jnp.asarray(txyz, dtype=dtype)
    return parse_to_xyz_t(
        to_frame, txyz[..., 1:4], txyz[..., 0], dtype=None, ustrip=ustrip
    )


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    txyz: gt.BBtLikeSz4,  # Cartesian, in the reference frame
    t_ref: gt.BBtLikeSz0 | None,
    /,
    *,
    dtype: Any = None,
    ustrip: OptUSys = None,
) -> tuple[gt.BBtSz3, gt.BBtSz0]:
    """Parse input argument to position & time."""
    txyz = jnp.asarray(txyz, dtype=dtype)
    t, xyz = txyz[..., 0], txyz[..., 1:4]
    t = eqx.error_if(
        t,
        t_ref is not None
        and u.ustrip(AllowValue, "", jnp.logical_not(jnp.array_equal(t_ref, t))),
        "t != txyz[..., 0], None",
    )
    return parse_to_xyz_t(to_frame, xyz, t, dtype=dtype, ustrip=ustrip)


@coord_dispatcher.multi(
    (cxf.AbstractReferenceFrame | None, gt.BBtQuSz3, gt.BBtQuSz0),
    (cxf.AbstractReferenceFrame | None, gt.BBtQuSz3, gt.BBtSz0 | float | int | None),
)
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    xyz: gt.BBtQorVSz3,
    t: gt.BBtQorVSz0 | float | int | None,
    /,
    *,
    dtype: Any = None,
    ustrip: OptUSys = None,
) -> tuple[gt.BBtQorVSz3, gt.BBtQorVSz0]:
    """Parse input arguments to position & time.

    ``t=None`` means the default time, ``t=0``.
    """
    xyz = jnp.asarray(xyz, dtype=dtype)
    # A Quantity so that a Quantity position keeps a Quantity time. The unit is
    # immaterial for zero.
    t = jnp.asarray(u.Q(0.0, "Myr") if t is None else t, dtype=dtype)

    if ustrip is not None:
        xyz = u.ustrip(AllowValue, ustrip["length"], xyz)
        t = u.ustrip(AllowValue, ustrip["time"], t)

    # The coordinates are assumed to be in the simulation frame and may need to
    # be transformed to the target frame.
    if to_frame is not None:
        op = cxf.frame_transform_op(gc.frames.simulation_frame, to_frame)
        xyz, t = op(xyz, t)

    return xyz, t


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    txyz: gt.BBtQuSz4,
    /,
    *,
    dtype: Any = None,
    ustrip: OptUSys = None,
) -> tuple[gt.BBtQorVSz3, gt.BBtQorVSz0]:
    """Parse input arguments to position & time."""
    ct, xyz = txyz[..., 0], txyz[..., 1:4]
    t = ct / speed_of_light
    return parse_to_xyz_t(to_frame, xyz, t, dtype=dtype, ustrip=ustrip)


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    txyz: gt.BBtQuSz4,
    tref: gt.BBtQuSz0 | None,
    /,
    *,
    dtype: Any = None,
    ustrip: OptUSys = None,
) -> tuple[gt.BBtQorVSz3, gt.BBtQorVSz0]:
    """Parse input arguments to position & time."""
    ct, xyz = txyz[..., 0], txyz[..., 1:4]
    t = ct / speed_of_light
    t = eqx.error_if(
        t,
        tref is not None
        and u.ustrip(AllowValue, "", jnp.logical_not(jnp.array_equal(tref, t))),
        "t != txyz[..., 0], None",
    )
    return parse_to_xyz_t(to_frame, xyz, t, dtype=dtype, ustrip=ustrip)


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    q: cx.vecs.AbstractPos3D,
    t: Any,
    /,
    *,
    dtype: Any = None,
    ustrip: OptUSys = None,
) -> tuple[gt.BBtQorVSz3, gt.BBtQorVSz0]:
    """Parse input arguments to position & time."""
    xyz = convert(q.vconvert(cx.CartesianPos3D), Quantity)
    return parse_to_xyz_t(to_frame, xyz, t, dtype=dtype, ustrip=ustrip)


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    q4: cx.vecs.FourVector,
    /,
    *,
    dtype: Any = None,
    ustrip: OptUSys = None,
) -> tuple[gt.BBtQuSz3, gt.BBtQuSz0]:
    """Parse input arguments to position & time."""
    return parse_to_xyz_t(to_frame, q4.q, q4.t, dtype=dtype, ustrip=ustrip)


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    q4: cx.vecs.FourVector,
    tref: gt.BBtQuSz0 | None,
    /,
    *,
    dtype: Any = None,
    ustrip: OptUSys = None,
) -> tuple[gt.BBtQuSz3, gt.BBtQuSz0]:
    """Parse input arguments to position & time."""
    t = q4.t
    t = eqx.error_if(
        t,
        tref is not None
        and u.ustrip(AllowValue, "", jnp.logical_not(jnp.array_equal(tref, t))),
        "t != q4.t, None",
    )
    return parse_to_xyz_t(to_frame, q4.q, t, dtype=dtype, ustrip=ustrip)


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    space: cxv.KinematicSpace,
    /,
    *,
    dtype: Any = None,
    ustrip: OptUSys = None,
) -> tuple[gt.BBtQorVSz3, gt.BBtQorVSz0]:
    """Parse input arguments to position & time."""
    q = space["length"]
    q = eqx.error_if(q, not isinstance(q, cx.vecs.FourVector), "q is not a FourVector")
    return parse_to_xyz_t(to_frame, q, dtype=dtype, ustrip=ustrip)


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    space: cxv.KinematicSpace,
    t: Any,
    /,
    *,
    dtype: Any = None,
    ustrip: OptUSys = None,
) -> tuple[gt.BBtQorVSz3, gt.BBtQorVSz0]:
    """Parse input arguments to position & time."""
    q = space["length"]

    # Case 1: 3D position, `t=None` is the default time
    if isinstance(q, cx.vecs.AbstractPos3D):
        return parse_to_xyz_t(to_frame, q, t, dtype=dtype, ustrip=ustrip)

    # Case 2: 4D position, time must be equal or None
    if isinstance(q, cx.vecs.FourVector):
        return parse_to_xyz_t(to_frame, q, t, dtype=dtype, ustrip=ustrip)

    msg = f"Unsupported position type: {type(q)}"
    raise TypeError(msg)


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    coord: cxf.AbstractCoordinate,
    /,
    *,
    dtype: Any = None,
    ustrip: OptUSys = None,
) -> tuple[gt.BBtQorVSz3, gt.BBtQorVSz0]:
    """Parse input arguments to position & time."""
    # Transform to the frame
    # TODO: think about the transformation of the time
    coord = coord.to_frame(gc.frames.simulation_frame if to_frame is None else to_frame)
    # Re-dispatch on the data
    # Now that the data is in the correct frame, we can just parse the data.
    return parse_to_xyz_t(None, coord.data, dtype=dtype, ustrip=ustrip)


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    coord: cxf.AbstractCoordinate,
    t: Any,
    /,
    *,
    dtype: Any = None,
    ustrip: OptUSys = None,
) -> tuple[gt.BBtQorVSz3, gt.BBtQorVSz0]:
    """Parse input arguments to position & time."""
    # Transform to the frame
    # TODO: think about the transformation of the time
    coord = coord.to_frame(gc.frames.simulation_frame if to_frame is None else to_frame)
    # Re-dispatch on the data
    # Now that the data is in the correct frame, we can just parse the data.
    return parse_to_xyz_t(None, coord.data, t, dtype=dtype, ustrip=ustrip)


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    w: gc.PhaseSpacePosition,
    t: Any,
    /,
    *,
    dtype: Any = None,
    ustrip: OptUSys = None,
) -> tuple[gt.BBtQuSz3, gt.BBtQuSz0]:
    """Parse input arguments to position & time."""
    # Transform to the frame
    # TODO: think about the transformation of the time
    w = w.to_frame(gc.frames.simulation_frame if to_frame is None else to_frame)
    # Re-dispatch on the data
    # Now that the data is in the correct frame, we can just parse the data.
    return parse_to_xyz_t(None, w.q, t, dtype=dtype, ustrip=ustrip)


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    wt: gc.AbstractPhaseSpaceCoordinate,
    t: Any,
    /,
    *,
    dtype: Any = None,
    ustrip: OptUSys = None,
) -> tuple[gt.BBtQuSz3, gt.BBtQuSz0]:
    """Parse input arguments to position & time."""
    # Transform to the frame
    wt = wt.to_frame(gc.frames.simulation_frame if to_frame is None else to_frame)
    # Parse `t`
    t = eqx.error_if(
        jnp.asarray(wt.t, dtype=dtype),
        t is not None
        and u.ustrip(AllowValue, "", jnp.logical_not(jnp.array_equal(wt.t, t))),
        "t != wt.t, None",
    )
    # Re-dispatch on the data
    # Now that the data is in the correct frame, we can just parse the data.
    return parse_to_xyz_t(None, wt.q, t, dtype=dtype, ustrip=ustrip)


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cxf.AbstractReferenceFrame | None,
    wt: gc.AbstractPhaseSpaceCoordinate,
    /,
    *,
    dtype: Any = None,
    ustrip: OptUSys = None,
) -> tuple[gt.BBtQuSz3, gt.BBtQuSz0]:
    """Parse input arguments to position & time."""
    # Transform to the frame
    wt = wt.to_frame(gc.frames.simulation_frame if to_frame is None else to_frame)
    # Re-dispatch on the data
    # Now that the data is in the correct frame, we can just parse the data.
    return parse_to_xyz_t(
        None, wt.q, jnp.asarray(wt.t, dtype=dtype), dtype=dtype, ustrip=ustrip
    )


def parse_pot_to_xyz_t(
    pot: "AbstractPotential", q: Any, /, *args: Any, **kwargs: Any
) -> tuple[Any, Any]:
    """`parse_to_xyz_t`, with the potential supplying the parsing context.

    Most inputs mean the same thing for every potential and are passed
    straight through. A `coordinax.vecs.RadialPos` is the exception: a radius
    only determines a position if the potential is spherically symmetric, so
    it is resolved here, where `AbstractPotential.symmetry` is known.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp
    >>> from galax.potential._src.utils import parse_pot_to_xyz_t

    >>> r = cx.vecs.RadialPos(r=u.Q(8.0, "kpc"))
    >>> t = u.Q(0, "Gyr")

    >>> pot = gp.HernquistPotential(m_tot=u.Q(1e12, "Msun"), r_s=u.Q(5, "kpc"),
    ...                             units="galactic")
    >>> parse_pot_to_xyz_t(pot, r, t)
    (Q([8., 0., 0.], 'kpc'), Q(0, 'Gyr'))

    A radius is ambiguous for anything else:

    >>> pot = gp.MiyamotoNagaiPotential(m_tot=u.Q(1e12, "Msun"), a=u.Q(5, "kpc"),
    ...                                 b=u.Q(1, "kpc"), units="galactic")
    >>> try:
    ...     parse_pot_to_xyz_t(pot, r, t)
    ... except TypeError as e:
    ...     print(e)
    MiyamotoNagaiPotential declares symmetry 'zrotation_zreflection'; a
    RadialPos needs 'spherical'. Pass a 3D position instead, e.g.
    coordinax.vecs.SphericalPos.

    """
    if isinstance(q, cxv.RadialPos):
        # Normalize before comparing. `symmetry` is usually a plain class
        # attribute, so nothing converts a hand-written string the way a
        # `ParameterField` converter would. Left unvalidated, a typo compares
        # unequal and the caller is told their potential is *ambiguous* --
        # which sends them looking for a direction to pass, when the real
        # fault is the declaration. `Symmetry` names the valid values.
        symmetry = Symmetry(pot.symmetry)
        if symmetry != Symmetry.SPHERICAL:
            msg = (
                f"{type(pot).__name__} declares symmetry '{symmetry}'; a "
                f"RadialPos needs '{Symmetry.SPHERICAL}'. Pass a 3D position "
                "instead, e.g. coordinax.vecs.SphericalPos."
            )
            raise TypeError(msg)
        q = q.vconvert(cx.CartesianPos3D)

    # TODO: frame
    return parse_to_xyz_t(None, q, *args, **kwargs)  # type: ignore[no-any-return]


# ============================================================================
# Moved here from `galax.dynamics._src.utils`: `potential` is the lower
# subpackage of the two that use it, so keeping it in `dynamics` meant
# `potential` importing upward.


def _identity[T](x: T) -> T:
    return x


def _reverse[T](x: op.CanGetitem[Any, T]) -> T:
    return x[::-1]


def cond_reverse[T](pred: Bool[Array, ""], x: T) -> T:
    """Reverse `x` if `pred` is True."""
    # A comparison of two quantities is itself a dimensionless quantity under
    # unxt 2, and `jax.lax.cond` only takes a bare array.
    pred = u.ustrip(AllowValue, "", pred)
    return cast("T", jax.lax.cond(pred, _reverse, _identity, x))
