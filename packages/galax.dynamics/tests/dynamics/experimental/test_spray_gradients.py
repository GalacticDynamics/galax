"""Gradients of `SpraySimulator` in every mode, DF and progenitor model."""

import diffrax as dfx
import jax
import jax.numpy as jnp
import jaxmore.loop_strategies as lstrat
import numpy as np
import pytest

import diffraxtra as dfxtra

import galax.dynamics as gd
import galax.fields as gf
import galax.potential as gp
from .test_spray_whole_stream import DFS, KEY, LEAP, MSAT, USYS, W_TODAY, plummer

gde = gd.experimental
FWD = dfxtra.DiffEqSolver(
    gf.Leapfrog(), stepsize_controller=dfx.ConstantStepSize(), adjoint=dfx.ForwardMode()
)
MODES = {
    "whole": lambda _: gde.WholeStream(),
    "copies": lambda loop: gde.PerParticle(gde.Copies(), loop),
    "track": lambda loop: gde.PerParticle(gde.SharedTrack(), loop),
}
MODE_LOOPS = [("whole", None)] + [
    (m, loop) for m in ("copies", "track") for loop in (lstrat.VMap, lstrat.Scan)
]


def _case(mode, loop, df_cls, prog):
    """Return a matrix entry: CI runs WholeStream fully, the rest on Chen+Plummer."""
    fast = mode == "whole" or (df_cls is gde.Chen2024DF and prog is not None)
    parts = (mode, getattr(loop, "__name__", None), df_cls.__name__)
    name = "-".join(p for p in parts if p) + ("-plummer" if prog else "-no-prog")
    return pytest.param(
        mode,
        loop,
        df_cls,
        prog,
        id=name,
        marks=() if fast else pytest.mark.slow,
    )


CASES = [
    _case(m, lp, df, prog)
    for m, lp in MODE_LOOPS
    for df in DFS
    for prog in (None, plummer)
]
X = (jnp.asarray(1.0), W_TODAY, jnp.asarray(500.0), jnp.asarray(0.0))
EPS = jnp.concat([jnp.asarray([1e-6]), jnp.full(6, 1e-6), jnp.asarray([1e-4, 1e-4])])


def _flat(gs):
    return jnp.concat([jnp.ravel(a) for a in gs])


@pytest.mark.filterwarnings("ignore:Chen2024DF without a progenitor:RuntimeWarning")
@pytest.mark.parametrize(("mode", "loop", "df_cls", "prog"), CASES)
def test_gradient_matrix(mode, loop, df_cls, prog) -> None:
    """Reverse mode vs central differences and vs ``jvp`` with ForwardMode.

    20 particles over 500 Myr, Leapfrog with 50 steps per release; the inputs
    are the host's mass scale, the progenitor's state, ``t_age`` and ``t1``.
    """

    def make(solver):
        sim = gde.SpraySimulator(
            df_cls(), MODES[mode](loop), progenitor_potential=prog, solver=solver
        )

        def f(s, w, t_age, t1):
            pot = gp.NFWPotential(m=1e12 * s, r_s=15.0, units=USYS)
            r = sim(KEY, pot, w, MSAT, gde.Uniform(t_age, 10, 50, t1))
            return jnp.sum(r.lead[:, :3]) + jnp.sum(r.trail[:, :3])

        return f

    f = make(LEAP)
    g = _flat(jax.jit(jax.grad(f, argnums=(0, 1, 2, 3)))(*X))
    jf = _flat(jax.jit(jax.jacfwd(make(FWD), argnums=(0, 1, 2, 3)))(*X))
    np.testing.assert_allclose(g, jf, rtol=1e-10, atol=1e-10 * jnp.abs(g).max())

    flat = _flat(X)

    def f_flat(z):
        return f(z[0], z[1:7], z[7], z[8])

    fd = jnp.asarray(
        [
            (f_flat(flat.at[i].add(EPS[i])) - f_flat(flat.at[i].add(-EPS[i])))
            / (2 * EPS[i])
            for i in range(flat.size)
        ]
    )
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7 * jnp.abs(g).max())
