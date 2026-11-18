"""Tests for the gradient policies."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from galax.nbody._src.backend import Sources, Targets
from galax.nbody._src.gradpolicy import Autodiff, NoGrad, StopGrad, as_grad_policy


def fn(s: Sources, t: Targets):
    return jnp.sum(s.m[:, None] * s.q) * t.q


s = Sources(
    q=jnp.array([[1.0, 2, 3], [0, 1, 0]]), m=jnp.array([1.0, 2.0]), h=jnp.zeros(2)
)
t = Targets(q=jnp.array([[1.0, 1, 1]]), h=jnp.zeros(1), self_index=jnp.array([-1]))


def test_autodiff_passes_gradients() -> None:
    g = jax.grad(
        lambda m: Autodiff().apply(fn, eqx.tree_at(lambda x: x.m, s, m), t).sum()
    )(s.m)
    assert jnp.any(g != 0)


def test_stopgrad_m_only_keeps_q_gradient() -> None:
    pol = StopGrad(wrt=frozenset({"m"}))
    gm = jax.grad(lambda m: pol.apply(fn, eqx.tree_at(lambda x: x.m, s, m), t).sum())(
        s.m
    )
    gq = jax.grad(lambda q: pol.apply(fn, eqx.tree_at(lambda x: x.q, s, q), t).sum())(
        s.q
    )
    assert jnp.all(gm == 0)
    assert jnp.any(gq != 0)


def test_stopgrad_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="subset"):
        StopGrad(wrt=frozenset({"x"}))


def test_nograd_forward_runs_and_grad_raises() -> None:
    pol = NoGrad()
    out = jax.jit(lambda s, t: pol.apply(fn, s, t))(s, t)
    assert jnp.allclose(out, fn(s, t))
    with pytest.raises(NotImplementedError, match="NoGrad"):
        jax.grad(lambda m: pol.apply(fn, eqx.tree_at(lambda x: x.m, s, m), t).sum())(
            s.m
        )


def test_as_grad_policy_aliases() -> None:
    assert isinstance(as_grad_policy("autodiff"), Autodiff)
    assert isinstance(as_grad_policy("stop"), StopGrad)
    assert isinstance(as_grad_policy("none"), NoGrad)
    p = Autodiff()
    assert as_grad_policy(p) is p
    with pytest.raises(ValueError, match="autodiff"):
        as_grad_policy("exact")
