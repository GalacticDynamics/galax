"""Indexing a `MockStreamArm` must index its times the same way."""

import pytest

import coordinax.frames as cxf
import quaxed.numpy as jnp
import unxt as u

import galax.dynamics as gd


def make_arm(n: int) -> gd.MockStreamArm:
    """Return an arm of `n` particles with distinguishable times."""
    return gd.MockStreamArm(
        q=u.Q([[1.0, 2, 3]] * n, "kpc"),
        p=u.Q([[1.0, 2, 3]] * n, "km/s"),
        t=u.Q([float(i) for i in range(n)], "Myr"),
        release_time=u.Q([float(i) for i in range(n)], "Myr"),
        frame=cxf.NoFrame(),
    )


@pytest.mark.parametrize("n", [1, 2, 3])
def test_array_index_keeps_times_aligned(n: int) -> None:
    """`q`, `t` and `release_time` must come back the same length.

    The regression: the time index was a literal `[True]`, so `t` was neither
    filtered by the caller's mask nor sized to the arm. For `n == 1` that
    silently kept a dropped particle's time; for longer arms it raised
    `IndexError`.
    """
    arm = make_arm(n)
    keep = jnp.asarray([True] + [False] * (n - 1))

    sub = arm[keep]

    assert sub.q.shape == sub.t.shape == sub.release_time.shape == (1,)


def test_array_index_selects_the_right_times() -> None:
    """Not just the right length -- the right elements."""
    arm = make_arm(3)

    sub = arm[jnp.asarray([False, True, True])]

    assert jnp.array_equal(sub.t, u.Q([1.0, 2.0], "Myr"))
    assert jnp.array_equal(sub.release_time, u.Q([1.0, 2.0], "Myr"))


def test_dropping_every_particle_drops_every_time() -> None:
    """The single-particle case that used to desync in silence."""
    arm = make_arm(1)

    sub = arm[jnp.asarray([False])]

    assert sub.q.shape == (0,)
    assert sub.t.shape == (0,)
    assert sub.release_time.shape == (0,)


@pytest.mark.parametrize(
    ("index", "expected"), [(1, ()), (slice(0, 2), (2,)), ((slice(None),), (3,))]
)
def test_ordinary_indices_still_work(index, expected) -> None:
    """Integer, slice and tuple indexing are unaffected."""
    sub = make_arm(3)[index]
    assert sub.q.shape == sub.t.shape == expected
