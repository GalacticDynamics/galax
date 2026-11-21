"""Tests for the O2GF n-vector lattice."""

import numpy as np
import pytest

from galax.actionangle._src._o2gf_nvecs import BOX_STEPS, LOOP_STEPS, n_vectors


def _gala_generate_n_vectors(N_max, dx=1, dy=1, dz=1, *, half_lattice=True):
    """Copy gala's ``generate_n_vectors`` (MIT license) as the reference.

    Copyright (c) gala developers, https://github.com/adrn/gala
    """
    vecs = np.meshgrid(
        np.arange(-N_max, N_max + 1, dx),
        np.arange(-N_max, N_max + 1, dy),
        np.arange(-N_max, N_max + 1, dz),
    )
    vecs = np.vstack(list(map(np.ravel, vecs))).T
    vecs = vecs[np.linalg.norm(vecs, axis=1) <= N_max]

    if half_lattice:
        ix = (
            (vecs[:, 2] > 0)
            | ((vecs[:, 2] == 0) & (vecs[:, 1] > 0))
            | ((vecs[:, 2] == 0) & (vecs[:, 1] == 0) & (vecs[:, 0] > 0))
        )
        vecs = vecs[ix]

    return np.array(sorted(vecs, key=lambda x: (x[0], x[1], x[2])))


def test_steps_constants():
    assert LOOP_STEPS == (1, 2, 2)
    assert BOX_STEPS == (2, 2, 2)


@pytest.mark.parametrize("n_max", [2, 4, 6, 8])
@pytest.mark.parametrize("steps", [LOOP_STEPS, BOX_STEPS])
def test_matches_gala_even(n_max, steps):
    np.testing.assert_array_equal(
        n_vectors(n_max, steps), _gala_generate_n_vectors(n_max, *steps)
    )


def test_odd_nmax_even_components():
    got = n_vectors(5, BOX_STEPS)
    assert (got % 2 == 0).all()
    for axis in range(3):
        assert (got[:, axis] == 0).any()
    want = n_vectors(6, BOX_STEPS)
    want = want[np.linalg.norm(want, axis=1) <= 5]
    np.testing.assert_array_equal(got, want)


def test_odd_nmax_loop_steps():
    got = n_vectors(5, LOOP_STEPS)
    assert (got[:, 0] % 2 == 1).any()
    assert (got[:, 1:] % 2 == 0).all()
    for axis in (1, 2):
        assert (got[:, axis] == 0).any()
