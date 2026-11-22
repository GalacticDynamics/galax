"""Integer n-vector lattice for O2GF (plain NumPy, called at trace time)."""

__all__ = ["BOX_STEPS", "LOOP_STEPS", "n_vectors"]

from typing import Final

import numpy as np

LOOP_STEPS: Final = (1, 2, 2)
BOX_STEPS: Final = (2, 2, 2)


def n_vectors(n_max: int, steps: tuple[int, int, int]) -> np.ndarray:
    """Half lattice of integer vectors with ``|n| <= n_max``, sorted lexicographically.

    On an axis with step 1 the components are all integers in ``[-n_max, n_max]``.
    On an axis with step 2 they are the even integers in ``[-m, m]``, where
    ``m`` is ``n_max`` rounded down to an even number.  The half lattice keeps
    ``k > 0``, or ``k = 0 and j > 0``, or ``k = j = 0 and i > 0``.

    At odd ``n_max`` this differs from gala, whose step-2 axes then use the odd
    integers (so they contain no zero); here they keep the even integers.
    """
    axes = [np.arange(-(n_max - n_max % s), n_max + 1, s) for s in steps]
    vecs = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    i, j, k = vecs.T
    keep = ((vecs**2).sum(axis=1) <= n_max**2) & (
        (k > 0) | ((k == 0) & (j > 0)) | ((k == 0) & (j == 0) & (i > 0))
    )
    vecs = vecs[keep]
    return vecs[np.lexsort((vecs[:, 2], vecs[:, 1], vecs[:, 0]))]
