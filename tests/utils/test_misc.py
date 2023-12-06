from __future__ import annotations

import numpy as np
import pytest

from galdynamix.utils._misc import first


@pytest.mark.parametrize(
    ("x", "x0"),
    [
        ([1, 2, 3], 1),
        ((1, 2, 3), 1),
        ({"a": 1, "b": 2, "c": 3}, "a"),
        (np.array([1, 2, 3]), 1),
        (range(1, 4), 1),
        (iter([1, 2, 3]), 1),
    ],
)
def test_first(x, x0):
    """Test the partial_jit function."""
    assert first(x) == x0
