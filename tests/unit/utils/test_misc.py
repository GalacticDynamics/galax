"""Test the :mod:`galax.utils._misc` module."""

from collections.abc import Iterable
from typing import TypeVar

import numpy as np
import pytest

from galax.utils._misc import zeroth

T = TypeVar("T")


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
def test_zeroth(x: Iterable[T], x0: T) -> None:
    """Test the function :func:`first`."""
    assert zeroth(x) == x0
