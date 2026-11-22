"""`L1L2LagrangePoints` must be a type hint beartype can check.

`galax.dynamics` is imported under jaxtyping's hook with beartype in
`tests/smoke/test_runtime_typechecking.py`; this is the fast check of the hint
that used to stop that import.
"""

from beartype import beartype

from galax.dynamics._src.cluster.api import L1L2LagrangePoints


def test_lagrange_points_is_a_beartype_checkable_return_hint() -> None:
    """Beartype rejects a generic `NamedTuple`, which `L1L2LagrangePoints` was."""

    @beartype
    def points() -> L1L2LagrangePoints:
        return L1L2LagrangePoints(l1=1.0, l2=2.0)

    assert points() == (1.0, 2.0)
