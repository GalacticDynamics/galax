import galax.potential as gp
from galax.potential._src import (
    api,
    base,
    base_multi,
    base_single,
    builtin,
    composite,
    frame,
    params,
)


def test_all() -> None:
    """Test the `galax.potential` package contents."""
    # Test detailed contents (not order)
    assert set(gp.__all__) == {
        "io",
        "params",
        "plot",
        *base.__all__,
        *base_single.__all__,
        *base_multi.__all__,
        *builtin.__all__,
        *composite.__all__,
        *params.__all__,
        *frame.__all__,
        *api.__all__,
    }
