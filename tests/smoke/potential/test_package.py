import galax.potential as gp
from galax.potential._src import (
    base,
    builtin,
    composite,
    core,
    frame,
    funcs,
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
        *builtin.__all__,
        *composite.__all__,
        *core.__all__,
        *params.__all__,
        *frame.__all__,
        *funcs.__all__,
    }
