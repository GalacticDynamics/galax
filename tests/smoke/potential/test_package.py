import galax.potential as gp
from galax.potential._potential import base, builtin, composite, core, param, special


def test_all() -> None:
    """Test the `galax.potential` package contents."""
    # Test detailed contents (not order)
    assert set(gp.__all__) == {
        "io",
        *base.__all__,
        *builtin.__all__,
        *composite.__all__,
        *core.__all__,
        *param.__all__,
        *special.__all__,
    }
