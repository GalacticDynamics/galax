import galax.potential as gp
from galax.potential import _potential


def test_all() -> None:
    """Test the `galax.potential` package contents."""
    # Test detailed contents (not order)
    assert set(gp.__all__) == {
        "io",
        *_potential.base.__all__,
        *_potential.builtin.__all__,
        *_potential.composite.__all__,
        *_potential.core.__all__,
        *_potential.param.__all__,
        *_potential.special.__all__,
    }
