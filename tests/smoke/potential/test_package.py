import galax.potential as gp
from galax.potential import _potential


def test_all():
    """Test the `galax.potential` package contents."""
    # Test correct dumping of contents
    assert gp.__all__ == _potential.__all__

    # Test detailed contents (not order)
    assert set(gp.__all__) == set(
        _potential.base.__all__
        + _potential.builtin.__all__
        + _potential.composite.__all__
        + _potential.core.__all__
        + _potential.param.__all__
        + _potential.special.__all__
    )
