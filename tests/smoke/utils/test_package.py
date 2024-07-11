from galax import utils


def test__all__() -> None:
    """Test that `galax.utils` has the expected `__all__`."""
    assert utils.__all__ == ["dataclasses", *utils._jax.__all__]
