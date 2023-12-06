import pytest

from galdynamix import utils


def test__all__():
    """Test that `galdynamix.utils` has the expected `__all__`."""
    assert utils.__all__ == [
        "dataclasses",
        *utils._jax.__all__,
        *utils._collections.__all__,
    ]


@pytest.mark.skip(reason="TODO")
def test_public_modules():
    """Test which modules are publicly importable."""
    # IDK how to discover all submodules of a package, even if they aren't
    # imported without relying on the filesystem. The filesystem is generally
    # safe, but I'd rather solve this generically. Low priority.
