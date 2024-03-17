from importlib.metadata import version

import galax as pkg


def test_version() -> None:
    assert version("galax") == pkg.__version__


def test_all() -> None:
    """Test the `galax` package contents."""
    # Test detailed contents (not order)
    assert set(pkg.__all__) == {
        "__version__",
        "__version_tuple__",
        # modules
        "coordinates",
        "dynamics",
        "potential",
        "typing",
        "utils",
    }
