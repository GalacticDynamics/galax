from importlib.metadata import version

import galax as pkg


def test_version():
    assert version("galax") == pkg.__version__
