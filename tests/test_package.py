from importlib.metadata import version

import galdynamix as pkg


def test_version():
    assert version("galdynamix") == pkg.__version__
