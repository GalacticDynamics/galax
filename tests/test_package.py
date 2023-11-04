from __future__ import annotations

import importlib.metadata

import galdynamix as m


def test_version():
    assert importlib.metadata.version("galdynamix") == m.__version__
