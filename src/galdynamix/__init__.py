"""galdynamix: Galactic Dynamix in Jax"""
from __future__ import annotations

__all__ = ["__version__"]

from jax import config

from ._version import version as __version__

config.update("jax_enable_x64", True)
