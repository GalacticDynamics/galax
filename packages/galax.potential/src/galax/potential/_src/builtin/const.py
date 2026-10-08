"""Constants. Private module."""

__all__ = ["SQRT2"]

from typing import Final

import quaxed.numpy as jnp

SQRT2: Final = jnp.sqrt(jnp.asarray(2.0))
