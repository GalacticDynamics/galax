"""Constants. Private module."""

__all__ = ["BURKERT_CONST", "LOG2", "SQRT2"]

from typing import Final

import quaxed.numpy as jnp

BURKERT_CONST: Final = 3 * jnp.log(jnp.asarray(2.0)) - 0.5 * jnp.pi
LOG2: Final = jnp.log(jnp.asarray(2.0))
SQRT2: Final = jnp.sqrt(jnp.asarray(2.0))
