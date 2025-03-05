"""galax: Galactic Dynamix in Jax."""

__all__ = ["AbstractLoopStrategy", "AbstractVMap", "Vectorize", "VMap", "Scan"]

from typing import NoReturn


class AbstractLoopStrategy:
    """Abstract class for loop strategies.

    Examples
    --------
    >>> from galax.utils.loop_strategies import AbstractLoopStrategy
    >>> try: AbstractLoopStrategy()
    ... except TypeError as e: print(e)
    Cannot instantiate LoopStrategy classes

    """

    def __new__(cls) -> NoReturn:
        """Raise an error when trying to instantiate the class."""
        msg = "Cannot instantiate LoopStrategy classes"
        raise TypeError(msg)


class Determine(AbstractLoopStrategy):
    """Loop strategy for :func:`jax.vmap`."""


class NoLoop(AbstractLoopStrategy):
    """Loop strategy for not performing a loop."""


class AbstractVMap(AbstractLoopStrategy):
    """Loop strategy for :func:`jax.vmap`."""


class Vectorize(AbstractVMap):
    """Loop strategy for :func:`jax.numpy.vectorize`."""


class VMap(AbstractVMap):
    """Loop strategy for flattening then :func:`jax.vmap`."""


class Scan(AbstractLoopStrategy):
    """Loop strategy for :func:`jax.vmap`."""
