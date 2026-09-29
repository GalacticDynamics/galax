"""Array shape utilities.

`coordinates` is the lowest subpackage that uses these, so they live here and
`potential` / `dynamics` consume them from this public module rather than
reaching into `_src`.
"""

__all__ = ["ArrayAnyShape", "batched_shape", "vector_batched_shape"]

from ._src.shape import ArrayAnyShape, batched_shape, vector_batched_shape
