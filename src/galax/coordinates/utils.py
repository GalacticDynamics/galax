"""Utilities. Semi-public API.

`coordinates` is the lowest subpackage that uses these, so they live here and
`potential` / `dynamics` consume them from this module rather than reaching
into `_src`. Not exported from `galax.coordinates` itself -- import the module
explicitly.
"""

__all__ = ["batched_shape", "vector_batched_shape"]

from ._src.shape import batched_shape, vector_batched_shape
