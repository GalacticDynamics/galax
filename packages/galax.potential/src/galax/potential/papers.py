"""`galax.potential.papers`: potentials set up as in published papers.

Each function is named for its paper's DOI, with every character that is not a
letter or digit replaced by ``_`` and a ``doi_`` prefix, so
``10.1093/mnras/stad551`` is ``doi_10_1093_mnras_stad551``. Its docstring
gives the paper and what the function returns; the documentation has the full
table.
"""

__all__ = ["doi_10_1093_mnras_stad551"]

from ._src.builtin.flattened_gnfw import doi_10_1093_mnras_stad551
