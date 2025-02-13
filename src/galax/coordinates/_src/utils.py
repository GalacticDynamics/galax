"""Utilities for phase-space coordinates. Private module."""

__all__ = ["PSPVConvertOptions", "SLICE_ALL"]

from typing import TYPE_CHECKING, Final, TypeAlias

import coordinax as cx

if TYPE_CHECKING:
    from typing import NotRequired, TypedDict

    class PSPVConvertOptions(TypedDict):
        q: type[cx.vecs.AbstractPos]
        p: NotRequired[type[cx.vecs.AbstractVel] | None]

else:  # need runtime for jaxtyping
    PSPVConvertOptions: TypeAlias = dict[
        str, type[cx.vecs.AbstractPos] | type[cx.vecs.AbstractVel] | None
    ]


SLICE_ALL: Final = slice(None)
