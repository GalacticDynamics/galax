"""Utilities for phase-space coordinates. Private module."""

__all__ = ("PSPVConvertOptions", "SLICE_ALL", "getitem_dispatcher", "getitem")

from typing import TYPE_CHECKING, Any, Final, TypeAlias

import plum

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


getitem_dispatcher = plum.Dispatcher()


@getitem_dispatcher.abstract
def getitem(self: Any, index: Any, /) -> Any:
    """Return a new object with the given slice applied."""
    raise NotImplementedError  # pragma: no cover --- IGNORE ---
