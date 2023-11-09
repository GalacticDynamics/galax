"""galdynamix: Galactic Dynamix in Jax"""

from __future__ import annotations

__all__ = ["ImmutableDict"]

from collections.abc import ItemsView, Iterator, KeysView, Mapping, ValuesView
from typing import (
    Self,
    TypeVar,
)

from jax.tree_util import register_pytree_node_class

V = TypeVar("V")


@register_pytree_node_class
class ImmutableDict(Mapping[str, V]):
    def __init__(self, /, *args: tuple[str, V], **kwargs: V) -> None:
        self._data: dict[str, V] = dict(*args, **kwargs)

    def __getitem__(self, key: str) -> V:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __hash__(self) -> int:
        return hash(tuple(self._data.items()))

    def keys(self) -> KeysView[str]:
        return self._data.keys()

    def values(self) -> ValuesView[V]:
        return self._data.values()

    def items(self) -> ItemsView[str, V]:
        return self._data.items()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._data!r})"

    # === PyTree ===

    def tree_flatten(self) -> tuple[tuple[V, ...], tuple[str, ...]]:
        """Specifies a flattening recipe.

        Returns
        -------
        tuple[V, ...] tuple[str, ...]
            a pair of an iterable with the children to be flattened recursively,
            and some opaque auxiliary data to pass back to the unflattening recipe.
            The auxiliary data is stored in the treedef for use during unflattening.
            The auxiliary data could be used, e.g., for dictionary keys.
        """
        return (tuple(self._data.values()), tuple(self._data.keys()))

    @classmethod
    def tree_unflatten(
        cls: type[Self], aux_data: tuple[str, ...], children: tuple[V, ...]
    ) -> Self[str, V]:  # type: ignore[misc]
        """Specifies an unflattening recipe.

        Params:
        aux_data: the opaque data that was specified during flattening of the
            current treedef.
        children: the unflattened children

        Returns:
        a re-constructed object of the registered type, using the specified
        children and auxiliary data.
        """
        return cls(tuple(zip(aux_data, children, strict=True)))  # type: ignore[arg-type]
