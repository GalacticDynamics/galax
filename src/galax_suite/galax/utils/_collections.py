"""galax: Galactic Dynamix in Jax."""

__all__ = ["ImmutableDict"]

from collections.abc import ItemsView, Iterable, Iterator, KeysView, Mapping, ValuesView
from typing import Any, TypeVar

from jax.tree_util import register_pytree_node_class

V = TypeVar("V")
T = TypeVar("T")


@register_pytree_node_class
class ImmutableDict(Mapping[str, V]):
    """Immutable string-keyed dictionary.

    Parameters
    ----------
    *args : tuple[str, V]
        Key-value pairs.
    **kwargs : V
        Key-value pairs.

    Examples
    --------
    >>> from galax.utils import ImmutableDict
    >>> d = ImmutableDict(a=1, b=2)
    >>> d
    ImmutableDict({'a': 1, 'b': 2})
    """

    def __init__(
        self,
        /,
        *args: Mapping[str, V] | tuple[str, V] | Iterable[tuple[str, V]],
        **kwargs: V,
    ) -> None:
        self._data: dict[str, V] = dict(*args, **kwargs)

    def __getitem__(self, key: str) -> V:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __hash__(self) -> int:
        """Hash.

        Normally, dictionaries are not hashable because they are mutable.
        However, this dictionary is immutable, so we can hash it.
        """
        return hash(tuple(self._data.items()))

    def keys(self) -> KeysView[str]:
        return self._data.keys()

    def values(self) -> ValuesView[V]:
        return self._data.values()

    def items(self) -> ItemsView[str, V]:
        return self._data.items()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._data!r})"

    def __or__(self, value: Any, /) -> "ImmutableDict[V]":
        if not isinstance(value, Mapping):
            return NotImplemented

        return type(self)(self._data | dict(value))

    def __ror__(self, value: Any) -> Any:
        return value | self._data

    # === PyTree ===

    def tree_flatten(self) -> tuple[tuple[V, ...], tuple[str, ...]]:
        """Flatten dict to the values (and keys).

        Returns
        -------
        tuple[V, ...] tuple[str, ...]
            A pair of an iterable with the values to be flattened recursively,
            and the keys to pass back to the unflattening recipe.
        """
        return (tuple(self._data.values()), tuple(self._data.keys()))

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[str, ...],
        children: tuple[V, ...],
    ) -> "ImmutableDict":  # type: ignore[type-arg] # TODO: upstream beartype fix for ImmutableDict[V]
        """Unflatten.

        Params:
        aux_data: the opaque data that was specified during flattening of the
            current treedef.
        children: the unflattened children

        Returns
        -------
        a re-constructed object of the registered type, using the specified
        children and auxiliary data.
        """
        return cls(tuple(zip(aux_data, children, strict=True)))
