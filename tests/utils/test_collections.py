from __future__ import annotations

import pytest

from galdynamix.utils import ImmutableDict


class TestImmutableDict:
    @pytest.mark.parametrize(
        ("arg", "kwargs"),
        [
            ((), {}),
            ({"a": 1, "b": 2}, {}),
            ([("a", 1), ("b", 2)], {}),
            ((("a", 1), ("b", 2)), {}),
        ],
    )
    def test_init(self, arg, kwargs):
        """Test initialization.

        Should be able to initialize with all the same input types as a regular
        dictionary.
        """
        d = ImmutableDict(arg, **kwargs)
        assert isinstance(d, ImmutableDict)
        assert d._data == dict(arg, **kwargs)

    def test_getitem(self):
        """Test `__getitem__`."""
        d = ImmutableDict(a=1, b=2)
        assert d["a"] == 1
        assert d["b"] == 2

    def test_iter(self):
        """Test `__iter__`."""
        d = ImmutableDict(a=1, b=2)
        assert list(d) == ["a", "b"]

    def test_len(self):
        """Test `__len__`."""
        d = ImmutableDict(a=1, b=2)
        assert len(d) == 2

    def test_hash(self):
        """Test `__hash__`."""
        d = ImmutableDict(a=1, b=2)
        assert hash(d) == hash(tuple(d.items()))

        # Not hashable if values aren't hashable.
        d = ImmutableDict(a=1, b={"c"})
        with pytest.raises(TypeError, match="unhashable type: 'set'"):
            hash(d)

    def test_keys(self):
        """Test `keys`."""
        d = ImmutableDict(a=1, b=2)
        assert list(d.keys()) == ["a", "b"]

    def test_values(self):
        """Test `values`."""
        d = ImmutableDict(a=1, b=2)
        assert list(d.values()) == [1, 2]

    def test_items(self):
        """Test `items`."""
        d = ImmutableDict(a=1, b=2)
        assert list(d.items()) == [("a", 1), ("b", 2)]

    def test_repr(self):
        """Test `__repr__`."""
        d = ImmutableDict(a=1, b=2)
        assert repr(d) == "ImmutableDict({'a': 1, 'b': 2})"

    # === Test pytree methods ===

    def test_tree_flatten(self):
        """Test `tree_flatten`."""
        d = ImmutableDict(a=1, b=2)
        assert d.tree_flatten() == ((1, 2), ("a", "b"))

    def test_tree_unflatten(self):
        """Test `tree_unflatten`."""
        d = ImmutableDict.tree_unflatten(("a", "b"), (1, 2))
        assert d == ImmutableDict(a=1, b=2)

        # round-trip
        d = ImmutableDict(a=1, b=2)
        flattened = d.tree_flatten()
        d2 = ImmutableDict.tree_unflatten(flattened[1], flattened[0])
        assert d2 == d
