"""The dynamics custom_types modules re-export the aliases held by `galax.fields`."""

import pytest

import galax.dynamics._src.custom_types as gdt
import galax.dynamics.custom_types as gt
import galax.fields.custom_types as ft

# Public aliases. The imported helpers `Any`, `Array`, `Int`, `Real`,
# `ScalarLike`, `TypeAlias` and `u` are not aliases and are not re-exported.
GT_NAMES = (
    "BBtFloatQuSz0",
    "BBtLikeSz0",
    "BBtQorVSz0",
    "BBtQorVSz3",
    "BBtQuSz0",
    "BBtQuSz3",
    "BBtQuSz4",
    "BBtSz0",
    "BBtSz3",
    "BBtSz4",
    "BBtSz6",
    "BBtSz7",
    "BtFloatQuSz0",
    "BtQuSz0",
    "BtQuSz3",
    "BtSz3",
    "BtSz6",
    "FloatQuSz0",
    "FloatQuSz3",
    "FloatSz0",
    "FloatSz3",
    "IntSz0",
    "LikeSz0",
    "OptArgs",
    "QuSz0",
    "QuSz1",
    "QuSz3",
    "QuSzTime",
    "RealScalarLike",
    "Shape",
    "Sz0",
    "Sz3",
    "Sz33",
    "Sz6",
    "SzAny",
    "SzN",
    "SzTime",
)

GDT_NAMES = (
    "Aarr",
    "BBtP",
    "BBtParr",
    "BBtQ",
    "BBtQP",
    "BBtQParr",
    "BBtQarr",
    "BQarr",
    "BtAarr",
    "BtP",
    "BtPAarr",
    "BtParr",
    "BtQ",
    "BtQP",
    "BtQParr",
    "BtQarr",
    "P",
    "PAarr",
    "Parr",
    "Q",
    "QP",
    "QParr",
    "Qarr",
)


@pytest.mark.parametrize("name", GT_NAMES)
def test_public_names_are_the_same_objects(name):
    assert getattr(gt, name) is getattr(ft, name)


@pytest.mark.parametrize("name", GDT_NAMES)
def test_private_names_are_the_same_objects(name):
    assert getattr(gdt, name) is getattr(ft, name)
