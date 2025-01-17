"""Type hints for `galax.dynamics`."""

__all__: list[str] = []

from typing import TypeAlias

from jaxtyping import Array, Shaped

from unxt.quantity import AbstractQuantity

Qarr: TypeAlias = Shaped[Array, "3"]
BtQarr: TypeAlias = Shaped[Qarr, "*batch"]
BBtQarr: TypeAlias = Shaped[Qarr, "*#batch"]

Q: TypeAlias = Shaped[AbstractQuantity, "3"]
BtQ: TypeAlias = Shaped[Q, "*batch"]
BBtQ: TypeAlias = Shaped[Q, "*#batch"]

Parr: TypeAlias = Shaped[Array, "3"]
BtParr: TypeAlias = Shaped[Parr, "*batch"]
BBtParr: TypeAlias = Shaped[Parr, "*#batch"]

P: TypeAlias = Shaped[AbstractQuantity, "3"]
BtP: TypeAlias = Shaped[P, "*batch"]
BBtP: TypeAlias = Shaped[P, "*#batch"]

Aarr: TypeAlias = Shaped[Array, "3"]
BtAarr: TypeAlias = Shaped[Aarr, "*batch"]

QParr: TypeAlias = tuple[Qarr, Parr]
BtQParr: TypeAlias = tuple[BtQarr, BtParr]
BBtQParr: TypeAlias = tuple[BBtQarr, BBtParr]

QP: TypeAlias = tuple[Q, P]
BtQP: TypeAlias = tuple[BtQ, BtP]
BBtQP: TypeAlias = tuple[BBtQ, BBtP]

PAarr: TypeAlias = tuple[Parr, Aarr]
BtPAarr: TypeAlias = tuple[BtParr, BtAarr]
