"""Type hints for `galax.dynamics`."""

__all__: list[str] = []

from typing import TypeAlias

import galax._custom_types as gt

# ---------------------------
# Q

Qarr: TypeAlias = gt.Sz3
BtQarr: TypeAlias = gt.BtSz3
BBtQarr: TypeAlias = gt.BBtSz3

Q: TypeAlias = gt.QuSz3
BtQ: TypeAlias = gt.BtQuSz3
BBtQ: TypeAlias = gt.BBtQuSz3

# ---------------------------
# P

Parr: TypeAlias = gt.Sz3
BtParr: TypeAlias = gt.BtSz3
BBtParr: TypeAlias = gt.BBtSz3

P: TypeAlias = gt.QuSz3
BtP: TypeAlias = gt.BtQuSz3
BBtP: TypeAlias = gt.BBtQuSz3

# ---------------------------
# A

Aarr: TypeAlias = gt.Sz3
BtAarr: TypeAlias = gt.BtSz3

# ---------------------------
# QP

QParr: TypeAlias = tuple[Qarr, Parr]
BtQParr: TypeAlias = tuple[BtQarr, BtParr]
BBtQParr: TypeAlias = tuple[BBtQarr, BBtParr]

QP: TypeAlias = tuple[Q, P]
BtQP: TypeAlias = tuple[BtQ, BtP]
BBtQP: TypeAlias = tuple[BBtQ, BBtP]

# ---------------------------
# PA

PAarr: TypeAlias = tuple[Parr, Aarr]
BtPAarr: TypeAlias = tuple[BtParr, BtAarr]
