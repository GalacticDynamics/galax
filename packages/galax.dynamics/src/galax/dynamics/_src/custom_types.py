"""Type hints for `galax.dynamics`."""

__all__: tuple[str, ...] = ()

from jaxtyping import Real
from typing import TypeAlias

import galax.dynamics.custom_types as gt
from galax.orbits.custom_types import (
    BBtP as BBtP,
    BBtParr as BBtParr,
    BBtQ as BBtQ,
    BBtQarr as BBtQarr,
    BBtQP as BBtQP,
    BBtQParr as BBtQParr,
    BtAarr as BtAarr,
    BtP as BtP,
    BtPAarr as BtPAarr,
    BtParr as BtParr,
    BtQ as BtQ,
    BtQarr as BtQarr,
    BtQP as BtQP,
    BtQParr as BtQParr,
    Parr as Parr,
    Qarr as Qarr,
    QParr as QParr,
)

# ---------------------------
# Q

BQarr: TypeAlias = Real[gt.Sz3, "batch"]

Q: TypeAlias = gt.QuSz3

# ---------------------------
# P

P: TypeAlias = gt.QuSz3

# ---------------------------
# A

Aarr: TypeAlias = gt.Sz3

# ---------------------------
# QP

QP: TypeAlias = tuple[Q, P]

# ---------------------------
# PA

PAarr: TypeAlias = tuple[Parr, Aarr]
