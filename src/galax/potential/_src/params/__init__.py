"""`galax.potential.params`."""

__all__ = [
    # Fields
    "ParameterField",
    # Parameters
    "AbstractParameter",
    "ParameterCallable",
    "ConstantParameter",
    "LinearParameter",
    "CustomParameter",
    # Attributes
    "AbstractParametersAttribute",
    "ParametersAttribute",
    "CompositeParametersAttribute",
]

from .attr import (
    AbstractParametersAttribute,
    CompositeParametersAttribute,
    ParametersAttribute,
)
from .base import AbstractParameter, ParameterCallable
from .constant import ConstantParameter
from .core import CustomParameter, LinearParameter
from .field import ParameterField
