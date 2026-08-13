"""``galax`` Potentials."""

__all__: tuple[str, ...] = ()


# NOTE: this avoids a circular import
# isort: split
from .base import AbstractPotential
from .plot import ProxyAbstractPotential

ProxyAbstractPotential.deliver(AbstractPotential)
