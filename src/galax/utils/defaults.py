"""Globally assumed default values across Galax."""

__all__ = ["DEFAULT_TIME"]

import unxt as u

# Default time value used when time is optional
DEFAULT_TIME = u.Quantity(0.0, "Myr")
