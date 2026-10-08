"""Spherical-harmonic expansion machinery. Private API.

See the public API in `galax.potential.harmonic`.
"""

__all__ = [
    "angular_grid",
    "asymptotic_coeffs",
    "default_angular_resolution",
    "eval_log_spline",
    "eval_log_spline_asympt",
    "fit_log_spline",
    "harmonic_coeffs",
    "lm_keys",
    "real_ylm",
    "solve_poisson_lm",
]

from .asympt import asymptotic_coeffs, eval_log_spline_asympt
from .poisson import solve_poisson_lm
from .project import (
    angular_grid,
    default_angular_resolution,
    harmonic_coeffs,
    lm_keys,
    real_ylm,
)
from .spline import eval_log_spline, fit_log_spline
