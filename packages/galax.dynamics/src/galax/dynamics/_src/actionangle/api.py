"""Dispatch stubs for action-angle-frequency estimation."""

__all__ = [
    "staeckel_fudge_delta",
    "staeckel_fudge",
    "isochrone_actions_angles_freqs",
]

from typing import Any

from plum import dispatch


@dispatch.abstract
def staeckel_fudge_delta(*args: Any, **kwargs: Any) -> Any:
    """Estimate the focal distance ``delta`` of the Staeckel fudge.

    Overloads:

    - ``(pot, q, t)``
    - ``(pot, w)``
    """
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def staeckel_fudge(*args: Any, **kwargs: Any) -> Any:
    """Estimate actions, angles and frequencies with the Staeckel fudge.

    Overloads:

    - ``(pot, q, p, t, delta, *, phi_inf=None)``
    - ``(pot, w, delta, *, phi_inf=None)``
    """
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def isochrone_actions_angles_freqs(*args: Any, **kwargs: Any) -> Any:
    """Compute actions, angles and frequencies in an isochrone potential.

    Overloads:

    - ``(pot, q, p, t)``
    - ``(pot, w)``
    """
    raise NotImplementedError  # pragma: no cover
