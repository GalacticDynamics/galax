"""Evaluate a potential's parameters at a time."""

__all__ = ["eval_params"]

from typing import Any

from plum import dispatch

import unxt as u

from galax.potential._src.base import AbstractPotential
from galax.potential._src.base_multi import AbstractCompositePotential


@dispatch
def eval_params(
    pot: AbstractPotential, t: Any, /, *, ustrip: bool = True
) -> dict[str, Any]:
    """Evaluate the parameters of a potential at time ``t``.

    Parameters
    ----------
    pot : `galax.potential.AbstractPotential`
        The potential.
    t : Any
        The time(s), anything `unxt.Quantity.from_` accepts as a time. A bare
        number is taken to be in the potential's time unit.
    ustrip : bool, keyword-only
        If `True` (default), strip each value to the potential's unit system
        and return plain arrays, ready to be passed to the potential's jitted
        functions. If `False`, return Quantities.

    Returns
    -------
    dict[str, Any]
        One entry per `galax.potential.params.ParameterField` of the
        potential, plus ``"G"`` from the potential's constants. For a
        composite potential, a dict of these for each component.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp
    >>> from galax.potential.params import eval_params

    >>> pot = gp.KeplerPotential(m_tot=u.Q(1e12, "Msun"), units="galactic")
    >>> sorted(eval_params(pot, 0.0))
    ['G', 'm_tot']

    >>> eval_params(pot, 0.0, ustrip=False)["m_tot"]
    Q(1.e+12, 'solMass')

    """
    t = u.Q.from_(t, pot.units["time"])
    out = {
        name: getattr(pot, name)(
            t, ustrip=pot.units[field.dimensions] if ustrip else None
        )
        for name, field in type(pot).param_objs.items()
    }
    G = pot.constants["G"]
    out["G"] = G.value if ustrip else G
    return out


@dispatch
def eval_params(
    pot: AbstractCompositePotential, t: Any, /, *, ustrip: bool = True
) -> dict[str, Any]:
    """Evaluate each component of a composite potential at time ``t``.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp
    >>> from galax.potential.params import eval_params

    >>> kepler = gp.KeplerPotential(m_tot=u.Q(1e12, "Msun"), units="galactic")
    >>> composite = gp.CompositePotential(kepler=kepler)
    >>> list(eval_params(composite, 0.0))
    ['kepler']

    """
    return {k: eval_params(p, t, ustrip=ustrip) for k, p in pot.items()}
