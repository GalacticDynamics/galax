#!/usr/bin/env python
"""Write the orbit animation on the time-dependent parameters page.

One orbit in a static triaxial NFW halo and the same orbit while the halo flattens
(``q1`` goes from 1 to 0.6 over 0.7 Gyr, a
:class:`~galax.potential.params.CustomParameter`), drawn on a canvas as they are
integrated. The page, ``docs/_static/flattening_orbit.html``, is
``scripts/flattening_orbit.template.html`` with the orbits embedded: it needs no
network.

Usage: ``uv run python scripts/make_flattening_orbit.py``
"""

import json
from pathlib import Path

import jax.numpy as jnp

import unxt as u

import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp

HERE = Path(__file__).parent
OUT = HERE.parent / "docs" / "_static" / "flattening_orbit.html"
PLACEHOLDER = '"__DATA__"'
M, R_S, Q_END, T_END, V_Y, N = 5.4e11, 15.0, 0.6, 0.7, 135.0, 400


def flattening(t: u.Q, q_end: float, t_end: u.Q) -> u.Q:
    """Return an axis ratio going from 1 to ``q_end`` at ``t_end``, then fixed."""
    frac = jnp.clip(u.ustrip("Gyr", t) / u.ustrip("Gyr", t_end), 0.0, 1.0)
    return u.Q(1.0 + (q_end - 1.0) * frac, "")


def halo(q1: u.Q | gp.params.AbstractParameter) -> gp.TriaxialNFWPotential:
    """Return the triaxial NFW halo with the given ``q1``."""
    return gp.TriaxialNFWPotential(
        m=u.Q(M, "Msun"), r_s=u.Q(R_S, "kpc"), q1=q1, q2=u.Q(1.0, ""), units="galactic"
    )


def rounded(a: u.Q, unit: str, ndigits: int) -> list[float]:
    """Return the values in ``unit``, rounded, as a list for JSON."""
    return [round(float(v), ndigits) for v in u.ustrip(unit, a)]


def main() -> None:
    """Integrate both orbits and write the page."""
    q1 = gp.params.CustomParameter(func=flattening, args=(Q_END, u.Q(T_END, "Gyr")))
    w0 = gc.PhaseSpacePosition(
        q=u.Q([8.0, 0.0, 0.0], "kpc"), p=u.Q([0.0, V_Y, 0.0], "km/s")
    )
    ts = u.Q(jnp.linspace(0.0, T_END, N), "Gyr")
    static = gd.evaluate_orbit(halo(u.Q(1.0, "")), w0, ts)
    flat = gd.evaluate_orbit(halo(q1), w0, ts)
    data = {
        "t": rounded(ts, "Gyr", 4),
        "xs": rounded(static.q.x, "kpc", 3),
        "ys": rounded(static.q.y, "kpc", 3),
        "xf": rounded(flat.q.x, "kpc", 3),
        "yf": rounded(flat.q.y, "kpc", 3),
        "q1": [round(float(u.ustrip("", q1(t))), 4) for t in ts],
    }
    template = (HERE / "flattening_orbit.template.html").read_text(encoding="utf-8")
    if template.count(PLACEHOLDER) != 1:
        msg = f"the template must hold {PLACEHOLDER} once"
        raise ValueError(msg)
    text = template.replace(PLACEHOLDER, json.dumps(data, separators=(",", ":")))
    OUT.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
