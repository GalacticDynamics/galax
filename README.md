<h1 align='center'> galax </h1>
<h2 align="center">Galactic and Gravitational Dynamics</h2>

## Installation

[![PyPI platforms][pypi-platforms]][pypi-link]
[![PyPI version][pypi-version]][pypi-link]

<!-- [![Conda-Forge][conda-badge]][conda-link] -->

```bash
pip install galax
```

## Documentation

[![Documentation Status][rtd-badge]][rtd-link]

Coming soon. In the meantime, if you've used `gala`, then `galax` should be
familiar!

## Quick example

Let's compute an orbit!

```python
import jax.numpy as jnp

import unxt as u
import coordinax as cx
import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp

w = gc.PhaseSpaceCoordinate(
    q=u.Quantity([8, 0, 0], "kpc"),
    p=u.Quantity([0, 220, 0], "km/s"),
    t=u.Quantity(0, "Myr"),
)

pot = gp.MilkyWayPotential()

orbit = gd.evaluate_orbit(pot, w, u.Quantity(jnp.linspace(0, 1, 100), "Gyr"))
print(orbit)
# Orbit(
#     q=<CartesianPos3D: (x, y, z) [kpc]
#         [[ 8.     0.     0.   ]
#          ...
#          [ 7.804 -0.106  0.   ]]>,
#     p=<CartesianVel3D: (x, y, z) [kpc / Myr]
#         [[ 0.     0.225  0.   ]
#          ...
#          [ 0.018  0.23   0.   ]]>,
#     t=Quantity(Array([0., ..., 1000.], dtype=float64), unit='Myr')
# )

orbit_sph = orbit.vconvert(cx.vecs.LonLatSphericalPos)
print(orbit_sph)
# Orbit(
#     q=<LonLatSphericalPos: (lon[rad], lat[deg], distance[kpc])
#         [[0.000e+00 3.858e-16 8.000e+00]
#          ...
#          [6.270e+00 3.858e-16 7.805e+00]]>,
#     p=<LonLatSphericalVelocity (lon[rad / Myr], lat[deg / Myr], distance[kpc / Myr])
#         [[ 0.028  0.     0.   ]
#          ...
#          [ 0.03   0.     0.015]]>,
#     t=Quantity(Array([0., ..., 1000.], dtype=float64), unit='Myr')
# )
```

## Citation

[![DOI][zenodo-badge]][zenodo-link]

If you found this library to be useful in academic work, then please cite.

## Development

[![Actions Status][actions-badge]][actions-link]

We welcome contributions!

Galax uses [`uv`](https://docs.astral.sh/uv) for development dependency
management (make sure you have `uv` version >=0.4.27 installed). For example, to
run the test suite, install the `test` group and run `pytest` with:

```
uv sync --group test
uv run pytest src docs tests
```

### Contributors

See the
[AUTHORS.rst](https://github.com/GalacticDynamics/galax/blob/main/AUTHORS.rst)
file for a complete list of contributors to the project.

The [`GalacticDynamics/galax`](https://github.com/GalacticDynamics/galax)
maintainers would like to thank
[@Michael Anckaert](https://github.com/MichaelAnckaert) for transferring the
`galax` project domain on [`PyPI`][pypi-link] for use by this package. Without
his generosity this package would have had a worse name.

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/GalacticDynamics/galax/workflows/CI/badge.svg
[actions-link]:             https://github.com/GalacticDynamics/galax/actions
[codecov-badge]:            https://codecov.io/gh/GalacticDynamics/galax/graph/badge.svg?token=PC553LZFFJ
[codecov-link]:             https://codecov.io/gh/GalacticDynamics/galax
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/galax
[conda-link]:               https://github.com/conda-forge/galax-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/GalacticDynamics/galax/discussions
[pypi-link]:                https://pypi.org/project/galax/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/galax
[pypi-version]:             https://img.shields.io/pypi/v/galax
[rtd-badge]:                https://readthedocs.org/projects/galax/badge/?version=latest
[rtd-link]:                 https://galax.readthedocs.io/en/latest/?badge=latest
[zenodo-badge]:             https://zenodo.org/badge/706347349.svg
[zenodo-link]:              https://zenodo.org/doi/10.5281/zenodo.11553324

<!-- prettier-ignore-end -->
