"""Generate `bfeax` reference values for the multipole-profile oracle tests.

Run manually, not as part of the test suite:

    git clone https://github.com/jnibauer/bfeax /tmp/bfeax
    git -C /tmp/bfeax checkout 2341204
    PYTHONPATH=/tmp/bfeax uv run python \
        tests/functional/reference/multipole_profile/generate.py

`bfeax` is deliberately NOT a test dependency: the values are vendored so the
suite stays hermetic and so a change upstream cannot silently move our
reference.
"""

import pathlib

import jax

jax.config.update("jax_enable_x64", True)  # noqa: FBT003

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from bfeax.density_coeffs import density_to_sph_coeffs  # noqa: E402
from bfeax.grid import make_radial_grid  # noqa: E402
from bfeax.poisson import _green_function_integral  # noqa: E402
from bfeax.potential import _lm_keys  # noqa: E402

R_MIN, R_MAX, N_R, L_MAX = 1e-2, 3.0e2, 128, 8


def _spheroid(alpha, beta, gamma, q_y=1.0, q_z=1.0):
    def rho(x, y, z):
        rt = jnp.sqrt(x**2 + (y / q_y) ** 2 + (z / q_z) ** 2)
        return rt ** (-gamma) * (1.0 + rt**alpha) ** ((gamma - beta) / alpha)

    return rho


CASES = {
    # name: (density, symmetry)
    "nfw_sph": (_spheroid(1.0, 3.0, 1.0), "spherical"),
    "nfw_tri": (_spheroid(1.0, 3.0, 1.0, q_y=0.8, q_z=0.5), "triaxial"),
    "hernquist_sph": (_spheroid(1.0, 4.0, 1.0), "spherical"),
    "plummer_sph": (_spheroid(2.0, 5.0, 0.0), "spherical"),
    "jaffe_sph": (_spheroid(1.0, 4.0, 2.0), "spherical"),
}


def main() -> None:
    r = make_radial_grid(N_R, R_MIN, R_MAX)
    out = {"r_knots": np.asarray(r)}

    for name, (rho, symmetry) in CASES.items():
        lm = _lm_keys(L_MAX, symmetry)
        coeffs = density_to_sph_coeffs(rho, r, L_MAX, lm_keys=lm)
        rho_lm = np.stack([np.asarray(coeffs[k]) for k in lm], axis=-1)
        phi_lm = np.stack(
            [np.asarray(_green_function_integral(r, coeffs[k], k[0])) for k in lm],
            axis=-1,
        )
        out[f"{name}_lm"] = np.asarray(lm, dtype=int)
        out[f"{name}_rho_lm"] = rho_lm
        out[f"{name}_phi_lm"] = phi_lm
        print(f"{name}: {len(lm)} modes, rho_lm {rho_lm.shape}, phi_lm {phi_lm.shape}")

    path = pathlib.Path(__file__).parent / "bfeax_reference.npz"
    np.savez_compressed(path, **out)
    print(f"wrote {path} ({path.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
