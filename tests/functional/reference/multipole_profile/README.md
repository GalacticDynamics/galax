# `bfeax` reference values

`bfeax_reference.npz` holds ρ_lm(r) and Φ_lm(r) computed by
[`jnibauer/bfeax`](https://github.com/jnibauer/bfeax) at commit `2341204`, used
as a bit-level oracle for `MultipoleProfilePotential`.

Grid: `r_min=1e-2`, `r_max=300`, `n_r=128`, `l_max=8`, log-spaced. Profiles:
`rho = r~^-gamma (1 + r~^alpha)^((gamma-beta)/alpha)` with
`r~ = sqrt(x^2 + (y/q_y)^2 + (z/q_z)^2)`.

| case            | (alpha, beta, gamma) | q_y | q_z | symmetry  |
| --------------- | -------------------- | --- | --- | --------- |
| `nfw_sph`       | (1, 3, 1)            | 1   | 1   | spherical |
| `nfw_tri`       | (1, 3, 1)            | 0.8 | 0.5 | triaxial  |
| `hernquist_sph` | (1, 4, 1)            | 1   | 1   | spherical |
| `plummer_sph`   | (2, 5, 0)            | 1   | 1   | spherical |
| `jaffe_sph`     | (1, 4, 2)            | 1   | 1   | spherical |

`Phi_lm` is in G=1 units — multiply by G for a unit-carrying comparison.

Regenerate with `generate.py`. Do not regenerate to make a failing test pass: a
mismatch means the port changed, which is the thing these values exist to
detect. The one deliberate divergence is the outer-tail sign fix
([jnibauer/bfeax#1](https://github.com/jnibauer/bfeax/issues/1)), which is
handled by narrowing the oracle test rather than by moving the reference.
