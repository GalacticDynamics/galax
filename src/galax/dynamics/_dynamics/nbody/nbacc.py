"""``galax`` dynamics."""
# ruff:noqa: F401

__all__: list[str] = []

from abc import abstractmethod
from functools import partial
from typing import TypeAlias

import equinox as eqx
from jaxtyping import Array, Float

import quaxed.array_api as xp
from unxt import AbstractUnitSystem, Quantity

Mass: TypeAlias = Float[Array, "N"]
MassQ: TypeAlias = Float[Quantity["mass"], "N"]
Position: TypeAlias = Float[Array, "N 3"]
PositionQ: TypeAlias = Float[Quantity["length"], "N 3"]
Velocity: TypeAlias = Float[Array, "N 3"]
Acceleration: TypeAlias = Float[Array, "N 3"]
AccelerationQ: TypeAlias = Float[Quantity["acceleration"], "N 3"]


class AbstractNBodyAcceleration(eqx.Module):  # type: ignore[misc]
    @abstractmethod
    def __call__(
        self,
        t: Float[Array, ""],
        y: PositionQ,
        mass: MassQ,
        args: tuple[Quantity],
    ) -> AccelerationQ: ...


# =============================================================================
# Direct N-body gravitational acceleration


@partial(eqx.filter_jit, inline=True)
def pairwise_gravitational_acceleration(
    q: PositionQ,
    masses: MassQ,
    *,
    G: Quantity,
    eps: Quantity["area"],
) -> AccelerationQ:
    """Compute the direct N-body gravitational acceleration between particles.

    This implementation requires computing the full pairwise distance matrix
    between all particles, which scales as O(N^2). For large N, this can be
    slow and memory-intensive.

    Parameters
    ----------
    q : array-like, shape (N, 3)
        The positions of the particles.
    masses : array-like, shape (N,)
        The masses of the particles.
    G : float, optional
        The gravitational constant. Default is 1.
    eps : float, optional
        Softening length to avoid division by zero. Default is 1e-12.

    Returns
    -------
    acc : array-like, shape (N, 3)
        The gravitational accelerations of the particles.

    Examples
    --------
    >>> q = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    >>> masses = np.array([1, 1, 1])
    >>> pairwise_gravitational_acceleration(q, masses)
    array([[ 0.        ,  0.        ,  0.        ],
           [-0.5       ,  0.        ,  0.        ],
           [ 0.        , -0.5       ,  0.        ]])

    """
    # Compute the squared L2 norms of the positions
    d2 = xp.sum(q**2, axis=1)  # shape: N

    # Compute pairwise squared distances using the L2 norms
    pairwise_d2 = d2[:, None] + d2[None, :] - 2 * q @ q.T

    # Compute the difference vectors ri - rj between all pairs of points
    diff = q[:, None, :] - q[None, :, :]  # shape: (N, N, 3)

    # Compute the inverse of the cubed distances 1 / |ri - rj|^3
    inv_dist_cubed = Quantity(
        xp.pow((pairwise_d2 + eps).to_units_value(d2.unit), -1.5), q.unit**-3
    )

    # Compute the pairwise gravitational accelerations
    acc_factors = G * masses * inv_dist_cubed  # Shape: (N, N)
    acc: AccelerationQ = xp.sum(diff * acc_factors[..., None], axis=0)  # shape: (N, 3)

    return acc


class DirectNBodyAcceleration(AbstractNBodyAcceleration):
    """Vector field for direct N-body gravitational acceleration."""

    softening_length: Float[Quantity["length"], ""]

    @partial(eqx.filter_jit, inline=True)
    def __call__(
        self,
        t: Float[Array, ""],  # noqa: ARG002
        y: PositionQ,
        masses: MassQ,
        args: tuple[Quantity],
    ) -> AccelerationQ:
        """Vector field for diffrax."""
        (G,) = args
        eps = self.softening_length**2
        return pairwise_gravitational_acceleration(y, masses, G=G, eps=eps)
