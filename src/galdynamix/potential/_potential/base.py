__all__ = ["AbstractPotentialBase", "AbstractPotential"]

import abc
import uuid
from dataclasses import KW_ONLY, fields
from typing import TYPE_CHECKING, Any

import astropy.units as u
import equinox as eqx
import jax.numpy as xp
from astropy.constants import G as _G
from jax import grad, hessian, jacfwd
from jaxtyping import Array, Float

from galdynamix.integrate._base import AbstractIntegrator
from galdynamix.integrate._builtin import DiffraxIntegrator
from galdynamix.typing import FloatScalar, Vector3, Vector6, VectorN
from galdynamix.units import UnitSystem, dimensionless
from galdynamix.utils import partial_jit

if TYPE_CHECKING:
    from galdynamix.potential._potential.composite import CompositePotential


class AbstractPotentialBase(eqx.Module):  # type: ignore[misc]
    """Potential Class."""

    units: eqx.AbstractVar[UnitSystem]

    ###########################################################################
    # Abstract methods that must be implemented by subclasses

    @abc.abstractmethod
    def potential_energy(self, q: Vector3, /, t: FloatScalar) -> FloatScalar:
        """Compute the potential energy at the given position(s).

        Parameters
        ----------
        q : :class:`~jax.Array`
            The position to compute the value of the potential.
        t : :class:`~jax.Array`
            The time at which to compute the value of the potential.

        Returns
        -------
        E : :class:`~jax.Array`
            The potential energy per unit mass or value of the potential.
        """
        raise NotImplementedError

    ###########################################################################
    # Parsing

    def _init_units(self) -> None:
        G = 1 if self.units == dimensionless else _G.decompose(self.units).value
        object.__setattr__(self, "_G", G)

        from galdynamix.potential._potential.param.field import ParameterField

        # Handle unit conversion for all fields, e.g. the parameters.
        for f in fields(self):
            # Process ParameterFields
            param = getattr(self.__class__, f.name, None)
            if isinstance(param, ParameterField):
                # Set, since the ``.units`` are now known
                param.__set__(self, getattr(self, f.name))

            # Other fields, check their metadata
            elif "dimensions" in f.metadata:
                value = getattr(self, f.name)
                if isinstance(value, u.Quantity):
                    value = value.to_value(
                        self.units[f.metadata.get("dimensions")],
                        equivalencies=f.metadata.get("equivalencies", None),
                    )
                    object.__setattr__(self, f.name, value)

    ###########################################################################
    # Core methods that use the above implemented functions

    @partial_jit()
    def __call__(self, q: Vector3, /, t: FloatScalar) -> FloatScalar:
        """Compute the potential energy at the given position(s).

        See Also
        --------
        potential_energy
        """
        return self.potential_energy(q, t)

    @partial_jit()
    def gradient(self, q: Vector3, /, t: FloatScalar) -> Vector3:
        """Compute the gradient of the potential at the given position(s).

        Parameters
        ----------
        q : :class:`~jax.Array`
            The position to compute the value of the potential. If the
            input position object has no units (i.e. is an `~numpy.ndarray`),
            it is assumed to be in the same unit system as the potential.
        t : :class:`~jax.Array`
            The time at which to compute the value of the potential.

        Returns
        -------
        :class:`~jax.Array`
            The gradient of the potential.
        """
        return grad(self.potential_energy)(q, t)

    @partial_jit()
    def density(self, q: Vector3, /, t: FloatScalar) -> Vector3:
        """Compute the density value at the given position(s).

        Parameters
        ----------
        q : :class:`~jax.Array`
            The position to compute the value of the potential. If the
            input position object has no units (i.e. is an `~numpy.ndarray`),
            it is assumed to be in the same unit system as the potential.
        t : :class:`~jax.Array`
            The time at which to compute the value of the potential.

        Returns
        -------
        :class:`~jax.Array`
            The potential energy or value of the potential.
        """
        # Note: trace(jacobian(gradient)) is faster than trace(hessian(energy))
        lap = xp.trace(jacfwd(self.gradient)(q, t))
        return lap / (4 * xp.pi * self._G)

    @partial_jit()
    def hessian(self, q: Vector3, /, t: FloatScalar) -> Vector3:
        """Compute the Hessian of the potential at the given position(s).

        Parameters
        ----------
        q : :class:`~jax.Array`
            The position to compute the value of the potential. If the
            input position object has no units (i.e. is an `~numpy.ndarray`),
            it is assumed to be in the same unit system as the potential.
        t : :class:`~jax.Array`
            The time at which to compute the value of the potential.

        Returns
        -------
        :class:`~jax.Array`
            The Hessian matrix of second derivatives of the potential.
        """
        return hessian(self.potential_energy)(q, t)

    ###########################################################################
    # Convenience methods

    @partial_jit()
    def acceleration(self, q: Vector3, /, t: FloatScalar) -> Vector3:
        """Compute the acceleration due to the potential at the given position(s).

        Parameters
        ----------
        q : :class:`~jax.Array`
            Position to compute the acceleration at.
        t : :class:`~jax.Array`
            Time at which to compute the acceleration.

        Returns
        -------
        :class:`~jax.Array`
            The acceleration. Will have the same shape as the input
            position array, ``q``.
        """
        return -self.gradient(q, t)

    # =========================================================================

    @partial_jit()
    def _integrator_F(
        self, t: FloatScalar, xv: Vector6, args: tuple[Any, ...]
    ) -> FloatScalar:
        return xp.hstack([xv[3:6], self.acceleration(xv[:3], t)])

    @partial_jit(static_argnames=("Integrator", "integrator_kw"))
    def integrate_orbit(
        self,
        qp0: Vector6,
        t0: FloatScalar,
        t1: FloatScalar,
        ts: Float[Array, "time"] | None,
        *,
        Integrator: type[AbstractIntegrator] = DiffraxIntegrator,
        integrator_kw: dict[str, Any] | None = None,
    ) -> VectorN:
        from galdynamix.dynamics._orbit import Orbit

        integrator = Integrator(self._integrator_F, **(integrator_kw or {}))
        ws = integrator.run(qp0, t0, t1, ts)
        return Orbit(q=ws[:, :3], p=ws[:, 3:-1], t=ws[:, -1], potential=self)

    ###########################################################################
    # Composite potentials

    def __add__(self, other: Any) -> "CompositePotential":
        if not isinstance(other, AbstractPotentialBase):
            return NotImplemented

        from galdynamix.potential._potential.composite import CompositePotential

        if isinstance(other, CompositePotential):
            return other.__ror__(self)

        return CompositePotential({str(uuid.uuid4()): self, str(uuid.uuid4()): other})


# ===========================================================================


class AbstractPotential(AbstractPotentialBase):
    _: KW_ONLY
    units: UnitSystem = eqx.field(
        default=None,
        converter=lambda x: dimensionless if x is None else UnitSystem(x),
        static=True,
    )
    _G: float = eqx.field(init=False, static=True, repr=False)

    def __post_init__(self) -> None:
        self._init_units()
