"""Symmetry vocabulary for potentials."""

__all__ = ["Symmetry"]

import enum


class Symmetry(enum.StrEnum):
    """The symmetry a potential asserts about itself.

    A potential declares its symmetry so that callers can tell what an input is
    allowed to leave unspecified. For example, a `coordinax.vecs.RadialPos` is
    only a well-defined position for a `Symmetry.SPHERICAL` potential: turning a
    radius into a 3D position otherwise requires choosing a direction.

    Being a `enum.StrEnum`, the members compare equal to -- and may be passed
    as -- their plain string values.

    Examples
    --------
    >>> import galax.potential as gp

    >>> gp.Symmetry.SPHERICAL
    <Symmetry.SPHERICAL: 'spherical'>

    >>> gp.Symmetry("spherical") is gp.Symmetry.SPHERICAL
    True

    >>> gp.Symmetry.SPHERICAL == "spherical"
    True

    `None` is an alias for `Symmetry.NONE`:

    >>> gp.Symmetry(None) is gp.Symmetry.NONE
    True

    An unknown string is an error naming the valid values. Note that
    "axisymmetric" is not one of them -- see the note below:

    >>> try:
    ...     gp.Symmetry("axisymmetric")
    ... except ValueError as e:
    ...     print(e)
    Unknown symmetry 'axisymmetric'. Use one of ('none', 'spherical',
    'zrotation', 'zrotation_zreflection', 'plane_reflection') or None.

    Notes
    -----
    **Why no "axisymmetric".** Elsewhere, "axisymmetric" often means
    invariance under z-rotation *and* the three plane reflections --
    `Symmetry.PLANE_REFLECTION` here, not `Symmetry.ZROTATION`. To avoid
    that ambiguity, this vocabulary spells z-rotation and z-reflection out
    as separate members instead of using the word.

    **Future.** These are composable bits -- z-rotation, z-reflection, and the
    three plane reflections are independent -- so this is expected to become a
    flag type (`enum.Flag`), at which point the members here become
    combinations rather than distinct values. Tracked at
    `GalacticDynamics/galax#189
    <https://github.com/GalacticDynamics/galax/issues/189>`_ and
    `#854 <https://github.com/GalacticDynamics/galax/issues/854>`_.

    """

    NONE = "none"
    """No symmetry is asserted."""

    SPHERICAL = "spherical"
    """Invariant under any rotation."""

    ZROTATION = "zrotation"
    """Invariant under rotation about z. Says nothing about ``z -> -z``."""

    ZROTATION_ZREFLECTION = "zrotation_zreflection"
    """`ZROTATION`, plus invariance under ``z -> -z``."""

    PLANE_REFLECTION = "plane_reflection"
    """Invariant under reflection about all three principal planes."""

    @classmethod
    def _missing_(cls, value: object) -> "Symmetry":
        """Accept `None` as an alias for `NONE`, and name the valid values.

        The likely guesses -- ``"axisymmetric"``, ``"triaxial"`` -- are not
        this vocabulary's words; see the note above.
        """
        if value is None:
            return cls.NONE
        names = tuple(s.value for s in cls)
        msg = f"Unknown symmetry {value!r}. Use one of {names} or None."
        raise ValueError(msg)
