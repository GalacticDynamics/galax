# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Draw the galax logo: the GalacticDynamics mark.

A painted spiral galaxy on a dark disk, circled by a ring and set between code
brackets. Every shape is drawn in data units, so a larger image is the same
picture, and the brush strokes come from a fixed seed, so every run draws the
same logo. Re-run it for a larger image::

    uv run docs/_static/make_logo.py                    # favicon.png, 512 px
    uv run docs/_static/make_logo.py --size 2048 big.png
"""

import argparse
import itertools
from pathlib import Path

from collections.abc import Callable

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Polygon

TEAL, PURPLE, NIGHT = "#66a19a", "#7738eb", "#030a23"
RADIUS = 0.6  # of the dark disk
GAP = 0.022  # the clear gap that sets the ring off from what lies under it
# One swoosh of the ring, traced from the GalacticDynamics logo: at each polar
# angle (degrees), the radii of its inner and outer edges. The other swoosh is
# the same shape turned half a circle. Each comes down in front of the disk's
# rim and ends in a point inside it.
SWOOSH = np.array(
    [
        (14, 0.52, 0.52),
        (17, 0.51, 0.565),
        (20, 0.512, 0.612),
        (22, 0.515, 0.646),
        (25, 0.525, 0.695),
        (28, 0.561, 0.747),
        (31, 0.60, 0.807),
        (34, 0.643, 0.853),
        (37, 0.685, 0.897),
        (40, 0.717, 0.928),
        (43, 0.74, 0.947),
        (46, 0.751, 0.952),
        (49, 0.752, 0.947),
        (52, 0.744, 0.933),
        (55, 0.735, 0.906),
        (58, 0.722, 0.875),
        (61, 0.707, 0.839),
        (64, 0.692, 0.804),
        (67, 0.676, 0.767),
        (70, 0.659, 0.732),
        (73, 0.644, 0.698),
        (76, 0.63, 0.666),
        (80, 0.615, 0.615),
    ]
)
WIND = 0.28  # log-spiral rate: r falls by exp(-WIND) per radian anticlockwise


def swoosh(turn: float, upto: float | None = None) -> np.ndarray:
    """Return the outline of one swoosh of the ring, turned by ``turn`` degrees.

    It runs from the outer edge's far end, round the point, and back along the
    inner edge. ``upto`` stops it at that angle, leaving the outline open.
    """
    angle, inner, outer = SWOOSH.T
    phi = np.linspace(angle[0], angle[-1] if upto is None else upto, 200)
    r = np.concatenate(
        [np.interp(phi, angle, outer)[::-1], np.interp(phi, angle, inner)]
    )
    phi = np.deg2rad(np.concatenate([phi[::-1], phi]) + turn)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi)])


def bracket(ax: plt.Axes, sign: int, colour: str) -> None:
    """Draw a thick round-ended < (``sign`` -1) or > (+1)."""
    pts = np.array([[0.557, 0.458], [0.9, 0.0], [0.557, -0.458]]) * [sign, 1]
    half = 0.094  # half the stroke width
    for p, q in itertools.pairwise(pts):
        n = np.array([-(q - p)[1], (q - p)[0]]) / np.linalg.norm(q - p) * half
        ax.add_patch(
            Polygon([p + n, q + n, q - n, p - n], color=colour, lw=0, zorder=0)
        )
    for p in pts:
        ax.add_patch(Circle(p, half, color=colour, lw=0, zorder=0))


def stroke(
    ax: plt.Axes,
    clip: Circle,
    theta0: float,
    r0: float,
    sweep: float,
    width: float,
    colour: str,
    alpha: float,
) -> None:
    """Paint one brush stroke along the spiral, from ``r0`` inwards.

    It runs ``sweep`` radians anticlockwise from angle ``theta0``, swelling to
    ``width`` early and tapering to a point at both ends.
    """
    s = np.linspace(0, 1, 80)
    theta = theta0 + sweep * s
    r = r0 * np.exp(-WIND * sweep * s)
    line = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    tangent = np.gradient(line, axis=0)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    w = 0.5 * width * np.sin(np.pi * s**0.6)[:, None]
    outline = np.concatenate([line + w * normal, (line - w * normal)[::-1]])
    poly = Polygon(outline, color=colour, alpha=alpha, lw=0)
    ax.add_patch(poly)
    poly.set_clip_path(clip)


def arm_colour(r: float, theta: float, rng: np.random.Generator) -> str:
    """Pick a stroke colour: cyan core, blue middle, purple and green rims."""
    if r < 0.15:
        return rng.choice(["#6ff3fb", "#4fd6f5", "#5aa8f0"])
    if r < 0.32:
        return rng.choice(["#2f6be0", "#2a4fc8", "#3a5fd8", "#5a4fe0", "#3f8af0"])
    # The outermost arm is green round the top right; the outer arms are purple
    # down the left, and blue between.
    if r > 0.33 and np.cos(theta - np.deg2rad(40)) > 0.4:
        return rng.choice(["#4fc88a", "#62d9a0", "#5ed49a", "#3aa8c0"])
    if np.cos(theta - np.deg2rad(200)) > 0.5:
        return rng.choice(["#6a3fd0", "#8a55e8", "#7b45d6", "#a77ff0"])
    return rng.choice(["#2f6be0", "#3f8af0", "#4b6fe0", "#6a5be0"])


def galaxy(ax: plt.Axes, rng: np.random.Generator) -> None:
    """Paint the spiral galaxy on the disk: arms of strokes, and a core."""
    clip = Circle((0, 0), RADIUS - 0.05, transform=ax.transData)
    ax.add_patch(Circle((0, 0), 0.2, color="#1d3fa8", lw=0))
    # Each arm is a log spiral, theta = phase - ln(r / 0.1) / WIND. Its strokes
    # start on it, a little scattered, and follow it inwards: long, thin and
    # tapered. Outer strokes go first, so the brighter inner ones lie over them.
    n_arms, n = 4, 130
    arm = rng.integers(0, n_arms, n)
    r0 = rng.uniform(0.14, 0.54, n)
    for a, r in sorted(zip(arm, r0, strict=True), key=lambda ar: -ar[1]):
        theta0 = 2 * np.pi * a / n_arms - np.log(r / 0.1) / WIND
        theta0 += rng.normal(0, 0.22)
        sweep = rng.uniform(1.4, 3.0)
        width = rng.uniform(0.4, 1.0) * (0.02 + 0.12 * r)
        colour = arm_colour(r * np.exp(-WIND * sweep / 3), theta0 + sweep / 3, rng)
        stroke(ax, clip, theta0, r, sweep, width, colour, alpha=0.92)
    # Swirling rings round the core.
    for r in rng.uniform(0.07, 0.22, 18):
        theta0, sweep = rng.uniform(0, 2 * np.pi), rng.uniform(2, 4)
        colour = rng.choice(["#6ff3fb", "#4fd6f5", "#5aa8f0", "#9ff8ff"])
        stroke(ax, clip, theta0, r, sweep, rng.uniform(0.015, 0.035), colour, 0.85)
    # Thin pale streaks, the brush's texture.
    for r in rng.uniform(0.15, 0.5, 40):
        theta0, sweep = rng.uniform(0, 2 * np.pi), rng.uniform(0.8, 1.6)
        pale = tuple(0.6 * c + 0.4 for c in to_rgb(arm_colour(r, theta0, rng)))
        stroke(ax, clip, theta0, r, sweep, 0.006 + 0.012 * r, pale, alpha=0.5)
    # The glowing core: many faint discs, densest at the centre.
    for rr in np.linspace(0.19, 0.02, 25):
        ax.add_patch(Circle((0, 0), rr, color="#9ff8ff", alpha=0.07, lw=0))
    ax.add_patch(Circle((0, 0), 0.075, color="white", lw=0))


def sparkle(ax: plt.Axes, xy: tuple[float, float], size: float, colour: str) -> None:
    """Draw a four-pointed star centred on ``xy``."""
    t = np.arange(8) * np.pi / 4
    r = np.where(np.arange(8) % 2 == 0, size, size / 4)
    outline = np.column_stack([xy[0] + r * np.sin(t), xy[1] + r * np.cos(t)])
    ax.add_patch(Polygon(outline, color=colour, lw=0))


def draw(ax: plt.Axes, rng: np.random.Generator) -> None:
    """Draw the logo on ``ax``, in the square [-1, 1] x [-1, 1]."""
    bracket(ax, -1, TEAL)
    bracket(ax, 1, PURPLE)

    ax.add_patch(Circle((0, 0), RADIUS, color=NIGHT, lw=0))
    galaxy(ax, rng)

    # Stars in the dark margin, sparkles, and two planets.
    phi, rho = rng.uniform(0, 2 * np.pi, 45), rng.uniform(0.45, RADIUS - 0.03, 45)
    for x, y in zip(rho * np.cos(phi), rho * np.sin(phi), strict=True):
        colour = rng.choice(["white", "white", "#9ff8ff", "#d7a8ef"])
        ax.add_patch(Circle((x, y), rng.uniform(0.004, 0.009), color=colour, lw=0))
    for xy, size, colour in (
        ((-0.36, 0.39), 0.035, "white"),
        ((-0.43, 0.29), 0.035, "#7fe0b0"),
        ((-0.2, 0.48), 0.022, "#7fe0b0"),
        ((0.43, -0.29), 0.03, "white"),
        ((0.29, -0.39), 0.04, "#e8ffff"),
        ((0.51, -0.35), 0.02, "#d7a8ef"),
    ):
        sparkle(ax, xy, size, colour)
    ax.add_patch(Circle((-0.49, 0.16), 0.035, color="#a77ff0", lw=0))
    ax.add_patch(Circle((0.18, -0.44), 0.028, color="#4f9fe0", lw=0))

    # The ring, in front. `gaps` cuts the clear gap round it.
    for turn in (0, 180):
        ax.add_patch(Polygon(swoosh(turn), color=NIGHT, lw=0))


def gaps(ax: plt.Axes) -> None:
    """Draw, in white on black, what is cut out round each swoosh.

    That is a gap along the half that lies over the disk and a bracket, and
    the hollow between the swoosh and the disk, which hides the bracket's end.
    The swoosh's other end merges into the disk.
    """
    points_per_unit = ax.figure.get_figwidth() * 72 / 2  # the axes span 2
    angle, inner, _ = SWOOSH.T
    phi = np.linspace(angle[0], angle[-1], 200)
    r_in = np.interp(phi, angle, inner)
    r = np.concatenate([r_in, np.minimum(r_in, RADIUS)[::-1]])
    for turn in (0, 180):
        ax.plot(
            *swoosh(turn, upto=60).T,
            color="white",
            lw=2 * GAP * points_per_unit,
            solid_capstyle="round",
        )
        rad = np.deg2rad(np.concatenate([phi, phi[::-1]]) + turn)
        hollow = np.column_stack([r * np.cos(rad), r * np.sin(rad)])
        ax.add_patch(Polygon(hollow, color="white", lw=0))
        # Above the line, which matplotlib draws over patches by default.
        ax.add_patch(Polygon(swoosh(turn), color="black", lw=0, zorder=3))


def render(size: int, paint: Callable[[plt.Axes], None], background: str) -> np.ndarray:
    """Return ``paint`` drawn on [-1, 1] x [-1, 1] as an RGBA array in [0, 1]."""
    dpi = 100
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi, facecolor=background)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(-1, 1), ylim=(-1, 1), aspect="equal")
    ax.axis("off")
    paint(ax)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba()) / 255
    plt.close(fig)
    return image


def main() -> None:
    """Parse the command line and save the logo."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "out",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("favicon.png"),
        help="output file (default: favicon.png)",
    )
    parser.add_argument("--size", type=int, default=512, help="pixels per side")
    args = parser.parse_args()

    rng = np.random.default_rng(201030)
    image = render(args.size, lambda ax: draw(ax, rng), background="none")
    # The gaps are cut out, so they show whatever the logo sits on.
    image[..., 3] *= 1 - render(args.size, gaps, background="black")[..., 0]
    plt.imsave(args.out, image)


if __name__ == "__main__":
    main()
