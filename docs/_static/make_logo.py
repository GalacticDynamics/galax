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
GAP = 0.022  # the clear gap that sets the ring off from the disk it crosses
# The painted galaxy's radius, measured from the GalacticDynamics logo. A dark
# margin of disk rings it, which each swoosh crosses before it dives under it.
PAINT = 0.52
# One swoosh of the ring, traced from the GalacticDynamics logo: at each polar
# angle (degrees), the radii of its inner and outer edges. The other swoosh is
# the same shape turned half a circle. Each crosses in front of the disk's rim
# and ends in a point, merging into the dark.
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
WIND = 0.12  # the outer arms fall inwards this much per radian anticlockwise


def swoosh(turn: float) -> np.ndarray:
    """Return the outline of one swoosh of the ring, turned by ``turn`` degrees."""
    angle, inner, outer = SWOOSH.T
    phi = np.linspace(angle[0], angle[-1], 200)
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
    colour: str | tuple[float, float, float],
    alpha: float,
    wind: float = WIND,
) -> None:
    """Paint one brush stroke, from ``r0`` inwards along a spiral.

    It runs ``sweep`` radians anticlockwise from angle ``theta0``, falling
    ``wind`` inwards per radian (0 for a circular arc), swelling to ``width``
    early and tapering to a point at both ends.
    """
    s = np.linspace(0, 1, 80)
    theta = theta0 + sweep * s
    r = r0 - wind * sweep * s
    line = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    tangent = np.gradient(line, axis=0)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    w = 0.5 * width * np.sin(np.pi * s**0.6)[:, None]
    outline = np.concatenate([line + w * normal, (line - w * normal)[::-1]])
    poly = Polygon(outline, color=colour, alpha=alpha, lw=0)
    ax.add_patch(poly)
    poly.set_clip_path(clip)


def arm_colour(theta: float, rng: np.random.Generator) -> str:
    """Pick an outer arm's colour from its angle.

    Mostly blue, purple down the left, green round the top right.
    """
    deg = np.rad2deg(theta) % 360
    if 100 < deg < 280 and rng.random() < 0.75:
        return rng.choice(["#6a3fd0", "#7b4fe0", "#8a5ce8", "#5b45c8"])
    if (deg > 350 or deg < 85) and rng.random() < 0.75:
        return rng.choice(["#4fc88a", "#5ed49a", "#45b58a", "#6fdca8"])
    return rng.choice(["#3b78e6", "#4a8ef0", "#55a8f5", "#2f62d8", "#62b8f5"])


def galaxy(ax: plt.Axes, rng: np.random.Generator) -> None:
    """Paint the spiral galaxy: an inner whirl, a dark lane, and outer arms.

    The structure is measured from the GalacticDynamics logo, unwrapped into
    polar coordinates: circular bands of cyan and blue out to r = 0.2, a dark
    lane, then thin tapered arms that fall inwards as they run anticlockwise.
    """
    clip = Circle((0, 0), PAINT, transform=ax.transData)
    # The galaxy's dark ground, which hides the point of each swoosh.
    ax.add_patch(Circle((0, 0), PAINT, color=NIGHT, lw=0))
    # Outer arms, between r = 0.22 and the edge, outermost first. Four arms
    # wind in, theta = phase + (0.5 - r) / WIND; each stroke starts near one.
    for r0 in np.sort(rng.uniform(0.3, PAINT + 0.04, 60))[::-1]:
        sweep = min(rng.uniform(1.2, 2.6), (r0 - 0.22) / WIND)
        phase = np.pi / 2 * rng.integers(4) + np.deg2rad(20)
        theta0 = phase + (0.5 - r0) / WIND + rng.normal(0, 0.25)
        colour = arm_colour(theta0 + sweep / 3, rng)
        width = rng.uniform(0.015, 0.05)
        stroke(ax, clip, theta0, r0, sweep, width, colour, alpha=0.95)
    # Pale streaks along them, the brush's texture.
    for r0 in rng.uniform(0.27, PAINT, 40):
        sweep = min(rng.uniform(0.8, 1.8), (r0 - 0.24) / WIND)
        theta0 = rng.uniform(0, 2 * np.pi)
        pale = tuple(0.5 * c + 0.5 for c in to_rgb(arm_colour(theta0, rng)))
        stroke(ax, clip, theta0, r0, sweep, rng.uniform(0.004, 0.01), pale, 0.8)
    # A few dim strokes in the dark lane.
    for r0 in rng.uniform(0.2, 0.25, 8):
        theta0, sweep = rng.uniform(0, 2 * np.pi), rng.uniform(1, 2.5)
        stroke(ax, clip, theta0, r0, sweep, 0.012, "#1f3a95", 0.9, wind=0.01)
    # The inner whirl: bands winding gently in, blue outside, cyan in, over a
    # ground that fades out into the lane.
    for rr, alpha in ((0.21, 0.35), (0.19, 0.6), (0.17, 1)):
        ax.add_patch(Circle((0, 0), rr, color="#2b5fd0", alpha=alpha, lw=0))
    for r0 in np.sort(rng.uniform(0.08, 0.21, 45))[::-1]:
        theta0, sweep = rng.uniform(0, 2 * np.pi), rng.uniform(2, 5)
        if r0 > 0.15:
            colour = rng.choice(["#4a8ef0", "#1d3a9a", "#5ab0f5", "#2f62d8"])
        else:
            colour = rng.choice(["#6ff3fb", "#4fd6f5", "#9ff8ff", "#3fa8ea"])
        width = rng.uniform(0.008, 0.022)
        stroke(ax, clip, theta0, r0, sweep, width, colour, 0.95, wind=0.015)
    # The glowing core: many faint discs, densest at the centre, cyan fading
    # into white.
    for rr in np.linspace(0.14, 0.08, 10):
        ax.add_patch(Circle((0, 0), rr, color="#9ff8ff", alpha=0.15, lw=0))
    for rr in np.linspace(0.085, 0.05, 8):
        ax.add_patch(Circle((0, 0), rr, color="white", alpha=0.3, lw=0))
    ax.add_patch(Circle((0, 0), 0.055, color="white", lw=0))


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
    # The ring crosses the disk's rim and runs under the galaxy. `gaps` cuts
    # the clear gap along its inner edge.
    for turn in (0, 180):
        ax.add_patch(Polygon(swoosh(turn), color=NIGHT, lw=0))
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


def gaps(ax: plt.Axes) -> None:
    """Draw, in white on black, what is cut out along each swoosh's inner edge.

    Outside the disk, that is the hollow between the swoosh and the disk,
    which hides the bracket's end. Where the swoosh crosses into the disk, it
    is a band GAP wide, tapering away by its point, where the swoosh merges
    into the dark.
    """
    angle, inner, _ = SWOOSH.T
    phi = np.linspace(angle[0], angle[-1], 400)
    r_in = np.interp(phi, angle, inner)
    width = GAP * np.clip((phi - angle[0]) / 10, 0, 1) * (phi < 50)
    r = np.concatenate([r_in, np.minimum(RADIUS, r_in - width)[::-1]])
    for turn in (0, 180):
        rad = np.deg2rad(np.concatenate([phi, phi[::-1]]) + turn)
        cut = np.column_stack([r * np.cos(rad), r * np.sin(rad)])
        ax.add_patch(Polygon(cut, color="white", lw=0))


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
