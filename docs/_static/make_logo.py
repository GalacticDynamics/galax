# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Draw the galax logo: the GalacticDynamics mark.

A painted spiral galaxy on a dark disk, circled by a ring and set between code
brackets. The disk, ring and brackets are outlines traced from the org's logo;
the galaxy is painted over them, its brush strokes from a fixed seed, so every
run draws the same logo. The docs use it as an SVG, sharp at any size; for a
bitmap, name a .png and give its size::

    uv run docs/_static/make_logo.py                     # favicon.svg
    uv run docs/_static/make_logo.py --size 2048 big.png
"""

import argparse
import io
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, PathPatch, Polygon
from matplotlib.path import Path as MplPath

TEAL, PURPLE, NIGHT = "#66a19a", "#7738eb", "#030a23"
RADIUS = 0.6  # of the dark disk
# The painted galaxy's radius, measured from the GalacticDynamics logo: a dark
# margin of the disk rings it.
PAINT = 0.5
WIND = 0.12  # the outer arms fall inwards this much per radian anticlockwise
# The flat shapes, traced from the GalacticDynamics logo: the edge of each at
# half intensity, simplified to within 0.0008 (0.005 for the straight-sided
# brackets). Points are x,y in [-1, 1] x [-1, 1]. DISK is the dark disk with the
# two swooshes of its ring, the gaps where the ring crosses the rim already
# cut; the brackets are what shows of each, reaching a little under the disk so
# no seam shows where they meet.
DISK = """
0.456,0.737 0.460,0.738 0.536,0.738 0.572,0.732 0.585,0.728 0.611,0.718
0.629,0.708 0.645,0.698 0.657,0.687 0.673,0.671 0.684,0.655 0.694,0.639
0.700,0.623 0.705,0.605 0.710,0.578 0.710,0.530 0.704,0.493 0.691,0.447
0.685,0.429 0.684,0.420 0.680,0.416 0.656,0.356 0.649,0.343 0.646,0.334
0.624,0.293 0.609,0.266 0.592,0.240 0.590,0.233 0.583,0.225 0.512,0.126
0.511,0.122 0.514,0.108 0.514,0.090 0.516,0.083 0.520,0.082 0.567,0.152
0.572,0.156 0.577,0.152 0.584,0.124 0.593,0.067 0.593,0.049 0.596,0.017
0.596,-0.008 0.594,-0.017 0.593,-0.053 0.584,-0.106 0.575,-0.143 0.562,-0.188
0.551,-0.215 0.539,-0.243 0.510,-0.297 0.488,-0.331 0.467,-0.359 0.445,-0.386
0.407,-0.426 0.380,-0.449 0.350,-0.473 0.316,-0.496 0.275,-0.520 0.230,-0.542
0.191,-0.557 0.138,-0.573 0.115,-0.578 0.060,-0.586 0.051,-0.586 0.042,-0.588
0.026,-0.588 0.017,-0.590 -0.024,-0.590 -0.033,-0.588 -0.056,-0.588 -0.065,-0.586
-0.074,-0.586 -0.101,-0.583 -0.150,-0.573 -0.206,-0.557 -0.252,-0.538 -0.291,-0.518
-0.344,-0.486 -0.383,-0.457 -0.396,-0.445 -0.406,-0.438 -0.443,-0.400 -0.480,-0.355
-0.498,-0.327 -0.502,-0.318 -0.510,-0.306 -0.500,-0.272 -0.488,-0.242 -0.467,-0.199
-0.451,-0.174 -0.451,-0.168 -0.457,-0.154 -0.460,-0.150 -0.464,-0.150 -0.480,-0.168
-0.494,-0.190 -0.508,-0.218 -0.527,-0.263 -0.532,-0.279 -0.544,-0.309 -0.560,-0.365
-0.563,-0.382 -0.564,-0.395 -0.568,-0.414 -0.568,-0.454 -0.567,-0.463 -0.564,-0.479
-0.557,-0.498 -0.551,-0.514 -0.538,-0.534 -0.526,-0.550 -0.510,-0.564 -0.492,-0.578
-0.476,-0.587 -0.430,-0.606 -0.407,-0.613 -0.387,-0.617 -0.376,-0.620 -0.367,-0.621
-0.351,-0.624 -0.335,-0.624 -0.325,-0.627 -0.316,-0.626 -0.309,-0.627 -0.254,-0.627
-0.225,-0.624 -0.213,-0.624 -0.181,-0.619 -0.120,-0.605 -0.118,-0.605 -0.116,-0.609
-0.150,-0.632 -0.186,-0.652 -0.227,-0.672 -0.273,-0.691 -0.298,-0.699 -0.310,-0.702
-0.319,-0.706 -0.339,-0.712 -0.346,-0.713 -0.351,-0.716 -0.360,-0.717 -0.367,-0.720
-0.375,-0.720 -0.383,-0.724 -0.391,-0.724 -0.399,-0.727 -0.417,-0.731 -0.430,-0.731
-0.440,-0.734 -0.464,-0.735 -0.472,-0.738 -0.544,-0.738 -0.552,-0.735 -0.569,-0.734
-0.595,-0.728 -0.625,-0.716 -0.636,-0.711 -0.665,-0.692 -0.683,-0.673 -0.701,-0.648
-0.714,-0.619 -0.721,-0.596 -0.721,-0.586 -0.724,-0.577 -0.724,-0.566 -0.726,-0.557
-0.726,-0.527 -0.724,-0.518 -0.724,-0.505 -0.722,-0.496 -0.721,-0.482 -0.717,-0.473
-0.717,-0.464 -0.710,-0.439 -0.707,-0.430 -0.706,-0.423 -0.703,-0.418 -0.690,-0.381
-0.673,-0.340 -0.645,-0.284 -0.616,-0.234 -0.615,-0.229 -0.605,-0.217 -0.600,-0.206
-0.580,-0.179 -0.575,-0.170 -0.514,-0.090 -0.511,-0.085 -0.510,-0.074 -0.512,-0.061
-0.515,-0.051 -0.519,-0.048 -0.589,-0.136 -0.593,-0.140 -0.599,-0.138 -0.600,-0.124
-0.603,-0.115 -0.604,-0.097 -0.607,-0.088 -0.607,-0.078 -0.608,-0.070 -0.607,-0.061
-0.610,-0.053 -0.610,0.044 -0.607,0.054 -0.608,0.060 -0.607,0.074 -0.604,0.085
-0.603,0.097 -0.600,0.106 -0.599,0.115 -0.597,0.122 -0.596,0.129 -0.593,0.138
-0.589,0.154 -0.582,0.176 -0.578,0.192 -0.564,0.225 -0.537,0.279 -0.531,0.288
-0.523,0.302 -0.516,0.311 -0.501,0.336 -0.490,0.350 -0.458,0.388 -0.430,0.417
-0.408,0.438 -0.373,0.465 -0.339,0.489 -0.296,0.515 -0.264,0.531 -0.229,0.546
-0.193,0.559 -0.136,0.575 -0.102,0.581 -0.060,0.586 -0.047,0.586 -0.038,0.588
0.031,0.588 0.095,0.581 0.140,0.571 0.191,0.555 0.218,0.545 0.268,0.522
0.289,0.511 0.335,0.482 0.380,0.447 0.402,0.427 0.409,0.418 0.431,0.397
0.463,0.357 0.483,0.329 0.492,0.313 0.492,0.307 0.489,0.300 0.468,0.258
0.449,0.225 0.449,0.220 0.456,0.204 0.458,0.202 0.462,0.201 0.473,0.215
0.487,0.238 0.509,0.281 0.536,0.352 0.550,0.402 0.553,0.425 0.553,0.466
0.549,0.491 0.542,0.509 0.535,0.523 0.520,0.545 0.503,0.561 0.490,0.571
0.472,0.583 0.458,0.590 0.424,0.604 0.410,0.608 0.380,0.616 0.344,0.622
0.335,0.622 0.310,0.626 0.246,0.626 0.238,0.624 0.207,0.622 0.168,0.617
0.115,0.606 0.113,0.609 0.115,0.612 0.127,0.621 0.182,0.652 0.213,0.666
0.282,0.694 0.312,0.705 0.375,0.723 0.423,0.733
"""
LEFT = """
-0.573,0.555 -0.536,0.545 -0.507,0.518 -0.484,0.455 -0.488,0.411 -0.522,0.348
-0.790,-0.001 -0.618,-0.226 -0.576,-0.171 -0.569,-0.174 -0.674,-0.365 -0.722,-0.530
-0.726,-0.520 -0.709,-0.423 -0.984,-0.063 -0.986,0.054 -0.637,0.514 -0.609,0.542
"""
RIGHT = """
0.678,0.427 0.959,0.067 0.975,0.024 0.975,-0.024 0.962,-0.061 0.933,-0.104
0.605,-0.529 0.574,-0.550 0.542,-0.558 0.510,-0.545 0.482,-0.516 0.463,-0.470
0.465,-0.411 0.493,-0.357 0.767,-0.003 0.583,0.234
"""


def outline(points: str) -> np.ndarray:
    """Parse one of the traced outlines into an (n, 2) array."""
    return np.array([p.split(",") for p in points.split()], dtype=float)


def smooth(points: np.ndarray) -> MplPath:
    """Return a closed Catmull-Rom curve through ``points``, as Bezier curves.

    Each span is exactly one cubic Bezier, which SVG stores as is, so a stroke
    is smooth at any size from eight points a side.
    """
    p0, p2, p3 = np.roll(points, 1, 0), np.roll(points, -1, 0), np.roll(points, -2, 0)
    c1, c2 = points + (p2 - p0) / 6, p2 - (p3 - points) / 6
    spans = np.stack([c1, c2, p2], axis=1).reshape(-1, 2)
    verts = np.concatenate([points[:1], spans, points[:1]])
    codes = [MplPath.MOVETO, *[MplPath.CURVE4] * len(spans), MplPath.CLOSEPOLY]
    return MplPath(verts, codes)


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
    s = np.linspace(0, 1, 8)
    theta = theta0 + sweep * s
    r = r0 - wind * sweep * s
    line = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    tangent = np.gradient(line, axis=0)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    w = 0.5 * width * np.sin(np.pi * s**0.6)[:, None]
    # Both sides meet at the tips; each tip is kept once.
    outline = np.concatenate([line + w * normal, (line - w * normal)[-2:0:-1]])
    poly = PathPatch(smooth(outline), color=colour, alpha=alpha, lw=0)
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
    # The strokes stop a little short of PAINT, as the org's do.
    clip = Circle((0, 0), PAINT - 0.02, transform=ax.transData)
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
    ax.add_patch(Polygon(outline(LEFT), color=TEAL, lw=0))
    ax.add_patch(Polygon(outline(RIGHT), color=PURPLE, lw=0))
    disk = Polygon(outline(DISK), color=NIGHT, lw=0)
    ax.add_patch(disk)
    galaxy(ax, rng)

    # Stars in the dark margin, sparkles, and two planets: on the disk only, so
    # none floats in a gap.
    phi, rho = rng.uniform(0, 2 * np.pi, 45), rng.uniform(0.45, RADIUS - 0.03, 45)
    for x, y in zip(rho * np.cos(phi), rho * np.sin(phi), strict=True):
        colour = rng.choice(["white", "white", "#9ff8ff", "#d7a8ef"])
        star = Circle((x, y), rng.uniform(0.004, 0.009), color=colour, lw=0)
        ax.add_patch(star)
        star.set_clip_path(disk)
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


def main() -> None:
    """Parse the command line and save the logo."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "out",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("favicon.svg"),
        help="output file, SVG or PNG by its extension (default: favicon.svg)",
    )
    parser.add_argument(
        "--size", type=int, default=512, help="pixels per side, for a PNG"
    )
    args = parser.parse_args()

    dpi = 100
    fig = plt.figure(figsize=(args.size / dpi, args.size / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(-1, 1), ylim=(-1, 1), aspect="equal")
    ax.axis("off")
    draw(ax, np.random.default_rng(201030))
    if args.out.suffix == ".svg":
        # No timestamp and fixed element ids, so a re-run gives the same file.
        mpl.rcParams["svg.hashsalt"] = "logo"
        svg = io.StringIO()
        fig.savefig(svg, format="svg", transparent=True, metadata={"Date": None})
        # Coordinates to 0.01 pt, far finer than any screen shows, and no
        # trailing spaces on path lines, which pre-commit would strip.
        text = re.sub(r"-?\d+\.\d{3,}", lambda m: f"{float(m[0]):.2f}", svg.getvalue())
        args.out.write_text(
            "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
        )
    else:
        fig.savefig(args.out, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
