# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Draw the galax logo: the GalacticDynamics mark.

A spiral galaxy on a dark disk, circled by a ring and set between code
brackets. The galaxy is random points along logarithmic spiral arms, from a
fixed seed, so every run draws the same logo. Re-run it for a larger image::

    uv run docs/_static/make_logo.py                    # favicon.png, 512 px
    uv run docs/_static/make_logo.py --size 2048 big.png
"""

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as MplPath

TEAL, PURPLE, NIGHT = "#66a19a", "#7738eb", "#030a23"
# Galaxy colours, centre outwards: white core, cyan, blue, violet, green.
GALAXY = LinearSegmentedColormap.from_list(
    "galaxy", ["#ffffff", "#9df7fd", "#478edd", "#7a4fd8", "#5ec98a"]
)


def ellipse(a: float, b: float, angle: float, n: int = 200) -> np.ndarray:
    """Return the vertices of an ellipse, semi-axes ``a`` and ``b``, rotated."""
    t = np.linspace(0, 2 * np.pi, n)
    x, y = a * np.cos(t), b * np.sin(t)
    c, s = np.cos(angle), np.sin(angle)
    return np.column_stack([c * x - s * y, s * x + c * y])


def draw(ax: plt.Axes, rng: np.random.Generator) -> None:
    """Draw the logo on ``ax``, in the square [-1, 1] x [-1, 1]."""
    # Code brackets, < and >, behind the ring and disk.
    for sign, colour in ((-1, TEAL), (1, PURPLE)):
        ax.plot(
            [sign * 0.55, sign * 0.88, sign * 0.55],
            [0.55, 0.0, -0.55],
            color=colour,
            lw=1,  # rescaled in `main` to the figure's size
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=0,
        )

    # The ring: an outer ellipse minus an inner one, thickest at its tips. The
    # disk, drawn next, hides the part that crosses it.
    tilt = np.deg2rad(39)
    outer, inner = ellipse(0.95, 0.42, tilt), ellipse(0.8, 0.37, tilt)[::-1]
    verts = np.concatenate([outer, inner])
    codes = np.full(len(verts), MplPath.LINETO)
    codes[[0, len(outer)]] = MplPath.MOVETO
    ax.add_patch(PathPatch(MplPath(verts, codes), color=NIGHT, lw=0))

    radius = 0.57
    ax.add_patch(Circle((0, 0), radius, color=NIGHT, lw=0))

    # Spiral arms: log spirals r = r0 exp(k theta), wound clockwise, with
    # scatter that widens outwards.
    n, n_arms, k = 4000, 4, 0.3
    r = 0.5 * rng.uniform(0, 1, n) ** 0.8 + 0.01
    arm = rng.integers(0, n_arms, n)
    theta = -np.log(r / 0.02) / k + 2 * np.pi * arm / n_arms
    theta += rng.normal(0, 0.12 + 0.25 * r, n)
    ax.scatter(
        r * np.cos(theta),
        r * np.sin(theta),
        s=rng.uniform(0.5, 6, n),
        c=r / 0.5 + rng.normal(0, 0.08, n),
        cmap=GALAXY,
        vmin=0,
        vmax=1,
        alpha=0.6,
        lw=0,
    )

    # Glowing core.
    for rr, alpha in ((0.16, 0.15), (0.1, 0.3), (0.06, 0.6), (0.035, 1)):
        ax.add_patch(Circle((0, 0), rr, color="#e8ffff", alpha=alpha, lw=0))

    # Stars, as four-pointed sparkles and dots, and two planets.
    sparkles = np.array([[-0.27, 0.32], [0.3, -0.42], [-0.38, -0.18], [0.36, 0.35]])
    ax.scatter(*sparkles.T, marker=(4, 1, 0), s=40, color="white", lw=0)
    phi, rho = rng.uniform(0, 2 * np.pi, 40), rng.uniform(0.3, 0.55, 40)
    ax.scatter(rho * np.cos(phi), rho * np.sin(phi), s=1.5, color="white", lw=0)
    ax.add_patch(Circle((-0.49, 0.15), 0.035, color="#b783f2", lw=0))
    ax.add_patch(Circle((0.18, -0.45), 0.03, color="#5db1e2", lw=0))


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

    dpi = 100
    fig = plt.figure(figsize=(args.size / dpi, args.size / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(-1, 1), ylim=(-1, 1), aspect="equal")
    ax.axis("off")

    draw(ax, np.random.default_rng(201030))
    # Scale everything sized in points (line widths, markers) with the image,
    # so a larger image is the same picture, not thinner strokes.
    scale = args.size / 512
    for line in ax.lines:
        line.set_linewidth(0.06 * args.size)
    for coll in ax.collections:
        coll.set_sizes(coll.get_sizes() * scale**2)

    fig.savefig(args.out, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
