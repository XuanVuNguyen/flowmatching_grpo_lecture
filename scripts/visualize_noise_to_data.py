r"""Animate the transport of Gaussian noise into a complex 2D distribution.

The plane is split: the **left** half shows $p_{\text{init}}$, an isotropic
Gaussian rendered with density contours, and the **right** half shows
$p_{\text{data}}$, a "two-moons" distribution rendered via KDE contours —
clearly more complex than a Gaussian. A cat photo is anchored as a
callout to one sample of $p_{\text{data}}$ to remind the viewer that,
semantically, the target is a distribution over cat images. A single
particle (red) is animated travelling from the noise distribution to a
point in the data distribution, tracing its trajectory.
"""

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from scipy.stats import gaussian_kde


T_FINAL = 1.0
N_FRAMES = 140
N_SAMPLES = 1500

X_LIM = (-6.2, 6.2)
Y_LIM = (-2.6, 3.4)

LEFT_CENTER = np.array([-3.2, 0.0])
RIGHT_CENTER = np.array([3.0, 0.0])

NOISE_COLOR = "#1f77b4"
DATA_COLOR = "#2ca02c"
TRACE_COLOR = "#e74c3c"


def sample_gaussian(n: int, sigma: float = 0.55, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, size=(n, 2)) + LEFT_CENTER


def sample_two_moons(n: int, noise: float = 0.06, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_a = n // 2
    n_b = n - n_a

    theta_a = rng.uniform(0.0, np.pi, n_a)
    moon_a = np.stack([np.cos(theta_a), np.sin(theta_a)], axis=1)
    moon_a += noise * rng.standard_normal(moon_a.shape)
    moon_a += np.array([-0.35, 0.15])

    theta_b = rng.uniform(np.pi, 2.0 * np.pi, n_b)
    moon_b = np.stack([np.cos(theta_b), np.sin(theta_b)], axis=1)
    moon_b += noise * rng.standard_normal(moon_b.shape)
    moon_b += np.array([0.35, -0.15])

    pts = np.concatenate([moon_a, moon_b], axis=0)
    pts *= 1.1
    return pts + RIGHT_CENTER


def make_cmap(name: str, base: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, [(1, 1, 1, 0), base], N=256)


def draw_gaussian_contour(ax, sigma: float = 0.55):
    xs = np.linspace(LEFT_CENTER[0] - 2.2, LEFT_CENTER[0] + 2.2, 180)
    ys = np.linspace(-2.2, 2.2, 180)
    GX, GY = np.meshgrid(xs, ys)
    pdf = np.exp(
        -((GX - LEFT_CENTER[0]) ** 2 + (GY - LEFT_CENTER[1]) ** 2)
        / (2.0 * sigma ** 2)
    )
    cmap = make_cmap("noise_cmap", NOISE_COLOR)
    ax.contourf(GX, GY, pdf, levels=10, cmap=cmap, alpha=0.85, zorder=1)
    ax.contour(
        GX, GY, pdf, levels=5, colors=NOISE_COLOR,
        linewidths=0.6, alpha=0.55, zorder=2,
    )


def draw_data_contour(ax, samples: np.ndarray):
    xs = np.linspace(RIGHT_CENTER[0] - 2.2, RIGHT_CENTER[0] + 2.2, 200)
    ys = np.linspace(-2.0, 2.0, 200)
    GX, GY = np.meshgrid(xs, ys)
    kde = gaussian_kde(samples.T, bw_method=0.15)
    Z = kde(np.stack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    cmap = make_cmap("data_cmap", DATA_COLOR)
    ax.contourf(GX, GY, Z, levels=10, cmap=cmap, alpha=0.85, zorder=1)
    ax.contour(
        GX, GY, Z, levels=5, colors=DATA_COLOR,
        linewidths=0.6, alpha=0.55, zorder=2,
    )


def make_trajectory(p0: np.ndarray, p1: np.ndarray, n: int) -> np.ndarray:
    """Smooth curved trajectory from p0 to p1 with a gentle vertical detour."""
    s = np.linspace(0.0, 1.0, n)
    eased = 0.5 - 0.5 * np.cos(np.pi * s)
    line = (1.0 - eased)[:, None] * p0 + eased[:, None] * p1
    bump = 0.55 * np.sin(np.pi * s)
    line[:, 1] += bump
    return line


def main(out_dir: Path, cat_image_path: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    x_init = sample_gaussian(N_SAMPLES)
    x_data = sample_two_moons(N_SAMPLES)

    # anchor for the cat callout: a representative point on the upper moon
    anchor_point = RIGHT_CENTER + np.array([-0.55, 1.05])

    # start and end of the animated trajectory — land exactly on the cat point
    rng2 = np.random.default_rng(123)
    p_start = x_init[int(rng2.integers(0, len(x_init)))]
    p_end = anchor_point
    path = make_trajectory(p_start, p_end, N_FRAMES)

    cat_img = mpimg.imread(str(cat_image_path))
    cat_img = cat_img[::18, ::18]

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    draw_gaussian_contour(ax)
    draw_data_contour(ax, x_data)

    ax.text(
        LEFT_CENTER[0], Y_LIM[0] + 0.25,
        r"$p_{\mathrm{init}} = \mathcal{N}(0, I)$",
        ha="center", va="bottom", fontsize=12, color=NOISE_COLOR, zorder=5,
    )
    ax.text(
        RIGHT_CENTER[0], Y_LIM[0] + 0.25,
        r"$p_{\mathrm{data}}$ (cat images)",
        ha="center", va="bottom", fontsize=12, color=DATA_COLOR, zorder=5,
    )

    cat_offset = OffsetImage(cat_img, zoom=0.42)
    cat_ab = AnnotationBbox(
        cat_offset,
        anchor_point,
        xybox=(RIGHT_CENTER[0] + 1.9, 2.6),
        xycoords="data", boxcoords="data",
        arrowprops={
            "arrowstyle": "-|>",
            "connectionstyle": "arc3,rad=-0.25",
            "color": "#333333",
            "lw": 1.3,
            "mutation_scale": 14,
            "shrinkA": 4,
            "shrinkB": 6,
        },
        frameon=True, pad=0.3,
        bboxprops={"edgecolor": "#333333", "linewidth": 1.1, "boxstyle": "round,pad=0.25"},
        zorder=20,
    )
    ax.add_artist(cat_ab)

    # marker on the anchor point so the arrow has something concrete to point at
    ax.plot(
        [anchor_point[0]], [anchor_point[1]],
        marker="o", ms=7, mfc=DATA_COLOR, mec="#1c5b1c", mew=0.9, zorder=6,
    )

    (trail,) = ax.plot([], [], color=TRACE_COLOR, lw=1.8, alpha=0.9, zorder=4)
    (head,) = ax.plot(
        [], [], marker="o", ms=10, mfc=TRACE_COLOR, mec="black", mew=0.7, zorder=5,
    )

    ax.plot(
        [p_start[0]], [p_start[1]],
        marker="s", ms=8, mfc=TRACE_COLOR, mec="black", mew=0.6, zorder=4,
    )

    time_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, fontsize=11,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "gray"},
    )

    def update(frame: int):
        trail.set_data(path[: frame + 1, 0], path[: frame + 1, 1])
        head.set_data([path[frame, 0]], [path[frame, 1]])
        t_now = frame / (N_FRAMES - 1)
        time_text.set_text(f"$t = {t_now:.2f}$")
        return [trail, head, time_text]

    anim = FuncAnimation(
        fig, update, frames=N_FRAMES, interval=33, blit=False, repeat=True,
    )

    fig.tight_layout()
    out_gif = out_dir / "noise_to_data.gif"
    anim.save(out_gif, writer=PillowWriter(fps=30))
    print(f"saved {out_gif}")

    update(N_FRAMES - 1)
    out_png = out_dir / "noise_to_data.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"saved {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    main(
        root / "slides" / "public" / "figures",
        root / "slides" / "public" / "images" / "cat.png",
    )
