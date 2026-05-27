"""Shared helpers for the visualize_*.py scripts.

Every script renders a GIF plus a representative PNG into
``slides/public/figures``. The helpers here collect the bits that were
duplicated across scripts: output paths, the colour palette, the
standard "t = ..." annotation, GIF/PNG save calls, and the
gaussian + two-moons + cat-callout setup shared by the two
``visualize_noise_to_data*`` scripts.
"""

from pathlib import Path
from typing import Tuple

import matplotlib.image as mpimg
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from scipy.stats import gaussian_kde


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "slides" / "public" / "figures"
IMAGES_DIR = PROJECT_ROOT / "slides" / "public" / "images"
CAT_IMAGE = IMAGES_DIR / "cat.png"

BLUE = "#1f77b4"
YELLOW = "#f4c20d"
GREEN = "#2ca02c"
RED_TRACE = "#e74c3c"
GRAY_CONTOUR = "#7f7f7f"


def time_text(ax: Axes, x: float = 0.02, y: float = 0.96):
    """Top-left rounded white badge used to print 't = ...' or 'step n=...'."""
    return ax.text(
        x, y, "", transform=ax.transAxes, fontsize=11,
        bbox={"boxstyle": "round", "facecolor": "white",
              "alpha": 0.9, "edgecolor": "gray"},
    )


def save_anim(anim: FuncAnimation, out_dir: Path, name: str, fps: int = 30) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.gif"
    anim.save(out, writer=PillowWriter(fps=fps))
    print(f"saved {out}")
    return out


def save_png(fig, out_dir: Path, name: str, dpi: int = 160) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"saved {out}")
    return out


def make_cmap(name: str, base: str) -> LinearSegmentedColormap:
    """Linear colormap from transparent white to `base` colour."""
    return LinearSegmentedColormap.from_list(name, [(1, 1, 1, 0), base], N=256)


# ---- Noise-to-data shared setup -----------------------------------------

LEFT_CENTER = np.array([-3.2, 0.0])
RIGHT_CENTER = np.array([3.0, 0.0])


def sample_gaussian(n: int, sigma: float = 0.55, seed: int = 42,
                    center: np.ndarray = LEFT_CENTER) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, size=(n, 2)) + center


def sample_two_moons(n: int, noise: float = 0.06, seed: int = 7,
                     center: np.ndarray = RIGHT_CENTER) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_a = n // 2
    n_b = n - n_a

    theta_a = rng.uniform(0.0, np.pi, n_a)
    moon_a = np.stack([np.cos(theta_a), np.sin(theta_a)], axis=1)
    moon_a += noise * rng.standard_normal(moon_a.shape) + np.array([-0.35, 0.15])

    theta_b = rng.uniform(np.pi, 2.0 * np.pi, n_b)
    moon_b = np.stack([np.cos(theta_b), np.sin(theta_b)], axis=1)
    moon_b += noise * rng.standard_normal(moon_b.shape) + np.array([0.35, -0.15])

    return 1.1 * np.concatenate([moon_a, moon_b], axis=0) + center


def draw_gaussian_contour(ax: Axes, sigma: float = 0.55,
                          center: np.ndarray = LEFT_CENTER,
                          color: str = BLUE) -> None:
    xs = np.linspace(center[0] - 2.2, center[0] + 2.2, 180)
    ys = np.linspace(-2.2, 2.2, 180)
    GX, GY = np.meshgrid(xs, ys)
    pdf = np.exp(-((GX - center[0]) ** 2 + (GY - center[1]) ** 2) / (2.0 * sigma ** 2))
    cmap = make_cmap("noise_cmap", color)
    ax.contourf(GX, GY, pdf, levels=10, cmap=cmap, alpha=0.85, zorder=1)
    ax.contour(GX, GY, pdf, levels=5, colors=color,
               linewidths=0.6, alpha=0.55, zorder=2)


def draw_kde_contour(ax: Axes, samples: np.ndarray,
                     center: np.ndarray = RIGHT_CENTER,
                     color: str = GREEN, bw_method: float = 0.15) -> None:
    xs = np.linspace(center[0] - 2.2, center[0] + 2.2, 200)
    ys = np.linspace(-2.0, 2.0, 200)
    GX, GY = np.meshgrid(xs, ys)
    kde = gaussian_kde(samples.T, bw_method=bw_method)
    Z = kde(np.stack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    cmap = make_cmap("data_cmap", color)
    ax.contourf(GX, GY, Z, levels=10, cmap=cmap, alpha=0.85, zorder=1)
    ax.contour(GX, GY, Z, levels=5, colors=color,
               linewidths=0.6, alpha=0.55, zorder=2)


def cat_callout(ax: Axes, anchor: np.ndarray, xybox: Tuple[float, float],
                image_path: Path = CAT_IMAGE, zoom: float = 0.42,
                stride: int = 18) -> AnnotationBbox:
    """Place a downsampled cat image with a curved arrow pointing at `anchor`."""
    img = mpimg.imread(str(image_path))[::stride, ::stride]
    box = AnnotationBbox(
        OffsetImage(img, zoom=zoom),
        anchor, xybox=xybox,
        xycoords="data", boxcoords="data",
        arrowprops={"arrowstyle": "-|>",
                    "connectionstyle": "arc3,rad=-0.25",
                    "color": "#333333", "lw": 1.3, "mutation_scale": 14,
                    "shrinkA": 4, "shrinkB": 6},
        frameon=True, pad=0.3,
        bboxprops={"edgecolor": "#333333", "linewidth": 1.1,
                   "boxstyle": "round,pad=0.25"},
        zorder=20,
    )
    ax.add_artist(box)
    return box


def label_endpoint_distributions(ax: Axes, y: float,
                                 left_label: str = r"$p_{\mathrm{init}} = \mathcal{N}(0, I)$",
                                 right_label: str = r"$p_{\mathrm{data}}$ (cat images)",
                                 ) -> None:
    """The two captions under the noise (left) and data (right) clouds."""
    ax.text(LEFT_CENTER[0], y, left_label,
            ha="center", va="bottom", fontsize=12, color=BLUE, zorder=5)
    ax.text(RIGHT_CENTER[0], y, right_label,
            ha="center", va="bottom", fontsize=12, color=GREEN, zorder=5)


def cleared_canvas(ax: Axes, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> None:
    """Reset axis: no ticks, no grid, no spines. Used for "scene" plots."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
