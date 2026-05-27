r"""Animate a probability path between $p_{\text{init}}$ and $p_{\text{data}}$.

$p_{\text{init}}$ is an isotropic Gaussian (drawn as light contours), and
$p_{\text{data}}$ is a smiley-shaped distribution. Samples are
linearly interpolated $X_t = (1-t)\,X_0 + t\,X_1$ so the marginal smoothly
morphs from the Gaussian to the smiley. One particular sample is drawn
as a small cat, following its own trajectory.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon

from _common import BLUE, FIGURES_DIR, GRAY_CONTOUR, save_anim, save_png, time_text


T_FINAL = 1.0
N_FRAMES = 120
N_SAMPLES = 900
X_LIM = (-2.5, 2.5)
Y_LIM = (-2.0, 2.0)

CAT_COLOR = "#f39c12"


def sample_gaussian(n: int, sigma: float = 0.55, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, size=(n, 2))


def sample_smiley(n: int, seed: int = 13) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_face = int(0.62 * n)
    n_eye = int(0.07 * n)
    n_smile = n - n_face - 2 * n_eye

    theta = rng.uniform(0.0, 2.0 * np.pi, n_face)
    r = 1.45 + 0.05 * rng.standard_normal(n_face)
    face = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

    eye_l = np.column_stack([
        -0.55 + 0.09 * rng.standard_normal(n_eye),
        0.55 + 0.09 * rng.standard_normal(n_eye),
    ])
    eye_r = np.column_stack([
        0.55 + 0.09 * rng.standard_normal(n_eye),
        0.55 + 0.09 * rng.standard_normal(n_eye),
    ])

    theta_s = rng.uniform(np.pi + 0.25, 2.0 * np.pi - 0.25, n_smile)
    r_s = 0.85 + 0.05 * rng.standard_normal(n_smile)
    smile = np.stack([r_s * np.cos(theta_s), -0.15 + r_s * np.sin(theta_s)], axis=1)

    return np.concatenate([face, eye_l, eye_r, smile], axis=0)


class Cat:
    """A small cat-face icon assembled from matplotlib patches."""

    def __init__(self, ax, size: float = 0.22, color: str = CAT_COLOR):
        self.size = size
        s = size
        self.head = Circle((0, 0), s * 0.85, fc=color, ec="black", lw=0.9, zorder=20)
        self.left_ear_rel = np.array([[-s * 0.85, s * 0.5], [-s * 0.4, s * 0.5], [-s * 0.6, s * 1.15]])
        self.right_ear_rel = np.array([[s * 0.85, s * 0.5], [s * 0.4, s * 0.5], [s * 0.6, s * 1.15]])
        self.left_ear = Polygon(self.left_ear_rel, fc=color, ec="black", lw=0.9, zorder=20)
        self.right_ear = Polygon(self.right_ear_rel, fc=color, ec="black", lw=0.9, zorder=20)
        self.left_eye = Circle((-s * 0.3, s * 0.18), s * 0.1, fc="black", zorder=21)
        self.right_eye = Circle((s * 0.3, s * 0.18), s * 0.1, fc="black", zorder=21)
        self.nose_rel = np.array([[-s * 0.13, -s * 0.08], [s * 0.13, -s * 0.08], [0.0, -s * 0.28]])
        self.nose = Polygon(self.nose_rel, fc="#ec7063", ec="black", lw=0.4, zorder=21)
        for p in self.artists():
            ax.add_patch(p)
        self.move_to(0.0, 0.0)

    def artists(self):
        return [self.head, self.left_ear, self.right_ear, self.left_eye, self.right_eye, self.nose]

    def move_to(self, x: float, y: float):
        s = self.size
        self.head.set_center((x, y))
        self.left_ear.set_xy(self.left_ear_rel + np.array([x, y]))
        self.right_ear.set_xy(self.right_ear_rel + np.array([x, y]))
        self.left_eye.set_center((x - s * 0.3, y + s * 0.18))
        self.right_eye.set_center((x + s * 0.3, y + s * 0.18))
        self.nose.set_xy(self.nose_rel + np.array([x, y]))


def gaussian_contour(ax, sigma: float = 0.55):
    xs = np.linspace(*X_LIM, 80)
    ys = np.linspace(*Y_LIM, 80)
    GX, GY = np.meshgrid(xs, ys)
    pdf = np.exp(-(GX ** 2 + GY ** 2) / (2.0 * sigma ** 2))
    return ax.contour(
        GX, GY, pdf,
        levels=[0.05, 0.2, 0.5, 0.85],
        colors=GRAY_CONTOUR, linewidths=0.7, alpha=0.45, zorder=1,
    )


def main():
    x_init = sample_gaussian(N_SAMPLES)
    x_data = sample_smiley(N_SAMPLES)

    # Match by polar angle for a smoother morph (random matching crosses too much).
    angle_i = np.arctan2(x_init[:, 1], x_init[:, 0])
    angle_d = np.arctan2(x_data[:, 1], x_data[:, 0])
    x_init = x_init[np.argsort(angle_i)]
    x_data = x_data[np.argsort(angle_d)]

    cat_x_init = np.array([0.25, -0.15])
    cat_x_data = np.array([0.55, 0.55])

    t = np.linspace(0.0, T_FINAL, N_FRAMES)

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("Probability path")
    ax.grid(alpha=0.12)

    gaussian_contour(ax)

    scatter = ax.scatter(x_init[:, 0], x_init[:, 1], s=9, c=BLUE, alpha=0.55, zorder=2)

    cat = Cat(ax, size=0.28)
    cat.move_to(*cat_x_init)

    t_text = time_text(ax)

    def update(frame: int):
        t_now = t[frame]
        x_t = (1.0 - t_now) * x_init + t_now * x_data
        scatter.set_offsets(x_t)
        cat_pos = (1.0 - t_now) * cat_x_init + t_now * cat_x_data
        cat.move_to(*cat_pos)
        t_text.set_text(f"$t = {t_now:.2f}$")
        return [scatter, t_text, *cat.artists()]

    anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=33, blit=False, repeat=True)
    fig.tight_layout()
    save_anim(anim, FIGURES_DIR, "probability_path")
    plt.close(fig)

    # Static 3-snapshot figure: t = 0, 0.5, 1.0
    snap_times = np.array([0.0, 0.5, 1.0])
    fig2, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for t_snap, ax2 in zip(snap_times, axes):
        gaussian_contour(ax2)
        x_t = (1.0 - t_snap) * x_init + t_snap * x_data
        ax2.scatter(x_t[:, 0], x_t[:, 1], s=9, c=BLUE, alpha=0.55, zorder=2)
        cat_snap = Cat(ax2, size=0.28)
        cat_pos = (1.0 - t_snap) * cat_x_init + t_snap * cat_x_data
        cat_snap.move_to(*cat_pos)
        ax2.set_xlim(*X_LIM)
        ax2.set_ylim(*Y_LIM)
        ax2.set_aspect("equal")
        ax2.set_title(f"$t = {t_snap:.1f}$")
        ax2.grid(alpha=0.12)
    fig2.suptitle("Probability path: $p_{\\text{init}} \\to p_{\\text{data}}$")
    fig2.tight_layout()
    save_png(fig2, FIGURES_DIR, "probability_path")


if __name__ == "__main__":
    main()
