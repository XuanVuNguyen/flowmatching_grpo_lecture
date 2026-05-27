r"""Animate the transport of Gaussian noise into a complex 2D distribution.

The plane is split: the **left** half shows $p_{\text{init}}$, an isotropic
Gaussian rendered with density contours, and the **right** half shows
$p_{\text{data}}$, a "two-moons" distribution rendered via KDE contours.
A cat photo is anchored as a callout to one sample of $p_{\text{data}}$.
A single particle (red) animates the trajectory from noise to data.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from _common import (
    FIGURES_DIR,
    LEFT_CENTER,
    RED_TRACE,
    RIGHT_CENTER,
    cat_callout,
    cleared_canvas,
    draw_gaussian_contour,
    draw_kde_contour,
    label_endpoint_distributions,
    sample_gaussian,
    sample_two_moons,
    save_anim,
    save_png,
    time_text,
)


T_FINAL = 1.0
N_FRAMES = 140
N_SAMPLES = 1500

X_LIM = (-6.2, 6.2)
Y_LIM = (-2.6, 3.4)


def make_trajectory(p0: np.ndarray, p1: np.ndarray, n: int) -> np.ndarray:
    """Smooth curved trajectory from p0 to p1 with a gentle vertical detour."""
    s = np.linspace(0.0, 1.0, n)
    eased = 0.5 - 0.5 * np.cos(np.pi * s)
    line = (1.0 - eased)[:, None] * p0 + eased[:, None] * p1
    bump = 0.55 * np.sin(np.pi * s)
    line[:, 1] += bump
    return line


def main():
    x_init = sample_gaussian(N_SAMPLES)
    x_data = sample_two_moons(N_SAMPLES)

    anchor_point = RIGHT_CENTER + np.array([-0.55, 1.05])

    rng2 = np.random.default_rng(123)
    p_start = x_init[int(rng2.integers(0, len(x_init)))]
    p_end = anchor_point
    path = make_trajectory(p_start, p_end, N_FRAMES)

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    cleared_canvas(ax, X_LIM, Y_LIM)

    draw_gaussian_contour(ax)
    draw_kde_contour(ax, x_data)
    label_endpoint_distributions(ax, Y_LIM[0] + 0.25)
    cat_callout(ax, anchor_point, xybox=(RIGHT_CENTER[0] + 1.9, 2.6))

    # marker on the anchor point so the arrow has something concrete to point at
    ax.plot([anchor_point[0]], [anchor_point[1]],
            marker="o", ms=7, mfc="#2ca02c", mec="#1c5b1c", mew=0.9, zorder=6)

    (trail,) = ax.plot([], [], color=RED_TRACE, lw=1.8, alpha=0.9, zorder=4)
    (head,) = ax.plot([], [], marker="o", ms=10,
                      mfc=RED_TRACE, mec="black", mew=0.7, zorder=5)
    ax.plot([p_start[0]], [p_start[1]], marker="s", ms=8,
            mfc=RED_TRACE, mec="black", mew=0.6, zorder=4)

    t_text = time_text(ax, y=0.95)

    def update(frame: int):
        trail.set_data(path[: frame + 1, 0], path[: frame + 1, 1])
        head.set_data([path[frame, 0]], [path[frame, 1]])
        t_now = frame / (N_FRAMES - 1)
        t_text.set_text(f"$t = {t_now:.2f}$")
        return [trail, head, t_text]

    anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=33, blit=False, repeat=True)
    fig.tight_layout()
    save_anim(anim, FIGURES_DIR, "noise_to_data")

    update(N_FRAMES - 1)
    save_png(fig, FIGURES_DIR, "noise_to_data")
    plt.close(fig)


if __name__ == "__main__":
    main()
