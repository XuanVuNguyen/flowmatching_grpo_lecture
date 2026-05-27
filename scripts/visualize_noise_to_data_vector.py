r"""Step-by-step transport from Gaussian noise into a complex distribution.

Same layout as `visualize_noise_to_data.py` — Gaussian on the left, two-moons
on the right with a cat callout — but the trajectory is now revealed as an
Euler-style sequence: at each step a yellow velocity arrow is drawn at the
current location, then the particle moves along that arrow to the next
waypoint. Repeats until the cat point is reached.
"""

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from scipy.stats import gaussian_kde


N_SAMPLES = 1500
N_STEPS = 10
PAUSE_FRAMES = 18
MOVE_FRAMES = 14
STEP_FRAMES = PAUSE_FRAMES + MOVE_FRAMES
HOLD_FRAMES = 30
N_FRAMES = N_STEPS * STEP_FRAMES + HOLD_FRAMES

X_LIM = (-6.2, 6.2)
Y_LIM = (-2.6, 3.4)

LEFT_CENTER = np.array([-3.2, 0.0])
RIGHT_CENTER = np.array([3.0, 0.0])

NOISE_COLOR = "#1f77b4"
DATA_COLOR = "#2ca02c"
TRACE_COLOR = "#e74c3c"
ARROW_COLOR = "#f4c20d"


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
    ax.contour(GX, GY, pdf, levels=5, colors=NOISE_COLOR,
               linewidths=0.6, alpha=0.55, zorder=2)


def draw_data_contour(ax, samples: np.ndarray):
    xs = np.linspace(RIGHT_CENTER[0] - 2.2, RIGHT_CENTER[0] + 2.2, 200)
    ys = np.linspace(-2.0, 2.0, 200)
    GX, GY = np.meshgrid(xs, ys)
    kde = gaussian_kde(samples.T, bw_method=0.15)
    Z = kde(np.stack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    cmap = make_cmap("data_cmap", DATA_COLOR)
    ax.contourf(GX, GY, Z, levels=10, cmap=cmap, alpha=0.85, zorder=1)
    ax.contour(GX, GY, Z, levels=5, colors=DATA_COLOR,
               linewidths=0.6, alpha=0.55, zorder=2)


def make_waypoints(p0: np.ndarray, p1: np.ndarray, n_steps: int) -> np.ndarray:
    """Sample n_steps + 1 waypoints along a smooth curved path p0 -> p1."""
    s = np.linspace(0.0, 1.0, n_steps + 1)
    eased = 0.5 - 0.5 * np.cos(np.pi * s)
    line = (1.0 - eased)[:, None] * p0 + eased[:, None] * p1
    bump = 0.7 * np.sin(np.pi * s)
    line[:, 1] += bump
    return line


def main(out_dir: Path, cat_image_path: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    x_init = sample_gaussian(N_SAMPLES)
    x_data = sample_two_moons(N_SAMPLES)

    anchor_point = RIGHT_CENTER + np.array([-0.55, 1.05])

    rng2 = np.random.default_rng(123)
    p_start = x_init[int(rng2.integers(0, len(x_init)))]
    waypoints = make_waypoints(p_start, anchor_point, N_STEPS)

    cat_img = mpimg.imread(str(cat_image_path))[::18, ::18]

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

    ax.text(LEFT_CENTER[0], Y_LIM[0] + 0.25,
            r"$p_{\mathrm{init}} = \mathcal{N}(0, I)$",
            ha="center", va="bottom", fontsize=12, color=NOISE_COLOR, zorder=5)
    ax.text(RIGHT_CENTER[0], Y_LIM[0] + 0.25,
            r"$p_{\mathrm{data}}$ (cat images)",
            ha="center", va="bottom", fontsize=12, color=DATA_COLOR, zorder=5)

    cat_offset = OffsetImage(cat_img, zoom=0.42)
    cat_ab = AnnotationBbox(
        cat_offset, anchor_point,
        xybox=(RIGHT_CENTER[0] + 1.9, 2.6),
        xycoords="data", boxcoords="data",
        arrowprops={
            "arrowstyle": "-|>",
            "connectionstyle": "arc3,rad=-0.25",
            "color": "#333333", "lw": 1.3, "mutation_scale": 14,
            "shrinkA": 4, "shrinkB": 6,
        },
        frameon=True, pad=0.3,
        bboxprops={"edgecolor": "#333333", "linewidth": 1.1, "boxstyle": "round,pad=0.25"},
        zorder=20,
    )
    ax.add_artist(cat_ab)

    ax.plot([anchor_point[0]], [anchor_point[1]],
            marker="o", ms=7, mfc=DATA_COLOR, mec="#1c5b1c", mew=0.9, zorder=6)

    # start marker
    ax.plot([p_start[0]], [p_start[1]],
            marker="s", ms=8, mfc=TRACE_COLOR, mec="black", mew=0.6, zorder=4)

    (trail,) = ax.plot([], [], color=TRACE_COLOR, lw=1.8, alpha=0.9, zorder=4)
    (head,) = ax.plot([], [], marker="o", ms=10, mfc=TRACE_COLOR,
                      mec="black", mew=0.7, zorder=5)

    arrow = ax.quiver(
        [p_start[0]], [p_start[1]], [0.0], [0.0],
        color=ARROW_COLOR, scale=1.0, scale_units="xy", angles="xy",
        width=0.007, headwidth=4.0, headlength=5.0, headaxislength=4.5,
        zorder=6, edgecolor="black", linewidth=0.4,
    )

    step_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, fontsize=11,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "gray"},
    )

    def update(frame: int):
        if frame >= N_STEPS * STEP_FRAMES:
            # final hold: particle at anchor, no arrow
            head.set_data([anchor_point[0]], [anchor_point[1]])
            trail.set_data(waypoints[:, 0], waypoints[:, 1])
            arrow.set_alpha(0.0)
            step_text.set_text(f"step $n={N_STEPS}$  ·  arrived")
            return [trail, head, arrow, step_text]

        step_idx = frame // STEP_FRAMES
        phase_frame = frame - step_idx * STEP_FRAMES

        p_a = waypoints[step_idx]
        p_b = waypoints[step_idx + 1]
        dv = p_b - p_a

        if phase_frame < PAUSE_FRAMES:
            now = p_a
            arrow_visible = True
        else:
            alpha = (phase_frame - PAUSE_FRAMES) / MOVE_FRAMES
            alpha = 0.5 - 0.5 * np.cos(np.pi * alpha)
            now = p_a + alpha * dv
            arrow_visible = True

        # trail: completed waypoints + current position
        if step_idx == 0:
            trail_pts = np.array([p_start, now])
        else:
            trail_pts = np.vstack([waypoints[: step_idx + 1], now[None, :]])
        trail.set_data(trail_pts[:, 0], trail_pts[:, 1])
        head.set_data([now[0]], [now[1]])

        if arrow_visible:
            arrow.set_offsets(np.array([[p_a[0], p_a[1]]]))
            arrow.set_UVC(np.array([dv[0]]), np.array([dv[1]]))
            arrow.set_alpha(1.0)
        else:
            arrow.set_alpha(0.0)

        step_text.set_text(f"step $n={step_idx}$  ·  $u_{{t_n}}(X_n)$ shown in yellow")
        return [trail, head, arrow, step_text]

    anim = FuncAnimation(
        fig, update, frames=N_FRAMES, interval=33, blit=False, repeat=True,
    )

    fig.tight_layout()
    out_gif = out_dir / "noise_to_data_vector.gif"
    anim.save(out_gif, writer=PillowWriter(fps=30))
    print(f"saved {out_gif}")

    # static snapshot: mid-trajectory with arrow visible
    update(4 * STEP_FRAMES + PAUSE_FRAMES // 2)
    out_png = out_dir / "noise_to_data_vector.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"saved {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    main(
        root / "slides" / "public" / "figures",
        root / "slides" / "public" / "images" / "cat.png",
    )
