r"""Step-by-step transport from Gaussian noise into a complex distribution.

Same layout as ``visualize_noise_to_data.py`` — Gaussian on the left, two-moons
on the right with a cat callout — but the trajectory is now revealed as an
Euler-style sequence: at each step a yellow velocity arrow is drawn at the
current location, then the particle moves along that arrow to the next
waypoint. Repeats until the cat point is reached.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from _common import (
    FIGURES_DIR,
    RED_TRACE,
    RIGHT_CENTER,
    YELLOW,
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


N_SAMPLES = 1500
N_STEPS = 10
PAUSE_FRAMES = 18
MOVE_FRAMES = 14
STEP_FRAMES = PAUSE_FRAMES + MOVE_FRAMES
HOLD_FRAMES = 30
N_FRAMES = N_STEPS * STEP_FRAMES + HOLD_FRAMES

X_LIM = (-6.2, 6.2)
Y_LIM = (-2.6, 3.4)


def make_waypoints(p0: np.ndarray, p1: np.ndarray, n_steps: int) -> np.ndarray:
    """Sample n_steps + 1 waypoints along a smooth curved path p0 -> p1."""
    s = np.linspace(0.0, 1.0, n_steps + 1)
    eased = 0.5 - 0.5 * np.cos(np.pi * s)
    line = (1.0 - eased)[:, None] * p0 + eased[:, None] * p1
    bump = 0.7 * np.sin(np.pi * s)
    line[:, 1] += bump
    return line


def main():
    x_init = sample_gaussian(N_SAMPLES)
    x_data = sample_two_moons(N_SAMPLES)

    anchor_point = RIGHT_CENTER + np.array([-0.55, 1.05])

    rng2 = np.random.default_rng(123)
    p_start = x_init[int(rng2.integers(0, len(x_init)))]
    waypoints = make_waypoints(p_start, anchor_point, N_STEPS)

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    cleared_canvas(ax, X_LIM, Y_LIM)

    draw_gaussian_contour(ax)
    draw_kde_contour(ax, x_data)
    label_endpoint_distributions(ax, Y_LIM[0] + 0.25)
    cat_callout(ax, anchor_point, xybox=(RIGHT_CENTER[0] + 1.9, 2.6))

    ax.plot([anchor_point[0]], [anchor_point[1]],
            marker="o", ms=7, mfc="#2ca02c", mec="#1c5b1c", mew=0.9, zorder=6)
    ax.plot([p_start[0]], [p_start[1]], marker="s", ms=8,
            mfc=RED_TRACE, mec="black", mew=0.6, zorder=4)

    (trail,) = ax.plot([], [], color=RED_TRACE, lw=1.8, alpha=0.9, zorder=4)
    (head,) = ax.plot([], [], marker="o", ms=10,
                      mfc=RED_TRACE, mec="black", mew=0.7, zorder=5)
    arrow = ax.quiver(
        [p_start[0]], [p_start[1]], [0.0], [0.0],
        color=YELLOW, scale=1.0, scale_units="xy", angles="xy",
        width=0.007, headwidth=4.0, headlength=5.0, headaxislength=4.5,
        zorder=6, edgecolor="black", linewidth=0.4,
    )

    step_text = time_text(ax, y=0.95)

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
        else:
            alpha = (phase_frame - PAUSE_FRAMES) / MOVE_FRAMES
            alpha = 0.5 - 0.5 * np.cos(np.pi * alpha)
            now = p_a + alpha * dv

        if step_idx == 0:
            trail_pts = np.array([p_start, now])
        else:
            trail_pts = np.vstack([waypoints[: step_idx + 1], now[None, :]])
        trail.set_data(trail_pts[:, 0], trail_pts[:, 1])
        head.set_data([now[0]], [now[1]])

        arrow.set_offsets(np.array([[p_a[0], p_a[1]]]))
        arrow.set_UVC(np.array([dv[0]]), np.array([dv[1]]))
        arrow.set_alpha(1.0)

        step_text.set_text(f"step $n={step_idx}$  ·  $u_{{t_n}}(X_n)$ shown in yellow")
        return [trail, head, arrow, step_text]

    anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=33, blit=False, repeat=True)
    fig.tight_layout()
    save_anim(anim, FIGURES_DIR, "noise_to_data_vector")

    update(4 * STEP_FRAMES + PAUSE_FRAMES // 2)
    save_png(fig, FIGURES_DIR, "noise_to_data_vector")
    plt.close(fig)


if __name__ == "__main__":
    main()
