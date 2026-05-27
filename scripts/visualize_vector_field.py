"""Animate a 2D time-evolving vector field u_t(x).

For each (x, t) the field returns a velocity in R^2; the animation sweeps t in
[0, 1] and redraws the quiver, illustrating that u_t is defined at every point
in space and changes with time.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from _common import BLUE, FIGURES_DIR, save_anim, save_png, time_text


T_FINAL = 1.0
N_FRAMES = 90

GRID_LIM = 1.6
GRID_N = 16

ARROW_LEN = 0.16

TRACER_STARTS = np.array(
    [
        [-1.2, -1.0],
        [1.0, -1.2],
        [-0.8, 1.1],
        [1.1, 1.0],
        [0.0, -0.4],
    ]
)


def velocity(t, x, y):
    """2D, time-dependent velocity field."""
    omega = 1.4 - 0.6 * t
    ux = -omega * y + 0.7 * np.sin(np.pi * t + 0.5 * x)
    uy = omega * x + 0.7 * np.cos(np.pi * t + 0.5 * y)
    return ux, uy


def integrate_tracers(starts, n_frames, t_final):
    dt = t_final / max(n_frames - 1, 1)
    pts = starts.copy().astype(float)
    history = [pts.copy()]
    for k in range(1, n_frames):
        t = (k - 1) * dt
        ux, uy = velocity(t, pts[:, 0], pts[:, 1])
        pts[:, 0] += dt * ux
        pts[:, 1] += dt * uy
        pts = np.clip(pts, -GRID_LIM * 1.05, GRID_LIM * 1.05)
        history.append(pts.copy())
    return np.stack(history, axis=0)


def main():
    xs = np.linspace(-GRID_LIM, GRID_LIM, GRID_N)
    ys = np.linspace(-GRID_LIM, GRID_LIM, GRID_N)
    X, Y = np.meshgrid(xs, ys)

    tracers = integrate_tracers(TRACER_STARTS, N_FRAMES, T_FINAL)

    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    ax.set_xlim(-GRID_LIM * 1.05, GRID_LIM * 1.05)
    ax.set_ylim(-GRID_LIM * 1.05, GRID_LIM * 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.grid(alpha=0.15)

    ux0, uy0 = velocity(0.0, X, Y)
    norm0 = np.sqrt(ux0 ** 2 + uy0 ** 2) + 1e-9
    Q = ax.quiver(
        X, Y, ux0 / norm0 * ARROW_LEN, uy0 / norm0 * ARROW_LEN, norm0,
        cmap="viridis", angles="xy", scale_units="xy", scale=1.0,
        width=0.005, headwidth=4.0, headlength=4.5, headaxislength=4.0,
        pivot="middle", edgecolor="black", linewidth=0.25, alpha=0.85,
    )

    trails = [ax.plot([], [], color=BLUE, lw=1.6, alpha=0.6, zorder=3)[0]
              for _ in range(len(TRACER_STARTS))]
    heads = ax.scatter(
        tracers[0, :, 0], tracers[0, :, 1],
        color=BLUE, s=42, edgecolor="black", linewidth=0.6, zorder=4,
    )

    t_text = time_text(ax, y=0.97)

    def update(frame: int):
        t = frame / max(N_FRAMES - 1, 1) * T_FINAL
        ux, uy = velocity(t, X, Y)
        mag = np.sqrt(ux ** 2 + uy ** 2) + 1e-9
        Q.set_UVC(ux / mag * ARROW_LEN, uy / mag * ARROW_LEN, mag)
        for i, trail in enumerate(trails):
            trail.set_data(tracers[: frame + 1, i, 0], tracers[: frame + 1, i, 1])
        heads.set_offsets(tracers[frame])
        t_text.set_text(rf"$t = {t:.2f}$")
        return (Q, heads, t_text, *trails)

    anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=50, blit=False, repeat=True)
    fig.tight_layout()
    save_anim(anim, FIGURES_DIR, "vector_field", fps=20)

    update(N_FRAMES // 3)
    save_png(fig, FIGURES_DIR, "vector_field")


if __name__ == "__main__":
    main()
