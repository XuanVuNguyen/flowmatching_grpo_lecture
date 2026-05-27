r"""Animate the flow $\Psi_t : \mathbb{R}^2 \to \mathbb{R}^2$.

A regular square grid (blue) is transported by a velocity field $u_t$
(yellow arrows). The grid warps smoothly as time advances. Also produces a
3-snapshot static figure at $t = 0,\ 0.5,\ 1.0$.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

from _common import BLUE, FIGURES_DIR, YELLOW, save_anim, save_png, time_text


T_FINAL = 1.0
N_FRAMES = 120
GRID_N = 11
GRID_RANGE = (-1.0, 1.0)
ARROW_N = 13
PLOT_RANGE = (-1.8, 1.8)


def velocity(t: float, x: np.ndarray) -> np.ndarray:
    """2D, mildly time-dependent velocity field. x has shape (2, ...)."""
    swirl = 1.0 - 0.25 * np.cos(2.0 * np.pi * t)
    vx = 0.4 * np.sin(np.pi * x[1]) * swirl
    vy = 0.4 * np.sin(np.pi * x[0]) * swirl
    return np.stack([vx, vy])


def integrate_grid(t_eval: np.ndarray):
    xs0 = np.linspace(*GRID_RANGE, GRID_N)
    ys0 = np.linspace(*GRID_RANGE, GRID_N)
    X0, Y0 = np.meshgrid(xs0, ys0)
    initial = np.stack([X0.flatten(), Y0.flatten()], axis=0)
    M = initial.shape[1]

    def rhs(t_val, x_flat):
        return velocity(t_val, x_flat.reshape(2, M)).flatten()

    sol = solve_ivp(
        rhs, (t_eval[0], t_eval[-1]), initial.flatten(),
        t_eval=t_eval, rtol=1e-7, atol=1e-9,
    )
    return sol.y.T.reshape(len(t_eval), 2, GRID_N, GRID_N)


def arrow_grid():
    xs = np.linspace(*PLOT_RANGE, ARROW_N)
    ys = np.linspace(*PLOT_RANGE, ARROW_N)
    AX, AY = np.meshgrid(xs, ys)
    return AX, AY


def draw_arrows(ax, AX, AY, arrow_pts, t_val):
    V = velocity(t_val, arrow_pts)
    return ax.quiver(
        AX, AY, V[0].reshape(AX.shape), V[1].reshape(AX.shape),
        color=YELLOW, scale=22, width=0.005,
        headwidth=3.5, headlength=4.5, headaxislength=4.0,
        edgecolor="black", linewidth=0.3, alpha=0.85, zorder=1,
    )


def main():
    t = np.linspace(0.0, T_FINAL, N_FRAMES)
    paths = integrate_grid(t)
    AX, AY = arrow_grid()
    arrow_pts = np.stack([AX.flatten(), AY.flatten()], axis=0)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(*PLOT_RANGE)
    ax.set_ylim(*PLOT_RANGE)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("Flow")
    ax.grid(alpha=0.12)

    quiv = draw_arrows(ax, AX, AY, arrow_pts, 0.0)
    h_lines = [ax.plot([], [], color=BLUE, lw=1.4, zorder=3)[0] for _ in range(GRID_N)]
    v_lines = [ax.plot([], [], color=BLUE, lw=1.4, zorder=3)[0] for _ in range(GRID_N)]
    t_text = time_text(ax)

    def update(frame: int):
        pos = paths[frame]
        for r, hl in enumerate(h_lines):
            hl.set_data(pos[0, r, :], pos[1, r, :])
        for c, vl in enumerate(v_lines):
            vl.set_data(pos[0, :, c], pos[1, :, c])
        V_t = velocity(t[frame], arrow_pts)
        quiv.set_UVC(V_t[0].reshape(AX.shape), V_t[1].reshape(AX.shape))
        t_text.set_text(f"$t = {t[frame]:.2f}$")
        return [*h_lines, *v_lines, quiv, t_text]

    anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=33, blit=False, repeat=True)
    fig.tight_layout()
    save_anim(anim, FIGURES_DIR, "flow")
    plt.close(fig)

    # 3-snapshot static figure: t = 0, 0.5, 1.0
    snap_times = np.array([0.0, 0.5, 1.0])
    snap_paths = integrate_grid(snap_times)

    fig2, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    for i, (t_snap, ax2) in enumerate(zip(snap_times, axes)):
        draw_arrows(ax2, AX, AY, arrow_pts, t_snap)
        pos = snap_paths[i]
        for r in range(GRID_N):
            ax2.plot(pos[0, r, :], pos[1, r, :], color=BLUE, lw=1.4, zorder=3)
        for c in range(GRID_N):
            ax2.plot(pos[0, :, c], pos[1, :, c], color=BLUE, lw=1.4, zorder=3)
        ax2.set_xlim(*PLOT_RANGE)
        ax2.set_ylim(*PLOT_RANGE)
        ax2.set_aspect("equal")
        ax2.set_title(f"$t = {t_snap:.1f}$")
        ax2.grid(alpha=0.12)
    fig2.suptitle("Flow")
    fig2.tight_layout()
    save_png(fig2, FIGURES_DIR, "flow")


if __name__ == "__main__":
    main()
