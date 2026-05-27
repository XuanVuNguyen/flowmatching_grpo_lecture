import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from _common import BLUE, FIGURES_DIR, YELLOW, save_anim, save_png, time_text


T_FINAL = 1.0
N_STEPS = 8
PAUSE_FRAMES = 22
MOVE_FRAMES = 26
STEP_FRAMES = PAUSE_FRAMES + MOVE_FRAMES
N_FRAMES = N_STEPS * STEP_FRAMES + PAUSE_FRAMES

T_LIM = (0.0, 1.0)
X_LIM = (-0.4, 2.4)
X0 = 0.5


def velocity(t: float, x: float | np.ndarray) -> np.ndarray:
    """1D, time-dependent right-hand side of the ODE."""
    return 4.0 * np.sin(2.0 * np.pi * t) - 0.4 * np.asarray(x)


def euler_steps(x0: float, n_steps: int, t_final: float):
    dt = t_final / n_steps
    xs = [x0]
    ts = [0.0]
    vs = []
    for _ in range(n_steps):
        v_n = float(velocity(ts[-1], xs[-1]))
        vs.append(v_n)
        xs.append(xs[-1] + dt * v_n)
        ts.append(ts[-1] + dt)
    return np.array(xs), np.array(ts), np.array(vs), dt


def main():
    xs, ts, vs, _ = euler_steps(X0, N_STEPS, T_FINAL)

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(*T_LIM)
    ax.set_ylim(*X_LIM)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$X$")
    ax.set_title("Euler simulation")
    ax.axhline(0, color="gray", lw=0.4, alpha=0.5)
    ax.grid(alpha=0.15)

    ax.plot(ts[0], xs[0], color=BLUE, marker="s", ms=8, mec="black", mew=0.7, zorder=3)
    ax.scatter(ts[1:], xs[1:], color=BLUE, s=18, alpha=0.25, zorder=2)

    (trail,) = ax.plot([], [], color=BLUE, lw=2.6, zorder=3)
    (head,) = ax.plot([], [], color=BLUE, marker="o", ms=11, mec="black", mew=0.8, zorder=4)
    arrow = ax.quiver(
        [ts[0]], [xs[0]], [0.0], [0.0],
        color=YELLOW, scale=1.0, scale_units="xy", angles="xy",
        width=0.007, headwidth=3.5, headlength=4.5, headaxislength=4.0,
        zorder=5, edgecolor="black", linewidth=0.4,
    )
    step_text = time_text(ax, y=0.94)

    def update(frame: int):
        step_idx = min(frame // STEP_FRAMES, N_STEPS - 1)
        phase_frame = frame - step_idx * STEP_FRAMES

        t_start, x_start = ts[step_idx], xs[step_idx]
        t_end, x_end = ts[step_idx + 1], xs[step_idx + 1]
        d_t, d_x = t_end - t_start, x_end - x_start

        if phase_frame < PAUSE_FRAMES:
            t_now, x_now = t_start, x_start
            arrow_visible = True
        elif phase_frame < STEP_FRAMES:
            alpha = (phase_frame - PAUSE_FRAMES) / MOVE_FRAMES
            alpha = 0.5 - 0.5 * np.cos(np.pi * alpha)
            t_now = t_start + alpha * d_t
            x_now = x_start + alpha * d_x
            arrow_visible = True
        else:
            t_now, x_now = t_end, x_end
            arrow_visible = False

        trail_t = np.concatenate([ts[: step_idx + 1], [t_now]])
        trail_x = np.concatenate([xs[: step_idx + 1], [x_now]])
        trail.set_data(trail_t, trail_x)
        head.set_data([t_now], [x_now])

        if arrow_visible:
            arrow.set_offsets(np.array([[t_start, x_start]]))
            arrow.set_UVC(np.array([d_t]), np.array([d_x]))
            arrow.set_alpha(1.0)
        else:
            arrow.set_alpha(0.0)

        step_text.set_text(
            f"step $n={step_idx}$,  $t_n={ts[step_idx]:.2f}$,  "
            f"$u_{{t_n}}(X_n)={vs[step_idx]:+.2f}$"
        )
        return trail, head, arrow, step_text

    anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=33, blit=False, repeat=True)
    fig.tight_layout()
    save_anim(anim, FIGURES_DIR, "euler_sampling")

    update(PAUSE_FRAMES // 2 + 2 * STEP_FRAMES)
    save_png(fig, FIGURES_DIR, "euler_sampling")


if __name__ == "__main__":
    main()
