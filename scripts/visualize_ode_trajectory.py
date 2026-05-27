import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

from _common import BLUE, FIGURES_DIR, save_anim, save_png, time_text


T_FINAL = 1.0
N_FRAMES = 240
T_LIM = (0.0, 1.0)
X_LIM = (-0.4, 2.4)
X0 = 0.5


def velocity(t: float, x: float | np.ndarray) -> np.ndarray:
    """1D, time-dependent right-hand side of the ODE."""
    return 4.0 * np.sin(2.0 * np.pi * t) - 0.4 * np.asarray(x)


def integrate(x0: float, t_eval: np.ndarray) -> np.ndarray:
    sol = solve_ivp(
        lambda t, x: [velocity(t, x[0])],
        (t_eval[0], t_eval[-1]),
        [x0],
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
    )
    return sol.y[0]


def main():
    t = np.linspace(0.0, T_FINAL, N_FRAMES)
    xs = integrate(X0, t)

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(*T_LIM)
    ax.set_ylim(*X_LIM)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$X$")
    ax.set_title("Trajectory")
    ax.axhline(0, color="gray", lw=0.4, alpha=0.5)
    ax.grid(alpha=0.15)

    ax.plot(t, xs, color=BLUE, lw=1.0, alpha=0.18)
    ax.plot(t[0], xs[0], color=BLUE, marker="s", ms=8, mec="black", mew=0.7)

    (trail,) = ax.plot([], [], color=BLUE, lw=2.6)
    (head,) = ax.plot([], [], color=BLUE, marker="o", ms=11, mec="black", mew=0.8)
    t_text = time_text(ax, y=0.94)

    def init():
        trail.set_data([], [])
        head.set_data([], [])
        t_text.set_text("")
        return trail, head, t_text

    def update(frame: int):
        trail.set_data(t[: frame + 1], xs[: frame + 1])
        head.set_data([t[frame]], [xs[frame]])
        t_text.set_text(f"$t = {t[frame]:.2f}$")
        return trail, head, t_text

    anim = FuncAnimation(
        fig, update, frames=N_FRAMES, init_func=init, interval=33, blit=True, repeat=True,
    )
    fig.tight_layout()
    save_anim(anim, FIGURES_DIR, "ode_trajectory")

    update(N_FRAMES - 1)
    save_png(fig, FIGURES_DIR, "ode_trajectory")


if __name__ == "__main__":
    main()
