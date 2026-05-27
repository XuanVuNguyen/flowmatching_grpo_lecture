"""Render a 2x2 GIF comparing an ODE rollout to SDE rollouts at three noise scales.

All four panels share the same sinusoidal drift u(t) = A * cos(2 pi t) and the
same starting point X_0 = 0. The top-left panel is the deterministic ODE
(sigma = 0); the other three are SDE rollouts with increasing sigma, each
showing several trajectories so the diversity injected by the per-step noise
is visible.

Output: slides/public/figures/ode_vs_sde.gif
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from _common import FIGURES_DIR, save_anim


T = 1.0
N_STEPS = 240
H = T / N_STEPS
T_GRID = np.linspace(0.0, T, N_STEPS + 1)

DRIFT_AMP = 12.0  # amplitude of the velocity field u(t)
X0 = 0.0
N_TRAJ = 4
SIGMAS = (0.4, 0.9, 1.6)
SEED = 7


def drift(t: np.ndarray | float) -> np.ndarray | float:
    # u(t) = A cos(2 pi t) so the analytical ODE solution is
    # X(t) = (A / 2 pi) sin(2 pi t), a sinusoid of amplitude ~A/(2 pi).
    return DRIFT_AMP * np.cos(2.0 * np.pi * t)


def simulate(sigma: float, n_traj: int, seed: int) -> np.ndarray:
    """Euler / Euler-Maruyama. Returns array of shape (n_traj, N_STEPS + 1)."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n_traj, N_STEPS + 1))
    x[:, 0] = X0
    for i in range(N_STEPS):
        dW = rng.standard_normal(n_traj) if sigma > 0 else 0.0
        x[:, i + 1] = x[:, i] + H * drift(T_GRID[i]) + sigma * np.sqrt(H) * dW
    return x


def main() -> None:
    ode = simulate(sigma=0.0, n_traj=1, seed=SEED)
    sdes = [simulate(sigma=s, n_traj=N_TRAJ, seed=SEED + k + 1)
            for k, s in enumerate(SIGMAS)]

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.6), sharex=True, sharey=True)

    panels = [
        (axes[0, 0], "ODE  (σ = 0)",          ode,     "#1f77b4"),
        (axes[0, 1], f"SDE  (σ = {SIGMAS[0]})", sdes[0], "#ff7f0e"),
        (axes[1, 0], f"SDE  (σ = {SIGMAS[1]})", sdes[1], "#2ca02c"),
        (axes[1, 1], f"SDE  (σ = {SIGMAS[2]})", sdes[2], "#d62728"),
    ]

    all_y = np.concatenate([ode.ravel()] + [s.ravel() for s in sdes])
    y_abs = float(np.max(np.abs(all_y))) + 0.5
    y_min, y_max = -y_abs, y_abs

    line_groups: list[list] = []
    for ax, title, traj, color in panels:
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0.0, T)
        ax.set_ylim(y_min, y_max)
        ax.grid(alpha=0.25)
        ax.set_xlabel("t")
        ax.set_ylabel(r"$X_t$")
        if traj is not ode:
            ax.plot(T_GRID, ode[0], color="#888888", lw=1.0,
                    ls="--", alpha=0.6, label="ODE")
        sub = []
        for _ in range(traj.shape[0]):
            (ln,) = ax.plot([], [], color=color, alpha=0.85, lw=1.6)
            sub.append(ln)
        line_groups.append(sub)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    # Downsample frames so the GIF is small but smooth.
    frame_step = 3
    frame_indices = list(range(0, N_STEPS + 1, frame_step))
    if frame_indices[-1] != N_STEPS:
        frame_indices.append(N_STEPS)

    def init():
        for sub in line_groups:
            for ln in sub:
                ln.set_data([], [])
        return [ln for sub in line_groups for ln in sub]

    def update(i: int):
        for (_, _, traj, _), sub in zip(panels, line_groups):
            for j, ln in enumerate(sub):
                ln.set_data(T_GRID[: i + 1], traj[j, : i + 1])
        return [ln for sub in line_groups for ln in sub]

    anim = FuncAnimation(fig, update, frames=frame_indices,
                         init_func=init, blit=True, interval=50)
    save_anim(anim, FIGURES_DIR, "ode_vs_sde", fps=24)


if __name__ == "__main__":
    main()
