"""Upgrade the notebooks to use the marginal-preserving rectified-flow SDE
(score-corrected drift) and the corresponding closed-form KL with the
time-dependent C_t factor.

Background
----------
The previous notebook implementation used the simplified SDE

    X_{t+h} = X_t + h * v(X_t, t) + sigma * sqrt(h) * eps

with the velocity field v as the drift. That is the C_t = 1 special case of
the marginal-preserving rectified-flow SDE from Appendix A of the slides:

    b_t(x) = v(x, t) + (sigma^2 / (2 (1 - t))) * (t * v(x, t) - x)

The corresponding closed-form per-step KL has a time-dependent weight:

    KL_step = h * C_t^2 / (2 sigma^2) * || v_theta - v_ref ||^2,
    C_t     = 1 + t * sigma^2 / (2 (1 - t)).

This script rewrites cells 14 and 19 of both notebooks to match the slide
math, and leaves cell 20 (grpo_train) untouched since it just calls
kl_to_ref_gaussian.
"""

from __future__ import annotations

import json
from pathlib import Path


NEW_CELL_14 = '''SIGMA = 0.1     # noise level for the SDE
K_STEPS = 50    # integration steps


def score_corrected_drift(v: torch.Tensor, x: torch.Tensor, t_scalar: float, sigma: float) -> torch.Tensor:
    """Marginal-preserving drift for rectified flow.

        b_t(x) = v + (sigma^2 / (2 (1 - t))) * (t * v - x).

    The (t*v - x) factor is the rectified-flow score expressed in the
    learned velocity (see Appendix A of the slides). With this drift the
    SDE preserves the same marginals p_t as the underlying ODE.
    """
    coef = sigma ** 2 / (2.0 * (1.0 - t_scalar))
    return v + coef * (t_scalar * v - x)


def c_factor(t_scalar: float, sigma: float) -> float:
    """C_t = 1 + t * sigma^2 / (2 (1 - t))."""
    return 1.0 + t_scalar * sigma ** 2 / (2.0 * (1.0 - t_scalar))


def rollout(model, n: int, K: int = K_STEPS, sigma: float = SIGMA):
    """Roll out n stochastic trajectories with the marginal-preserving SDE.

    Returns (x_final, log_prob, trajectory).
    """
    dt = 1.0 / K
    std = sigma * math.sqrt(dt)
    x = torch.randn(n, 2, device=device)
    log_prob = torch.zeros(n, device=device)
    traj = [x.clone()]
    for k in range(K):
        t_scalar = k * dt
        t = torch.full((n,), t_scalar, device=device)
        v = model(x, t)
        b = score_corrected_drift(v, x, t_scalar, sigma)
        mean = x + b * dt
        x_next = mean + std * torch.randn_like(x)
        # log-prob of a 2D isotropic Gaussian
        lp = -0.5 * ((x_next - mean) ** 2).sum(-1) / (std ** 2) \\
             - 2 * math.log(std * math.sqrt(2 * math.pi))
        log_prob = log_prob + lp
        x = x_next
        traj.append(x.clone())
    return x, log_prob, traj


def logprob_of_trajectory(model, traj, K: int = K_STEPS, sigma: float = SIGMA):
    """Recompute log-prob of an *existing* trajectory under `model`.
    Used for the importance-sampling ratio.
    """
    dt = 1.0 / K
    std = sigma * math.sqrt(dt)
    log_prob = torch.zeros(traj[0].shape[0], device=device)
    for k in range(K):
        t_scalar = k * dt
        t = torch.full((traj[k].shape[0],), t_scalar, device=device)
        v = model(traj[k], t)
        b = score_corrected_drift(v, traj[k], t_scalar, sigma)
        mean = traj[k] + b * dt
        lp = -0.5 * ((traj[k + 1] - mean) ** 2).sum(-1) / (std ** 2) \\
             - 2 * math.log(std * math.sqrt(2 * math.pi))
        log_prob = log_prob + lp
    return log_prob


def kl_to_ref_gaussian(model, ref_model, traj, K: int = K_STEPS, sigma: float = SIGMA):
    """Closed-form KL(pi_theta || pi_ref) summed over the K Euler steps.

    Per-step transitions are Gaussian with the same covariance under both
    theta and ref. Subtracting the means cancels the score-correction's
    `-x` piece, leaving

        mu_theta - mu_ref = h * C_t * (v_theta - v_ref),

    so the per-step KL is

        KL_step = h * C_t^2 / (2 * sigma^2) * || v_theta - v_ref ||^2,

    with C_t = 1 + t * sigma^2 / (2 (1 - t)) varying step by step.

    Returns a tensor of shape (n,) with the trajectory-level KL.
    """
    dt = 1.0 / K
    kl = torch.zeros(traj[0].shape[0], device=device)
    for k in range(K):
        t_scalar = k * dt
        t = torch.full((traj[k].shape[0],), t_scalar, device=device)
        v_theta = model(traj[k], t)
        with torch.no_grad():
            v_ref = ref_model(traj[k], t)
        C_t = c_factor(t_scalar, sigma)
        kl = kl + 0.5 * dt * (C_t ** 2) / (sigma ** 2) * ((v_theta - v_ref) ** 2).sum(-1)
    return kl
'''


NEW_KL_MARKDOWN = '''### The KL estimator

Every Euler step of the marginal-preserving SDE has Gaussian transitions $\\pi(X_{t+h}\\mid X_t) = \\mathcal{N}(X_t + h\\, b_t(X_t),\\ \\sigma^2 h\\, I)$, with the **score-corrected drift**

$$b_t(x) \\;=\\; v(x, t) \\;+\\; \\frac{\\sigma^2}{2(1-t)}\\bigl(t\\, v(x, t) - x\\bigr).$$

The covariance is the same under $\\pi_\\theta$ and $\\pi_{\\mathrm{ref}}$, so the trajectory-level KL has a **closed form** — no Monte-Carlo estimator needed. The $-x$ piece of the score correction cancels in $\\mu_\\theta - \\mu_{\\mathrm{ref}}$, leaving a time-dependent weight $C_{t_k}^2$ on the squared velocity residual:

$$\\mathrm{KL}(\\pi_\\theta \\| \\pi_{\\mathrm{ref}}) \\;=\\; \\sum_{k=0}^{K-1} \\frac{h\\, C_{t_k}^2}{2\\sigma^2}\\,\\bigl\\| v_\\theta(X_{t_k}, t_k) - v_{\\mathrm{ref}}(X_{t_k}, t_k) \\bigr\\|^2,
\\qquad C_t \\;=\\; 1 + \\frac{t\\,\\sigma^2}{2(1-t)}.$$

This is what `kl_to_ref_gaussian` (defined above) computes. It is exact (not an estimator), deterministic given the trajectory, and lower-variance than the trajectory-level $k3$ estimator that we would otherwise need.'''


OLD_KL_MARKDOWN_MARKER = "### The KL estimator"


def src_str(cell) -> str:
    return ''.join(cell['source'])


def set_src(cell, text: str) -> None:
    cell['source'] = text.splitlines(keepends=True)


def patch_notebook(path: Path) -> None:
    nb = json.loads(path.read_text())

    # --- Cell 14: rewrite rollout/logprob/kl helpers to use score-corrected drift ---
    cell14 = nb['cells'][14]
    src14 = src_str(cell14)
    if 'score_corrected_drift' in src14:
        print("  cell 14: already patched")
    else:
        set_src(cell14, NEW_CELL_14)
        print("  cell 14: rewrote SDE helpers (rollout, logprob, KL) with score correction")

    # --- Cell 19: rewrite KL-estimator markdown to include C_t ---
    cell19 = nb['cells'][19]
    src19 = src_str(cell19)
    idx = src19.find(OLD_KL_MARKDOWN_MARKER)
    if idx == -1:
        print("  cell 19: '### The KL estimator' marker not found; skipping")
    elif 'C_{t_k}' in src19:
        print("  cell 19: already patched")
    else:
        set_src(cell19, src19[:idx] + NEW_KL_MARKDOWN)
        print("  cell 19: rewrote KL-estimator subsection with C_t factor")

    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    for name in ('flow_matching_grpo.ipynb', 'flow_matching_grpo_smiley.ipynb'):
        print(f"\n== {name} ==")
        patch_notebook(repo / name)


if __name__ == '__main__':
    main()
