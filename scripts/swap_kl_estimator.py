"""Swap the k3 KL estimator in the GRPO notebooks for the closed-form Gaussian KL.

Edits cells 14, 19, and 20 of both flow_matching_grpo.ipynb and
flow_matching_grpo_smiley.ipynb (which share identical content at those
positions). Run once; safe to re-run because each edit is a targeted
substring replace.
"""

from __future__ import annotations

import json
from pathlib import Path


NEW_KL_HELPER = '''


def kl_to_ref_gaussian(model, ref_model, traj, K: int = K_STEPS, sigma: float = SIGMA):
    """Closed-form KL(pi_theta || pi_ref) summed over the K Euler steps.

    Each per-step policy is an isotropic Gaussian
        pi(X_{t+h} | X_t) = N(X_t + h * v(X_t, t),  sigma^2 * h * I)
    with the *same* covariance under both theta and ref. So the per-step KL
    between two such Gaussians collapses to a scaled squared distance between
    the means:
        KL_step = (h / (2 * sigma^2)) * || v_theta(X_t, t) - v_ref(X_t, t) ||^2.

    Returns a tensor of shape (n,) with the trajectory-level KL.
    """
    dt = 1.0 / K
    kl = torch.zeros(traj[0].shape[0], device=device)
    for k in range(K):
        t = torch.full((traj[k].shape[0],), k * dt, device=device)
        v_theta = model(traj[k], t)
        with torch.no_grad():
            v_ref = ref_model(traj[k], t)
        kl = kl + 0.5 * dt / (sigma ** 2) * ((v_theta - v_ref) ** 2).sum(-1)
    return kl
'''


NEW_KL_MARKDOWN = '''### The KL estimator

Because every Euler step's policy is an isotropic Gaussian with the *same* fixed covariance $\\sigma^2 h\\, I$ under both $\\pi_\\theta$ and $\\pi_{\\mathrm{ref}}$ (only the mean depends on $\\theta$), the trajectory-level KL has a **closed form** — no Monte-Carlo estimator needed:

$$\\mathrm{KL}(\\pi_\\theta \\| \\pi_{\\mathrm{ref}}) \\;=\\; \\sum_{k=0}^{K-1} \\frac{h}{2\\sigma^2}\\,\\bigl\\| v_\\theta(X_{t_k}, t_k) - v_{\\mathrm{ref}}(X_{t_k}, t_k) \\bigr\\|^2.$$

This is what `kl_to_ref_gaussian` (defined above) computes. It is exact (not an estimator), deterministic given the trajectory, and lower-variance than the trajectory-level $k3$ estimator that we would otherwise need.'''


OLD_KL_MARKDOWN_START = "### The KL estimator"


OLD_GRPO_KL_BLOCK = """            ref_logp = logprob_of_trajectory(ref_model, traj)   # for KL term

        # --- (2) Group-relative advantages ---
        r_grp = r.view(n_groups, group_size)
        adv = (r_grp - r_grp.mean(1, keepdim=True)) / (r_grp.std(1, keepdim=True) + 1e-6)
        adv = adv.reshape(n).detach()

        traj_det = [s.detach() for s in traj]
        old_logp = old_logp.detach()
        ref_logp = ref_logp.detach()

        # --- (3) PPO-style updates with KL penalty ---
        for _ in range(inner_epochs):
            new_logp = logprob_of_trajectory(model, traj_det)
            ratio = torch.exp(new_logp - old_logp)
            unclipped = ratio * adv
            clipped   = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
            pg_loss = -torch.min(unclipped, clipped).mean()

            # k3 KL estimator (Schulman)
            log_r = ref_logp - new_logp
            kl = (torch.exp(log_r) - log_r - 1).mean()
"""


NEW_GRPO_KL_BLOCK = """
        # --- (2) Group-relative advantages ---
        r_grp = r.view(n_groups, group_size)
        adv = (r_grp - r_grp.mean(1, keepdim=True)) / (r_grp.std(1, keepdim=True) + 1e-6)
        adv = adv.reshape(n).detach()

        traj_det = [s.detach() for s in traj]
        old_logp = old_logp.detach()

        # --- (3) PPO-style updates with KL penalty ---
        for _ in range(inner_epochs):
            new_logp = logprob_of_trajectory(model, traj_det)
            ratio = torch.exp(new_logp - old_logp)
            unclipped = ratio * adv
            clipped   = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
            pg_loss = -torch.min(unclipped, clipped).mean()

            # Closed-form Gaussian KL: per-step Gaussians with identical covariance
            # under theta and ref => KL collapses to a scaled L2 distance of means.
            kl = kl_to_ref_gaussian(model, ref_model, traj_det).mean()
"""


def src_str(cell) -> str:
    return ''.join(cell['source'])


def set_src(cell, text: str) -> None:
    # Preserve nbformat's list-of-lines convention.
    lines = text.splitlines(keepends=True)
    cell['source'] = lines


def patch_notebook(path: Path) -> None:
    nb = json.loads(path.read_text())

    # --- Cell 14: append closed-form KL helper ---
    cell14 = nb['cells'][14]
    src14 = src_str(cell14)
    if 'kl_to_ref_gaussian' not in src14:
        set_src(cell14, src14.rstrip() + NEW_KL_HELPER)
        print(f"  cell 14: added kl_to_ref_gaussian")
    else:
        print(f"  cell 14: already patched")

    # --- Cell 19: replace KL-estimator markdown subsection ---
    cell19 = nb['cells'][19]
    src19 = src_str(cell19)
    idx = src19.find(OLD_KL_MARKDOWN_START)
    if idx == -1:
        print(f"  cell 19: '### The KL estimator' marker not found; skipping")
    elif 'closed form' in src19.lower():
        print(f"  cell 19: already patched")
    else:
        new_src19 = src19[:idx] + NEW_KL_MARKDOWN
        set_src(cell19, new_src19)
        print(f"  cell 19: rewrote KL-estimator subsection")

    # --- Cell 20: swap k3 estimator for closed-form call ---
    cell20 = nb['cells'][20]
    src20 = src_str(cell20)
    if OLD_GRPO_KL_BLOCK in src20:
        set_src(cell20, src20.replace(OLD_GRPO_KL_BLOCK, NEW_GRPO_KL_BLOCK))
        print(f"  cell 20: swapped k3 for closed-form KL")
    elif 'kl_to_ref_gaussian' in src20:
        print(f"  cell 20: already patched")
    else:
        print(f"  cell 20: WARNING — expected block not found, leaving cell unchanged")

    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    for name in ('flow_matching_grpo.ipynb', 'flow_matching_grpo_smiley.ipynb'):
        print(f"\n== {name} ==")
        patch_notebook(repo / name)


if __name__ == '__main__':
    main()
