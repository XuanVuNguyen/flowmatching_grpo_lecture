import logging

import torch

from . import utils
from .model import VelocityNet


logger = logging.getLogger(__name__)


def train_flow_matching(device, steps: int = 4000, batch: int = 1024, lr: float = 1e-3, log_every: int = 500):
    model = VelocityNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    losses = []

    for step in range(steps):
        x1 = utils.sample_checkerboard(batch).to(device)
        x0 = torch.randn_like(x1)
        t = torch.rand(batch, device=device)

        # Linear interpolation path
        xt = (1 - t).view(-1, 1) * x0 + t.view(-1, 1) * x1
        target = x1 - x0
        pred = model(xt, t)
        loss = ((pred - target) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())

        if step % log_every == 0:
            logger.info(f"  step {step}: loss {loss.item():.4f}")

    return model, losses


def grpo_train(
    model,
    ref_model,
    reward_func,
    step_number,
    sigma,
    iters: int = 150,
    group_size: int = 16,
    n_groups: int = 4,
    lr: float = 1e-4,
    kl_coef: float = 0.05,
    clip_ratio: float = 0.2,
    inner_epochs: int = 2,
    log_every: int = 20,
):
    """Group Relative Policy Optimization for a stochastic flow policy."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"reward": [], "kl": [], "pg_loss": []}

    for it in range(iters):
        n = group_size * n_groups

        # rollouts
        with torch.no_grad():
            x_final, old_logp, traj = utils.rollout(model, n, step_number=step_number, sigma=sigma)
            r = reward_func(x_final)
            ref_logp = utils.logprob_of_trajectory(ref_model, traj, step_number=step_number, sigma=sigma)

        # advantage
        r_grp = r.view(n_groups, group_size)
        adv = (r_grp - r_grp.mean(1, keepdim=True)) / (r_grp.std(1, keepdim=True) + 1e-6)
        adv = adv.reshape(n).detach()

        traj_det = [s.detach() for s in traj]
        old_logp = old_logp.detach()
        ref_logp = ref_logp.detach()

        for _ in range(inner_epochs):
            new_logp = utils.logprob_of_trajectory(model, traj_det, step_number=step_number, sigma=sigma)
            ratio = torch.exp(new_logp - old_logp)
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
            pg_loss = -torch.min(unclipped, clipped).mean()

            # k3 KL estimator by John Schulman: http://joschu.net/blog/kl-approx.html
            log_r = ref_logp - new_logp
            kl = (torch.exp(log_r) - log_r - 1).mean()
            
            loss = pg_loss + kl_coef * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        history["reward"].append(r.mean().item())
        history["kl"].append(kl.item())
        history["pg_loss"].append(pg_loss.item())

        if it % log_every == 0:
            logger.info(f"iter {it:3d}: reward {r.mean():.3f}  KL {kl:.3f}  pg_loss {pg_loss:.4f}")

    return history
