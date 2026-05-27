import math
import torch

def sample_checkerboard(n: int) -> torch.Tensor:
    SQUARE_SIZE = 0.7  # < 1.0 creates gaps; smaller = bigger gaps

    # Pick which column (0,1,2,3) and row-pair each point belongs to
    col = torch.randint(0, 4, (n,))           # 4 columns: x in {-2,-1,0,1}
    row_choice = torch.randint(0, 2, (n,))    # 2 occupied rows per column

    # Position within the square: in [0, SQUARE_SIZE) instead of [0, 1)
    x_offset = torch.rand(n) * SQUARE_SIZE
    y_offset = torch.rand(n) * SQUARE_SIZE

    # Center the small square inside its unit cell so gaps are symmetric
    pad = (1 - SQUARE_SIZE) / 2
    x = col.float() - 2 + pad + x_offset

    # Checkerboard row pattern: even cols use rows {-2, 0}, odd cols use {-1, 1}
    base_row = row_choice.float() * 2 - 2 + (col % 2).float()
    y = base_row + pad + y_offset

    return torch.stack([x, y], dim=1)


@torch.no_grad()
def sample_ode(model, n: int, step_number: int) -> torch.Tensor:
    """Deterministic Euler integration of dx/dt = v(x, t) from t=0 (noise) to t=1 (data)."""
    device = next(model.parameters()).device

    x = torch.randn(n, 2, device=device)
    dt = 1.0 / step_number

    for step in range(step_number):
        t = torch.full((n,), step * dt, device=device)
        x = x + model(x, t) * dt
        
    return x.cpu()


def rollout(model, n: int, step_number: int, sigma: float):
    """Roll out n stochastic trajectories with the marginal-preserving SDE"""
    device = next(model.parameters()).device
    dt = 1.0 / step_number
    std = sigma * math.sqrt(dt)

    x = torch.randn(n, 2, device=device)
    log_prob = torch.zeros(n, device=device)
    traj = [x.clone()]

    for step in range(step_number):
        t_scalar = step * dt
        t = torch.full((n,), t_scalar, device=device)
        
        v = model(x, t)
        b = score_corrected_drift(v, x, t_scalar, sigma)

        mean = x + b * dt
        x = mean + std * torch.randn_like(x)

        lp = -0.5 * ((x - mean) ** 2).sum(-1) / std ** 2 - 2 * math.log(std * math.sqrt(2 * math.pi))
        log_prob = log_prob + lp
        traj.append(x.clone())

    return x, log_prob, traj


def score_corrected_drift(v: torch.Tensor, x: torch.Tensor, t_scalar: float, sigma: float) -> torch.Tensor:
    coef = sigma ** 2 / (2.0 * (1.0 - t_scalar))
    return v + coef * (t_scalar * v - x)

def logprob_of_trajectory(model, traj, step_number, sigma: float):
    """Recompute log-prob of an existing trajectory under `model`."""
    device = next(model.parameters()).device
    dt = 1.0 / step_number
    std = sigma * math.sqrt(dt)
    log_prob = torch.zeros(traj[0].shape[0], device=device)

    for step in range(step_number):
        t_scalar = step * dt
        t = torch.full((traj[step].shape[0],), t_scalar, device=device)
        v = model(traj[step], t)
        b = score_corrected_drift(v, traj[step], t_scalar, sigma)
        mean = traj[step] + b * dt

        lp = -0.5 * ((traj[step + 1] - mean) ** 2).sum(-1) / std ** 2 - 2 * math.log(std * math.sqrt(2 * math.pi))
        log_prob = log_prob + lp

    return log_prob