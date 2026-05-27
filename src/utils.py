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


def score_corrected_drift(v: torch.Tensor, x: torch.Tensor, t_scalar: float, sigma: float) -> torch.Tensor:
    """Marginal-preserving drift for rectified flow.

        b_t(x) = v + (sigma^2 / (2 (1 - t))) * (t * v - x).

    The (t*v - x) factor is the rectified-flow score expressed in the
    learned velocity (see the Appendix A derivation in the slides). With
    this drift the SDE preserves the same marginals p_t as the ODE.
    """
    coef = sigma ** 2 / (2.0 * (1.0 - t_scalar))
    return v + coef * (t_scalar * v - x)


def c_factor(t_scalar: float, sigma: float) -> float:
    """C_t = 1 + t * sigma^2 / (2 (1 - t)).

    The time-dependent scalar that survives mu_theta - mu_ref after the
    score-correction's -x piece cancels.
    """
    return 1.0 + t_scalar * sigma ** 2 / (2.0 * (1.0 - t_scalar))


def rollout(model, n: int, step_number: int, sigma: float):
    """Roll out n stochastic trajectories with the marginal-preserving SDE.

    Returns (x_final, log_prob, trajectory).
    """
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
        # log-prob of a 2D isotropic Gaussian
        lp = -0.5 * ((x - mean) ** 2).sum(-1) / std ** 2 - 2 * math.log(std * math.sqrt(2 * math.pi))
        log_prob = log_prob + lp
        traj.append(x.clone())

    return x, log_prob, traj


def logprob_of_trajectory(model, traj, step_number, sigma: float):
    """Recompute log-prob of an existing trajectory under `model`.

    Used for the importance-sampling ratio.
    """
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


def kl_to_ref_gaussian(model, ref_model, traj, step_number, sigma: float):
    """Closed-form KL(pi_theta || pi_ref) summed over Euler steps.

    Per-step transitions are Gaussian with the same covariance under both
    theta and ref. Subtracting the means cancels the score-correction's
    -x piece, leaving

        mu_theta - mu_ref = h * C_t * (v_theta - v_ref),

    so the per-step KL is

        KL_step = h * C_t^2 / (2 * sigma^2) * || v_theta - v_ref ||^2,

    with C_t = 1 + t * sigma^2 / (2 (1 - t)) varying step by step.

    Returns a tensor of shape (n,) with the trajectory-level KL.
    """
    device = next(model.parameters()).device
    dt = 1.0 / step_number
    kl = torch.zeros(traj[0].shape[0], device=device)

    for step in range(step_number):
        t_scalar = step * dt
        t = torch.full((traj[step].shape[0],), t_scalar, device=device)
        v_theta = model(traj[step], t)
        with torch.no_grad():
            v_ref = ref_model(traj[step], t)
        C_t = c_factor(t_scalar, sigma)
        kl = kl + 0.5 * dt * (C_t ** 2) / (sigma ** 2) * ((v_theta - v_ref) ** 2).sum(-1)

    return kl