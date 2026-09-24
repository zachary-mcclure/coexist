"""Weight-movement diagnostics shared by the tuning scripts.

Tracks how far the trainable weights have moved from pretrained: a per-step
relative L2 (for the step log) and a final summary (for the saved JSON), so
every tune records its parameter count and movement to line up against its
benchmark verdict (see weight_diagnostics.py).
"""
import torch


def theta0_norm(theta0):
    with torch.no_grad():
        return float(torch.sqrt(sum((t.double() ** 2).sum() for t in theta0)))


def rel_move(params, theta0, norm0):
    """Relative L2 weight change ||theta - theta0|| / ||theta0||, cheap for the loop."""
    with torch.no_grad():
        d2 = float(sum(((p - t) ** 2).sum() for p, t in zip(params, theta0)))
    return (d2 ** 0.5) / norm0 if norm0 > 0 else float("nan")


def move_stats(params, theta0):
    """Final summary dict: n trained, relative L2, largest single move, %moved."""
    with torch.no_grad():
        d = torch.cat([(p - t).flatten() for p, t in zip(params, theta0)]).detach().cpu()
    n0 = theta0_norm(theta0)
    return {"n_trained": int(d.numel()),
            "rel_dtheta": (float((d ** 2).sum()) ** 0.5) / n0 if n0 > 0 else None,
            "max_abs_dw": float(d.abs().max()),
            "frac_moved": float((d.abs() > 1e-4).double().mean())}
