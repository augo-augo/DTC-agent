from __future__ import annotations

import math
from typing import Iterable

import torch


def safe_log(x: torch.Tensor, eps: float = 1e-8, clamp_min: float = -10.0) -> torch.Tensor:
    """Compute a logarithm while protecting against zeros and underflow.

    Args:
        x: Input tensor whose elements are logged.
        eps: Minimum value added before the logarithm.
        clamp_min: Lower bound applied to the resulting log values.

    Returns:
        Tensor of logarithms with values clamped to ``clamp_min``.
    """
    return torch.log(torch.clamp(x, min=eps)).clamp(min=clamp_min)


def safe_entropy_gaussian(variance: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute Gaussian entropy while clamping pathological variances.

    Args:
        variance: Estimated variance tensor of the Gaussian distribution.
        eps: Minimum variance threshold used for numerical stability.

    Returns:
        Tensor containing entropy values bounded to a reasonable range.
    """
    var_safe = torch.clamp(variance, min=eps, max=1e6)
    entropy = 0.5 * torch.log((2 * math.pi * math.e) * var_safe)
    return torch.clamp(entropy, min=-20.0, max=20.0)


def sanitize_tensor(x: torch.Tensor, replacement: float = 0.0) -> torch.Tensor:
    """Replace NaN or infinite values in ``x`` with a finite fallback.

    Args:
        x: Input tensor potentially containing non-finite entries.
        replacement: Value substituted where entries are non-finite.

    Returns:
        Tensor with the same shape as ``x`` containing only finite values.
    """
    mask = torch.isfinite(x)
    if bool(mask.all()):
        return x
    return torch.where(mask, x, torch.full_like(x, replacement))


def jensen_shannon_divergence_stable(
    distributions: Iterable[torch.distributions.Distribution],
    num_samples: int = 1,
    max_jsd: float = 10.0,
) -> torch.Tensor:
    """Compute Jensen-Shannon divergence with additional numerical guards.

    Args:
        distributions: Iterable of distributions produced by the ensemble.
        num_samples: Number of Monte Carlo samples per distribution.
        max_jsd: Upper clamp applied to the resulting divergence.

    Returns:
        Tensor containing the stabilized Jensen-Shannon divergence estimate.

    Raises:
        ValueError: If ``distributions`` is empty.
    """
    dists = list(distributions)
    if not dists:
        raise ValueError("At least one distribution is required")

    num_models = len(dists)
    kl_terms: list[torch.Tensor] = []

    for _ in range(max(1, num_samples)):
        for idx, dist in enumerate(dists):
            sample = dist.rsample((1,)).squeeze(0)
            sample = sanitize_tensor(sample, replacement=0.0)

            log_probs: list[torch.Tensor] = []
            for other in dists:
                try:
                    lp = other.log_prob(sample).float()
                except RuntimeError:
                    lp = torch.full(sample.shape, -20.0, device=sample.device, dtype=torch.float32)
                lp = sanitize_tensor(lp, replacement=-20.0)
                lp = torch.clamp(lp, min=-20.0, max=20.0)
                log_probs.append(lp)

            stacked = torch.stack(log_probs)
            logsumexp = torch.logsumexp(stacked, dim=0)
            mixture_log_prob = logsumexp - math.log(num_models)

            kl = log_probs[idx] - mixture_log_prob
            kl = sanitize_tensor(kl, replacement=0.0)
            kl_terms.append(kl)

    js = torch.stack(kl_terms).mean()
    js = sanitize_tensor(js, replacement=0.0)
    return torch.clamp(js, min=0.0, max=max_jsd)


def estimate_observation_entropy_stable(
    observation: torch.Tensor,
    eps: float = 1e-6,
    max_entropy: float = 20.0,
) -> torch.Tensor:
    """Estimate observation entropy with defensive handling of anomalies.

    Args:
        observation: Observation batch ``[batch, channels, height, width]``.
        eps: Minimum variance used when computing entropy.
        max_entropy: Upper clamp applied to the resulting entropy values.

    Returns:
        Tensor of entropy estimates for each observation in the batch.

    Raises:
        ValueError: If ``observation`` does not have 4 dimensions.
    """
    if observation.ndim != 4:
        raise ValueError("observation must be [batch, channels, height, width]")

    batch_size = observation.size(0)
    flat = observation.reshape(batch_size, -1).float()

    finite_mask = torch.isfinite(flat)
    safe_flat = torch.where(finite_mask, flat, torch.zeros_like(flat))
    counts = finite_mask.sum(dim=1).clamp_min(1).float()

    mean = safe_flat.sum(dim=1) / counts
    centered = torch.where(
        finite_mask,
        safe_flat - mean.unsqueeze(1),
        torch.zeros_like(safe_flat),
    )

    variance_sum = centered.pow(2).sum(dim=1)
    variance = (variance_sum / counts).clamp(min=eps, max=1e6)

    entropy = safe_entropy_gaussian(variance, eps=eps)
    return torch.clamp(entropy, min=0.0, max=max_entropy)


def sanitize_gradients(model: torch.nn.Module, max_norm: float = 5.0) -> int:
    """Replace non-finite gradients with zeros and report replacements.

    Args:
        model: Module whose parameters are inspected for invalid gradients.
        max_norm: Unused legacy parameter retained for compatibility.

    Returns:
        Number of gradient elements replaced due to non-finite values.
    """
    bad_count = 0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad
        finite_mask = torch.isfinite(grad)
        if bool(finite_mask.all()):
            continue
        bad_count += grad.numel() - int(finite_mask.sum().item())
        param.grad = torch.where(finite_mask, grad, torch.zeros_like(grad))
    return bad_count


def optimize_with_loss_check() -> None:  # pragma: no cover - helper inserted inline
    raise RuntimeError(
        "This helper should not be called directly; copy the snippet into Trainer.train_step."
    )


def add_gradient_sanitization() -> None:  # pragma: no cover - helper inserted inline
    raise RuntimeError(
        "This helper should not be called directly; copy the snippet into Trainer.train_step."
    )


def add_step_diagnostics() -> None:  # pragma: no cover - helper inserted inline
    raise RuntimeError(
        "This helper should not be called directly; copy the snippet into TrainingLoop.step."
    )


def diagnose_nan_comprehensive(tensor: torch.Tensor, name: str, step: int) -> bool:
    """Log rich diagnostics when ``tensor`` contains non-finite values.

    Args:
        tensor: Tensor to inspect for NaN or Inf values.
        name: Friendly name identifying the tensor in logs.
        step: Training step associated with the diagnostic.

    Returns:
        ``True`` when non-finite values were found and logged, ``False`` otherwise.
    """
    if tensor.numel() == 0:
        return False

    finite_mask = torch.isfinite(tensor)
    if bool(finite_mask.all()):
        return False

    num_nan = int(torch.isnan(tensor).sum().item())
    num_inf = int(torch.isinf(tensor).sum().item())
    num_finite = int(finite_mask.sum().item())
    total = tensor.numel()

    print(f"\n[STEP {step}] ⚠️  NaN/Inf detected in {name}")
    print(f"  Shape: {tuple(tensor.shape)}")
    print(f"  NaN: {num_nan}/{total} ({100 * num_nan / total:.2f}%)")
    print(f"  Inf: {num_inf}/{total} ({100 * num_inf / total:.2f}%)")
    print(f"  Finite: {num_finite}/{total} ({100 * num_finite / total:.2f}%)")

    if num_finite > 0:
        finite_vals = tensor[finite_mask]
        print(
            "  Finite stats: min={:.4f}, max={:.4f}, mean={:.4f}".format(
                float(finite_vals.min()),
                float(finite_vals.max()),
                float(finite_vals.mean()),
            )
        )

    return True
