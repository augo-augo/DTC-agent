from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F


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


def forward_empowerment_stable(self, action: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    """Stable InfoNCE computation with defensive checks.

    Args:
        self: Empowerment estimator instance.
        action: Batch of actions supplied to the estimator.
        latent: Corresponding latent states.

    Returns:
        Tensor containing bounded empowerment estimates.
    """
    action = sanitize_tensor(action, replacement=0.0)
    latent = sanitize_tensor(latent, replacement=0.0)

    embedded_action = self.action_proj(action)
    embedded_latent = self.latent_proj(latent)

    embedded_action = sanitize_tensor(embedded_action, replacement=0.0)
    embedded_latent = sanitize_tensor(embedded_latent, replacement=0.0)

    negatives = self._collect_negatives(latent)
    negatives = sanitize_tensor(negatives, replacement=0.0)

    all_latents = torch.cat([embedded_latent.unsqueeze(1), negatives], dim=1)

    temperature = torch.clamp(self.temperature, min=0.01, max=10.0)
    logits = torch.einsum("bd,bnd->bn", embedded_action, all_latents) / temperature
    logits = sanitize_tensor(logits, replacement=0.0)
    logits = torch.clamp(logits, min=-20.0, max=20.0)

    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

    loss = F.cross_entropy(logits, labels, reduction="none", label_smoothing=0.01)
    loss = sanitize_tensor(loss, replacement=0.0)

    self._enqueue_latents(latent.detach())

    return -torch.clamp(loss, min=-10.0, max=10.0)


def reward_normalizer_call_stable(self, reward: torch.Tensor) -> torch.Tensor:
    """Normalize rewards while aggressively sanitizing NaN/Inf values.

    Args:
        self: Reward normalizer instance.
        reward: Reward tensor to normalize.

    Returns:
        Tensor of normalized rewards constrained to the configured range.
    """
    if reward.numel() == 0:
        return reward

    reward = sanitize_tensor(reward, replacement=0.0)
    reward_fp32 = reward.float()

    if torch.isfinite(reward_fp32).all():
        self.stats.update(reward_fp32)

    mean = sanitize_tensor(self.stats.mean, replacement=0.0)
    var = sanitize_tensor(self.stats.var, replacement=1.0)
    var = torch.clamp(var, min=self.eps, max=1e6)

    denom = torch.sqrt(var + self.eps)
    normalized = (reward_fp32 - mean) / denom

    normalized = sanitize_tensor(normalized, replacement=0.0)
    normalized = torch.clamp(normalized, -self.clamp_value, self.clamp_value)

    return normalized.to(dtype=reward.dtype)


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


def decoder_forward_stable(self, latent_slots: torch.Tensor) -> torch.distributions.Distribution:
    """Stable ``SharedDecoder.forward`` implementation with bounded variance.

    Args:
        self: Decoder instance whose ``forward`` is being stabilized.
        latent_slots: Latent slot tensor ``[batch, slot_dim]``.

    Returns:
        ``Independent`` Normal distribution over reconstructed observations.

    Raises:
        ValueError: If ``latent_slots`` has incorrect dimensionality or slot size.
    """
    if latent_slots.ndim != 2:
        raise ValueError("latent_slots must have shape [batch, slot_dim]")
    batch, slot_dim = latent_slots.shape
    if slot_dim != self.config.slot_dim:
        raise ValueError("slot_dim mismatch with decoder configuration")

    init_h, init_w = self.config.initial_spatial
    hidden = self.activation(self.fc(latent_slots))
    hidden = hidden.view(batch, self.config.hidden_channels[0], init_h, init_w)
    mean = self.deconv(hidden)

    _, target_h, target_w = self.config.observation_shape
    if mean.shape[-2:] != (target_h, target_w):
        mean = F.interpolate(mean, size=(target_h, target_w), mode="bilinear", align_corners=False)
    mean = self.output_activation(mean)

    log_std_clamped = torch.clamp(self.log_std, min=-5.0, max=0.5)
    std = torch.exp(log_std_clamped).clamp(min=1e-4, max=2.0)

    return torch.distributions.Independent(
        torch.distributions.Normal(mean, std),
        len(self.config.observation_shape),
    )


def slot_attention_forward_stable(self, inputs: torch.Tensor) -> torch.Tensor:
    """Slot Attention forward pass with protected attention normalization.

    Args:
        self: Slot Attention module instance.
        inputs: Tensor of shape ``[batch, num_tokens, dim]``.

    Returns:
        Tensor of slot embeddings produced after stabilized attention updates.
    """
    b, n, d = inputs.shape
    inputs = self.norm_inputs(inputs)
    mu = self.slot_mu.expand(b, self.num_slots, -1)
    sigma = F.softplus(self.slot_sigma)
    slots = mu + sigma * torch.randn_like(mu)

    k = self.project_k(inputs)
    v = self.project_v(inputs)

    for _ in range(self.iters):
        slots_prev = slots
        slots = self.norm_slots(slots)
        q = self.project_q(slots)

        dots = torch.matmul(k, q.transpose(1, 2)) / (d**0.5)
        dots = torch.clamp(dots, min=-20.0, max=20.0)

        attn = dots.softmax(dim=-1) + self.epsilon
        attn_sum = attn.sum(dim=-2, keepdim=True).clamp(min=1e-6)
        attn = attn / attn_sum

        updates = torch.matmul(attn.transpose(1, 2), v)
        slots = self.gru(
            updates.reshape(-1, d),
            slots_prev.reshape(-1, d),
        )
        slots = slots.view(b, self.num_slots, d)
        slots = slots + self.mlp(self.norm_mlp(slots))

    return slots


def optimize_with_loss_check() -> None:  # pragma: no cover - helper inserted inline
    raise RuntimeError(
        "This helper should not be called directly; copy the snippet into TrainingLoop._optimize."
    )


def add_gradient_sanitization() -> None:  # pragma: no cover - helper inserted inline
    raise RuntimeError(
        "This helper should not be called directly; copy the snippet into TrainingLoop._optimize."
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
