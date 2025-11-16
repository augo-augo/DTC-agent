"""Reusable streaming statistics utilities."""

from __future__ import annotations

import torch

from .stability import sanitize_tensor

__all__ = ["RunningMeanStd", "RewardNormalizer"]


class RunningMeanStd:
    """Track running mean and variance for streaming tensors."""

    def __init__(self, epsilon: float = 1e-4, device: torch.device | None = None) -> None:
        self.epsilon = float(epsilon)
        base_device = torch.device(device) if device is not None else torch.device("cpu")
        self.device = base_device
        self.mean = torch.zeros(1, device=base_device)
        self.var = torch.ones(1, device=base_device)
        self.count = torch.tensor(self.epsilon, device=base_device)

    def to(self, device: torch.device | str) -> "RunningMeanStd":
        """Move the internal buffers to ``device``."""

        target = torch.device(device)
        if target == self.device:
            return self
        self.mean = self.mean.to(target)
        self.var = self.var.to(target)
        self.count = self.count.to(target)
        self.device = target
        return self

    def update(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return

        values = x.detach().to(dtype=torch.float32)
        self.to(values.device)
        values = values.to(device=self.device)
        values = values.reshape(-1, 1)

        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False)
        batch_count = values.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: float
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count


class RewardNormalizer:
    """Normalize intrinsic reward streams to stabilize training."""

    def __init__(
        self,
        device: torch.device | None = None,
        epsilon: float = 1e-4,
        clamp_value: float = 5.0,
        min_variance: float | None = None,
        max_variance: float = 1e6,
    ) -> None:
        self.running_stats = RunningMeanStd(device=device, epsilon=epsilon)
        self.clamp_value = clamp_value
        self._min_variance = min_variance
        self._max_variance = max_variance

    @property
    def stats(self) -> RunningMeanStd:
        return self.running_stats

    def __call__(self, reward: torch.Tensor) -> torch.Tensor:
        if reward.numel() == 0:
            return reward

        reward = sanitize_tensor(reward, replacement=0.0)
        reward_fp32 = reward.detach().to(dtype=torch.float32)
        if reward_fp32.device != self.running_stats.device:
            self.running_stats.to(reward_fp32.device)

        if torch.isfinite(reward_fp32).all():
            self.running_stats.update(reward_fp32)

        mean = sanitize_tensor(self.running_stats.mean, replacement=0.0)
        var = sanitize_tensor(self.running_stats.var, replacement=1.0)
        min_var = self._min_variance if self._min_variance is not None else self.running_stats.epsilon
        var = torch.clamp(var, min=min_var, max=self._max_variance)

        denom = torch.sqrt(var + self.running_stats.epsilon)
        normalized = (reward_fp32 - mean) / denom
        normalized = sanitize_tensor(normalized, replacement=0.0)
        normalized = torch.clamp(normalized, -self.clamp_value, self.clamp_value)

        return normalized.to(dtype=reward.dtype)
