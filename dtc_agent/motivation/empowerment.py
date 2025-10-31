from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from typing import TextIO

import torch
from torch import nn

from dtc_agent.utils import sanitize_tensor


def _safe_console_write(message: str, stream: TextIO) -> None:
    data = (message + "\n").encode("utf-8", "replace")
    fileno = getattr(stream, "fileno", None)
    if callable(fileno):
        try:
            os.write(fileno(), data)
            return
        except OSError:
            pass
    try:
        stream.write(message + "\n")
    except Exception:
        pass


def _console_info(message: str) -> None:
    _safe_console_write(message, sys.stdout)


def _console_warn(message: str) -> None:
    _safe_console_write(message, sys.stderr)


@dataclass
class EmpowermentConfig:
    """Configuration for the InfoNCE empowerment estimator.

    Attributes:
        latent_dim: Dimensionality of the latent state representation.
        action_dim: Dimensionality of the agent action vector.
        hidden_dim: Width of the projection MLPs.
        queue_capacity: Number of latent states stored for negative sampling.
        temperature: Softmax temperature used in the InfoNCE objective.
    """

    latent_dim: int
    action_dim: int
    hidden_dim: int = 128
    queue_capacity: int = 128
    temperature: float = 0.1


class InfoNCEEmpowermentEstimator(nn.Module):
    """Estimate empowerment using a replay-backed InfoNCE objective."""

    def __init__(self, config: EmpowermentConfig) -> None:
        """Initialize projection heads and replay storage.

        Args:
            config: Hyper-parameters describing the estimator architecture.
        """

        super().__init__()
        self.config = config
        self.action_proj = nn.Sequential(
            nn.Linear(config.action_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.latent_proj = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.temperature = nn.Parameter(torch.tensor(config.temperature))
        self.register_buffer("_queue", torch.zeros(config.queue_capacity, config.latent_dim))
        self.register_buffer("_queue_step", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_queue_count", torch.zeros(1, dtype=torch.long))
        self._diag_counter = 0

    def forward(self, action: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Compute empowerment rewards for the provided action-latent pairs.

        Args:
            action: Batch of agent actions.
            latent: Corresponding latent states.

        Returns:
            Tensor of per-sample empowerment values (negative InfoNCE loss).
        """
        action = sanitize_tensor(action, replacement=0.0)
        latent = sanitize_tensor(latent, replacement=0.0)

        action_float = action.float()
        latent_float = latent.float()

        embedded_action = self.action_proj(action_float)
        embedded_latent = self.latent_proj(latent_float)

        embedded_action = sanitize_tensor(embedded_action, replacement=0.0)
        embedded_latent = sanitize_tensor(embedded_latent, replacement=0.0)

        negatives = self._collect_negatives(latent)
        negatives = sanitize_tensor(negatives, replacement=0.0)

        all_latents = torch.cat([embedded_latent.unsqueeze(1), negatives], dim=1)

        # CRITICAL: sanitize the temperature parameter before it participates in
        # any division to prevent NaNs from corrupting downstream gradients.
        with torch.no_grad():
            if not torch.isfinite(self.temperature).all():
                _console_warn("[Empowerment] CRITICAL: Temperature corrupted, resetting")
                self.temperature.copy_(
                    torch.tensor(
                        self.config.temperature,
                        device=self.temperature.device,
                        dtype=self.temperature.dtype,
                    )
                )

        temperature = torch.clamp(self.temperature.detach(), min=0.05, max=5.0)

        logits = torch.einsum("bd,bnd->bn", embedded_action, all_latents) / temperature
        logits = sanitize_tensor(logits, replacement=0.0)
        logits = torch.clamp(logits, min=-20.0, max=20.0)

        if self.training:
            self._diag_counter += 1
            should_log = self._diag_counter % 100 == 0
        else:
            should_log = False

        if should_log:
            with torch.no_grad():
                labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
                predicted = logits.argmax(dim=-1)
                accuracy = (predicted == labels).float().mean()
                _console_info("[Empowerment Diagnostic]")
                _console_info(f"  Logits: mean={logits.mean():.3f}, std={logits.std():.3f}")
                _console_info(f"  Temperature: {float(temperature.mean()):.4f}")
                _console_info(f"  Contrastive accuracy: {accuracy:.3f} (target: 0.6-0.8)")
                if accuracy > 0.95:
                    _console_warn("  ⚠️  WARNING: Accuracy too high - queue may be contaminated")
                elif accuracy < 0.4:
                    _console_warn("  ⚠️  WARNING: Accuracy too low - embeddings may be broken")

        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = nn.functional.cross_entropy(
            logits, labels, reduction="none", label_smoothing=0.01
        )
        loss = sanitize_tensor(loss, replacement=0.0)

        self._enqueue_latents(latent.detach())

        return -torch.clamp(loss, min=-10.0, max=10.0)

    def _collect_negatives(self, latent: torch.Tensor) -> torch.Tensor:
        """Collect a diverse batch of negative samples from the replay queue."""

        batch, _ = latent.shape
        available = int(self._queue_count.item())

        if available == 0:
            return self.latent_proj(latent.detach().float()).unsqueeze(1)

        capacity = self._queue.size(0)
        limit = min(available, capacity)
        current_ptr = int(self._queue_step.item()) % capacity

        # CRITICAL: Exclude the most recent entries to avoid temporal correlation
        # This prevents the queue from containing the current batch as negatives
        if limit <= 1:
            return self.latent_proj(latent.detach().float()).unsqueeze(1)

        min_age = max(int(limit * 0.15), 16)
        min_age = min(min_age, limit - max(batch, 16))

        valid_indices: list[int] = []
        for i in range(limit):
            age = (current_ptr - i + capacity) % capacity
            if age > min_age:
                valid_indices.append(i)

        if len(valid_indices) < max(8, batch):
            exclusion_window = min(8, limit // 4)
            valid_indices = [
                i
                for i in range(limit)
                if (current_ptr - i + capacity) % capacity > exclusion_window
            ]

        queue_tensor = self._queue[:limit]
        if queue_tensor.device != latent.device:
            queue_tensor = queue_tensor.to(latent.device, non_blocking=True)

        valid_indices_tensor = torch.as_tensor(
            valid_indices, device=latent.device, dtype=torch.long
        )

        if valid_indices_tensor.numel() < batch:
            noise = torch.randn_like(latent) * 0.1
            return self.latent_proj((latent.detach() + noise).float()).unsqueeze(1)

        # Sample without replacement
        num_negatives = batch
        perm = torch.randperm(valid_indices_tensor.numel(), device=latent.device)
        idx = valid_indices_tensor[perm[:num_negatives]]

        sampled = queue_tensor.index_select(0, idx).detach()
        sampled_float = sampled.float()
        embedded = self.latent_proj(sampled_float)
        return embedded.unsqueeze(1)

    def _enqueue_latents(self, latent: torch.Tensor) -> None:
        if latent.numel() == 0:
            return
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
        device = self._queue.device
        data = latent.detach().to(device=device, dtype=self._queue.dtype, non_blocking=True)
        capacity = self._queue.size(0)
        start = int(self._queue_step.item())
        positions = (torch.arange(data.size(0), device=device, dtype=torch.long) + start) % capacity
        self._queue.index_copy_(0, positions, data.detach())
        self._queue_step.add_(data.size(0))
        self._queue_count.add_(data.size(0))
        self._queue_count.clamp_(max=capacity)

    def get_queue_diagnostics(self) -> dict[str, float]:
        """Return basic diagnostics about the latent replay queue.

        Returns:
            Mapping containing ``queue_size`` and ``queue_diversity`` metrics.
        """
        with torch.no_grad():
            available = int(self._queue_count.item())
            if available == 0:
                return {"queue_size": 0.0, "queue_diversity": 0.0}

            capacity = self._queue.size(0)
            limit = min(available, capacity)
            queue_tensor = self._queue[:limit]

            if limit > 1:
                n_pairs = min(100, limit * (limit - 1) // 2)
                idx1 = torch.randint(0, limit, (n_pairs,), device=queue_tensor.device)
                idx2 = torch.randint(0, limit, (n_pairs,), device=queue_tensor.device)
                mask = idx1 != idx2
                idx1 = idx1[mask]
                idx2 = idx2[mask]
                if len(idx1) > 0:
                    pairs = queue_tensor[idx1] - queue_tensor[idx2]
                    diversity = pairs.norm(dim=-1).mean().item()
                else:
                    diversity = 0.0
            else:
                diversity = 0.0

            return {
                "queue_size": float(available),
                "queue_diversity": float(diversity),
            }
