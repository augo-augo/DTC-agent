from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict

import torch
from torch import nn


@dataclass
class TemporalSelfConfig:
    """Configuration for the temporal self state machine."""

    stimulus_history_window: int = 1000
    stimulus_ema_momentum: float = 0.1
    dream_entropy_scale: float = 10.0
    actor_entropy_scale: float = 5.0
    learning_rate_scale: float = 2.0
    alpha_fast: float = 0.3
    alpha_slow: float = 0.01
    anxiety_penalty: float = 0.1
    lambda_comp_base: float = 1.0
    lambda_comp_bored: float = 0.1
    lambda_emp_base: float = 0.5
    lambda_safety_base: float = 0.5
    lambda_safety_anxious: float = 2.0
    lambda_survival_base: float = 0.1
    lambda_explore_base: float = 0.65
    lambda_explore_bored: float = 2.0
    lambda_explore_anxious: float = 0.1
    novelty_mix_latent_base: float = 0.5
    novelty_mix_decoded_base: float = 0.5
    novelty_mix_latent_bored: float = 0.8
    novelty_mix_decoded_bored: float = 0.2
    novelty_mix_latent_learning: float = 0.3
    novelty_mix_decoded_learning: float = 0.7


class TemporalSelfModule(nn.Module):
    """Tracks competence, stimulus history, and cognitive modes."""

    def __init__(self, config: TemporalSelfConfig) -> None:
        super().__init__()
        self.config = config
        self.field_dim = 3  # ema_fast, ema_slow, deficit
        window = max(1, int(config.stimulus_history_window))
        self.register_buffer("ema_fast", torch.tensor(0.0))
        self.register_buffer("ema_slow", torch.tensor(0.0))
        self.register_buffer("stimulus_level", torch.tensor(0.0))
        self.register_buffer("_ema_initialized", torch.tensor(0.0))
        self.stimulus_history: Deque[float] = deque(maxlen=window)
        self._last_breakdown: dict[str, torch.Tensor] | None = None

    @property
    def last_competence_breakdown(self) -> dict[str, torch.Tensor] | None:
        return self._last_breakdown

    def snapshot(self) -> Dict[str, Any]:
        """Capture internal buffers for temporary what-if simulations."""

        return {
            "ema_fast": self.ema_fast.detach().clone(),
            "ema_slow": self.ema_slow.detach().clone(),
            "stimulus_level": self.stimulus_level.detach().clone(),
            "_ema_initialized": self._ema_initialized.detach().clone(),
            "stimulus_history": deque(self.stimulus_history, maxlen=self.stimulus_history.maxlen),
        }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restore buffers from :meth:`snapshot`."""

        device = self.ema_fast.device
        self.ema_fast.copy_(snapshot["ema_fast"].to(device=device))
        self.ema_slow.copy_(snapshot["ema_slow"].to(device=device))
        self.stimulus_level.copy_(snapshot["stimulus_level"].to(device=device))
        self._ema_initialized.copy_(snapshot["_ema_initialized"].to(device=device))
        history: Deque[float] = snapshot["stimulus_history"]
        self.stimulus_history = deque(history, maxlen=self.stimulus_history.maxlen)

    def get_default_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, Any]:
        """Return an initial temporal-state dictionary for the given batch."""

        device = device or self.ema_fast.device
        dtype = dtype or self.ema_fast.dtype
        zeros_field = torch.zeros(batch_size, self.field_dim, device=device, dtype=dtype)
        r_comp = torch.zeros(batch_size, device=device, dtype=dtype)
        default_mode = "STARTING"
        return {
            "field_tensor": zeros_field,
            "R_comp": r_comp,
            "cognitive_mode": default_mode,
            "stimulus_deficit": 0.0,
            "stimulus_level": float(self.stimulus_level.detach().item()),
            "learning_rate_scale": 1.0,
            "reward_lambdas": self._reward_lambdas_for_mode(default_mode),
            "novelty_mix": self._novelty_mix_for_mode(default_mode),
        }

    def forward(
        self,
        previous_state: dict[str, Any] | None,
        raw_novelty_batch: torch.Tensor,
    ) -> dict[str, Any]:
        """Update internal EMAs and emit the latest temporal field."""

        del previous_state  # Maintained for future compatibility
        if raw_novelty_batch.ndim == 0:
            raw_novelty_batch = raw_novelty_batch.unsqueeze(0)
        batch_size = raw_novelty_batch.shape[0]
        device = self.ema_fast.device
        dtype = raw_novelty_batch.dtype
        novelty_tensor = raw_novelty_batch.detach().to(device=device, dtype=torch.float32)
        with torch.no_grad():
            novelty_mean = novelty_tensor.mean()
            if not torch.isfinite(novelty_mean):
                novelty_mean = torch.tensor(1.0, device=device)

            if self._ema_initialized.item() == 0:
                init_value = torch.clamp(novelty_mean, min=0.05, max=10.0)
                self.ema_fast.copy_(init_value)
                self.ema_slow.copy_(init_value)
                self.stimulus_level.copy_(init_value)
                self._ema_initialized.fill_(1.0)

            ema_fast_prev = self.ema_fast.clone()
            ema_slow_prev = self.ema_slow.clone()

            alpha_fast = float(self.config.alpha_fast)
            alpha_slow = float(self.config.alpha_slow)
            new_fast = (1.0 - alpha_fast) * self.ema_fast + alpha_fast * novelty_mean
            new_slow = (1.0 - alpha_slow) * self.ema_slow + alpha_slow * novelty_mean
            self.ema_fast.copy_(new_fast)
            self.ema_slow.copy_(new_slow)

            momentum = max(0.0, min(1.0, float(self.config.stimulus_ema_momentum)))
            stimulus_level = (1.0 - momentum) * self.stimulus_level + momentum * novelty_mean
            self.stimulus_level.copy_(stimulus_level)

            self.stimulus_history.append(float(novelty_mean.item()))
            if self.stimulus_history:
                baseline_val = sum(self.stimulus_history) / len(self.stimulus_history)
            else:
                baseline_val = float(stimulus_level.item())

            progress = ema_slow_prev - self.ema_fast
            anxiety = self.config.anxiety_penalty * torch.relu(self.ema_fast - ema_slow_prev)
            progress_detached = progress.detach()
            anxiety_detached = anxiety.detach()
            self._last_breakdown = {
                "progress": progress_detached,
                "anxiety": anxiety_detached,
                "ema_fast": self.ema_fast.detach().clone(),
                "ema_slow": self.ema_slow.detach().clone(),
            }

            comp_scalar = (progress - anxiety).to(dtype=dtype)
            r_comp = torch.full((batch_size,), float(comp_scalar.item()), device=device, dtype=dtype)

            stimulus_level_value = float(stimulus_level.detach().item())
            stimulus_deficit = max(0.0, baseline_val - stimulus_level_value)
            stimulus_surplus = max(0.0, stimulus_level_value - baseline_val)

            cognitive_mode = "LEARNING"
            if stimulus_deficit > 0.1:
                cognitive_mode = "BORED"
            elif float(anxiety_detached.item()) > 0.0:
                cognitive_mode = "ANXIOUS"

            lr_scale = 1.0 + stimulus_surplus * self.config.learning_rate_scale
            lr_scale = max(1.0, min(self.config.learning_rate_scale, lr_scale))

            deficit_tensor = torch.full(
                (batch_size,), stimulus_deficit, device=device, dtype=dtype
            )
            field_tensor = torch.stack(
                (
                    self.ema_fast.to(dtype=dtype).expand(batch_size),
                    self.ema_slow.to(dtype=dtype).expand(batch_size),
                    deficit_tensor,
                ),
                dim=-1,
            )
            reward_lambdas = self._reward_lambdas_for_mode(cognitive_mode)
            novelty_mix = self._novelty_mix_for_mode(cognitive_mode)

        return {
            "field_tensor": field_tensor,
            "R_comp": r_comp,
            "cognitive_mode": cognitive_mode,
            "stimulus_deficit": stimulus_deficit,
            "stimulus_level": stimulus_level_value,
            "learning_rate_scale": lr_scale,
            "reward_lambdas": reward_lambdas,
            "novelty_mix": novelty_mix,
        }

    def _reward_lambdas_for_mode(self, cognitive_mode: str) -> dict[str, float]:
        """Derive dynamic reward weights from the current cognitive mode."""

        mode = cognitive_mode.upper()
        lambda_comp = float(self.config.lambda_comp_base)
        lambda_explore = float(self.config.lambda_explore_base)
        lambda_safety = float(self.config.lambda_safety_base)

        if mode == "BORED":
            lambda_comp = float(self.config.lambda_comp_bored)
            lambda_explore = float(self.config.lambda_explore_bored)
        elif mode == "ANXIOUS":
            lambda_explore = float(self.config.lambda_explore_anxious)
            lambda_safety = float(self.config.lambda_safety_anxious)

        return {
            "comp": lambda_comp,
            "explore": lambda_explore,
            "safety": lambda_safety,
            "emp": float(self.config.lambda_emp_base),
            "survival": float(self.config.lambda_survival_base),
        }

    def _novelty_mix_for_mode(self, cognitive_mode: str) -> list[float]:
        """Blend latent/decoded novelty contributions based on cognitive mode."""

        mode = cognitive_mode.upper()
        latent_weight = float(self.config.novelty_mix_latent_base)
        decoded_weight = float(self.config.novelty_mix_decoded_base)

        if mode == "BORED":
            latent_weight = float(self.config.novelty_mix_latent_bored)
            decoded_weight = float(self.config.novelty_mix_decoded_bored)
        elif mode == "LEARNING":
            latent_weight = float(self.config.novelty_mix_latent_learning)
            decoded_weight = float(self.config.novelty_mix_decoded_learning)

        return self._normalize_mix(latent_weight, decoded_weight)

    @staticmethod
    def _normalize_mix(latent_weight: float, decoded_weight: float) -> list[float]:
        """Ensure novelty mix weights are non-negative and normalized."""

        latent = max(0.0, float(latent_weight))
        decoded = max(0.0, float(decoded_weight))
        total = latent + decoded
        if total <= 0.0:
            return [0.5, 0.5]
        return [latent / total, decoded / total]
