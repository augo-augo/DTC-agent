from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from dtc_agent.motivation.metrics import (
    AdaptiveNoveltyCalculator,
    ensemble_epistemic_novelty,
)


class _RewardScaler:
    """Running mean/variance normalizer applied to individual intrinsic components."""

    def __init__(self, clamp_value: float = 5.0, eps: float = 1e-6) -> None:
        """Set up the normalizer state.

        Args:
            clamp_value: Symmetric magnitude limit applied to normalized values.
            eps: Small constant added for numerical stability.
        """

        self.clamp_value = clamp_value
        self.eps = eps
        self.min_var = 0.1
        self.mean: torch.Tensor | None = None
        self.var: torch.Tensor | None = None
        self.count: torch.Tensor | None = None

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        """Normalize a reward component using running statistics.

        Args:
            value: Reward component tensor to normalize.

        Returns:
            Tensor with stabilized mean and variance, clamped to
            ``[-clamp_value, clamp_value]``.
        """

        if value.numel() == 0:
            return value

        # Short-circuit on non-finite inputs to avoid contaminating running stats.
        if not torch.all(torch.isfinite(value.detach())):
            return torch.zeros_like(value)

        stats_value = value.detach().to(dtype=torch.float32)
        if self.mean is None or self.mean.device != stats_value.device:
            device = stats_value.device
            self.mean = torch.zeros(1, device=device)
            self.var = torch.ones(1, device=device)
            self.count = torch.tensor(self.eps, device=device)
        flat = stats_value.view(-1)
        batch_mean = flat.mean()
        batch_var = flat.var(unbiased=False)
        batch_count = flat.numel()
        with torch.no_grad():
            delta = batch_mean - self.mean
            total = self.count + batch_count
            self.mean = self.mean + delta * (batch_count / total)
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total
            self.var = torch.clamp(m2 / total, min=self.min_var)
            self.count = total
        mean = self.mean.to(dtype=value.dtype)
        var = self.var.to(dtype=value.dtype)
        normalized = (value - mean) / torch.sqrt(var + self.eps)
        return normalized.clamp(-self.clamp_value, self.clamp_value)


@dataclass
class IntrinsicRewardConfig:
    """Configuration describing how intrinsic reward components are combined.

    Attributes:
        novelty_high: Threshold identifying unusually novel events.
        component_clip: Clamp value applied after per-component normalization.
    """

    novelty_high: float
    component_clip: float = 5.0


class IntrinsicRewardGenerator:
    """Combine multiple motivational signals into a scalar intrinsic reward."""

    def __init__(
        self,
        config: IntrinsicRewardConfig,
        empowerment_estimator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        """Initialize the generator and supporting normalization buffers.

        Args:
            config: Hyper-parameters controlling reward weighting and smoothing.
            empowerment_estimator: Callable estimating empowerment from an
                action-latent pair.
        """

        self.config = config
        self.empowerment_estimator = empowerment_estimator
        self.novelty_metric = ensemble_epistemic_novelty
        self.novelty_calculator = AdaptiveNoveltyCalculator(target_mean=1.0)
        self._scalers = {
            "competence": _RewardScaler(clamp_value=config.component_clip),
            "empowerment": _RewardScaler(clamp_value=config.component_clip),
            "survival": _RewardScaler(clamp_value=config.component_clip),
            "explore": _RewardScaler(clamp_value=config.component_clip),
        }
        self._default_reward_lambdas: dict[str, float] = {
            "comp": 0.0,
            "emp": 0.0,
            "survival": 0.0,
            "explore": 0.0,
        }

    def get_novelty(
        self,
        predicted_latents: Sequence[torch.Tensor | torch.distributions.Distribution],
        predicted_observations: Sequence[torch.distributions.Distribution] | None,
        *,
        novelty_mix: Sequence[float] | None = None,
    ) -> torch.Tensor:
        """Compute epistemic novelty with aleatoric subtraction.

        Args:
            predicted_latents: Sequence of latent predictions from the
                dynamics ensemble.
            predicted_observations: Retained for compatibility; unused.
            novelty_mix: Unused placeholder for backward compatibility.

        Returns:
            Tensor containing the normalized novelty estimate per batch item.
        """

        del predicted_observations, novelty_mix
        if not isinstance(predicted_latents, Sequence) or len(predicted_latents) == 0:
            raise ValueError("predicted_latents must contain at least one element")

        means = []
        variances = []
        for latent in predicted_latents:
            if isinstance(latent, torch.distributions.Distribution):
                means.append(latent.mean.to(dtype=torch.float32))
                variances.append(latent.variance.to(dtype=torch.float32))
            else:
                tensor_latent = latent.to(dtype=torch.float32)
                means.append(tensor_latent)
                variances.append(torch.zeros_like(tensor_latent))

        stacked_means = torch.stack(means, dim=0)
        epistemic = stacked_means.var(dim=0, unbiased=False).mean(dim=-1)
        stacked_vars = torch.stack(variances, dim=0)
        aleatoric = stacked_vars.mean(dim=0).mean(dim=-1)
        clean_novelty = torch.relu(epistemic - aleatoric)
        return self.novelty_calculator(clean_novelty)

    def get_survival(self, self_state: torch.Tensor | None) -> torch.Tensor:
        """Encourage maintaining vital stats such as health and food.

        Args:
            self_state: Optional tensor containing survival-related signals.

        Returns:
            Tensor of bonuses/penalties encouraging healthy self-state values.
        """

        if self_state is None:
            return torch.tensor(0.0)

        if self_state.ndim == 1:
            state = self_state.unsqueeze(0)
        else:
            state = self_state

        if state.size(-1) < 2:
            return torch.zeros(state.size(0), device=state.device)

        health = state[:, 0].clamp(0.0, 1.0)
        food = state[:, 1].clamp(0.0, 1.0)

        survival_bonus = (health > 0.5).float() + (food > 0.5).float()

        health_critical = (health < 0.3).float()
        food_critical = (food < 0.3).float()
        survival_penalty = (health_critical + food_critical) * -3.0

        return survival_bonus + survival_penalty

    def get_intrinsic_reward(
        self,
        temporal_state: dict[str, torch.Tensor] | None,
        novelty_batch: torch.Tensor,
        observation_entropy: torch.Tensor,
        action: torch.Tensor,
        latent: torch.Tensor,
        self_state: torch.Tensor | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Aggregate individual motivational signals into a single reward.

        Args:
            temporal_state: Output dictionary from :class:`TemporalSelfModule`
                containing ``reward_lambdas`` and optional competence traces.
            novelty_batch: Epistemic novelty estimates for the current batch.
            observation_entropy: Entropy of the agent's observations.
            action: Actions sampled by the policy.
            latent: Latent features associated with ``action``.
            self_state: Optional auxiliary state for survival tracking.
            return_components: When ``True`` also return normalized and raw
                component dictionaries.

        Returns:
            Either the scalar intrinsic reward tensor, or a tuple containing the
            reward along with normalized and raw component dictionaries when
            ``return_components`` is ``True``.
        """

        del observation_entropy  # Retained for API compatibility
        batch = action.shape[0]
        device = action.device
        dtype = novelty_batch.dtype
        if temporal_state is None:
            r_comp = torch.zeros(batch, device=device, dtype=dtype)
        else:
            lp_tensor = temporal_state.get("learning_progress")
            if lp_tensor is None:
                lp_tensor = temporal_state.get("R_comp")
            if lp_tensor is None:
                r_comp = torch.zeros(batch, device=device, dtype=dtype)
            else:
                r_comp = lp_tensor.to(device=device, dtype=dtype)
                if r_comp.ndim == 0:
                    r_comp = r_comp.expand(batch)
                elif r_comp.size(0) != batch:
                    if r_comp.size(0) == 1:
                        r_comp = r_comp.expand(batch)
                    else:
                        r_comp = r_comp[:batch]
        if temporal_state is None:
            lambdas = self._default_reward_lambdas
        else:
            reward_entry = temporal_state.get("reward_lambdas")
            if isinstance(reward_entry, dict):
                lambdas = reward_entry
            else:
                lambdas = self._default_reward_lambdas
        r_emp = self.empowerment_estimator(action, latent)
        r_survival = self.get_survival(self_state).to(device=action.device)
        r_explore = novelty_batch
        if r_comp.ndim == 0:
            r_comp = r_comp.expand(batch)
        if r_survival.ndim == 0:
            r_survival = r_survival.expand(batch)
        elif r_survival.size(0) != batch and r_survival.size(0) > 0:
            if r_survival.size(0) == 1:
                r_survival = r_survival.expand(batch)
            else:
                r_survival = r_survival[:batch]
        if r_explore.ndim == 0:
            r_explore = r_explore.expand(batch)
        normalized = {
            "competence": self._scalers["competence"](r_comp),
            "empowerment": self._scalers["empowerment"](r_emp),
            "survival": self._scalers["survival"](r_survival),
            "explore": r_explore,
        }
        lambda_comp = float(lambdas.get("comp", 0.0))
        lambda_emp = float(lambdas.get("emp", 0.0))
        lambda_survival = float(lambdas.get("survival", 0.0))
        lambda_explore = float(lambdas.get("explore", 0.0))
        intrinsic = (
            lambda_comp * normalized["competence"]
            + lambda_emp * normalized["empowerment"]
            + lambda_survival * normalized["survival"]
            + lambda_explore * normalized["explore"]
        )
        if return_components:
            raw = {
                "competence": r_comp.detach(),
                "empowerment": r_emp.detach(),
                "survival": r_survival.detach(),
                "explore": r_explore.detach(),
            }
            return intrinsic, normalized, raw
        return intrinsic
