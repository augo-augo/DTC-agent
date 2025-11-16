from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from dtc_agent.motivation.metrics import AdaptiveNoveltyCalculator
from dtc_agent.utils.statistics import RewardNormalizer


NoveltyMetric = Callable[
    [Sequence[torch.Tensor], Sequence[torch.distributions.Distribution] | None],
    torch.Tensor,
]


@dataclass
class IntrinsicRewardConfig:
    """Configuration describing how intrinsic reward components are combined.

    Attributes:
        alpha_fast: EMA coefficient for the fast competence tracker.
        novelty_high: Threshold identifying unusually novel events.
        anxiety_penalty: Penalty applied when novelty spikes.
        safety_entropy_floor: Minimum acceptable observation entropy.
        lambda_comp: Weight for the competence component.
        lambda_emp: Weight for the empowerment component.
        lambda_safety: Weight for the safety penalty.
        lambda_survival: Weight for the survival bonus.
        lambda_explore: Weight for raw novelty-based exploration.
        component_clip: Clamp value applied after per-component normalization.
        alpha_slow: EMA coefficient for the slow competence tracker.
    """

    alpha_fast: float
    novelty_high: float
    anxiety_penalty: float
    safety_entropy_floor: float
    lambda_comp: float
    lambda_emp: float
    lambda_safety: float
    lambda_survival: float = 0.0
    lambda_explore: float = 0.0
    component_clip: float = 5.0
    alpha_slow: float = 0.01


class IntrinsicRewardGenerator:
    """Combine multiple motivational signals into a scalar intrinsic reward."""

    def __init__(
        self,
        config: IntrinsicRewardConfig,
        empowerment_estimator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        novelty_metric: NoveltyMetric,
    ) -> None:
        """Initialize the generator and supporting normalization buffers.

        Args:
            config: Hyper-parameters controlling reward weighting and smoothing.
            empowerment_estimator: Callable estimating empowerment from an
                action-latent pair.
            novelty_metric: Function mapping ensemble predictions to novelty.
        """

        self.config = config
        self.empowerment_estimator = empowerment_estimator
        self.novelty_metric = novelty_metric
        self.novelty_calculator = AdaptiveNoveltyCalculator(target_mean=1.0)
        self.ema_fast = torch.tensor(0.0)
        self.ema_slow = torch.tensor(0.0)
        self.alpha_slow = config.alpha_slow
        self._ema_initialized = False
        normalizer_kwargs = dict(
            clamp_value=config.component_clip,
            epsilon=1e-6,
            min_variance=0.1,
        )
        self._scalers = {
            "competence": RewardNormalizer(**normalizer_kwargs),
            "empowerment": RewardNormalizer(**normalizer_kwargs),
            "safety": RewardNormalizer(**normalizer_kwargs),
            "survival": RewardNormalizer(**normalizer_kwargs),
            "explore": RewardNormalizer(**normalizer_kwargs),
        }

    def get_novelty(
        self,
        predicted_latents: Sequence[torch.Tensor],
        predicted_observations: Sequence[torch.distributions.Distribution] | None,
    ) -> torch.Tensor:
        """Compute epistemic novelty using ensemble disagreement.

        Args:
            predicted_latents: Sequence of latent predictions from the
                dynamics ensemble.
            predicted_observations: Optional decoded observation distributions
                corresponding to ``predicted_latents``.

        Returns:
            Tensor containing the normalized novelty estimate per batch item.
        """

        raw_novelty = self.novelty_metric(predicted_latents, predicted_observations)
        return self.novelty_calculator(raw_novelty)

    def get_competence(self, novelty: torch.Tensor) -> torch.Tensor:
        """Track competence progress from novelty statistics.

        Args:
            novelty: Novelty estimates for the current batch.

        Returns:
            Tensor representing progress minus anxiety for each item in the
            batch.
        """

        if self.ema_fast.device != novelty.device:
            self.ema_fast = self.ema_fast.to(novelty.device)
            self.ema_slow = self.ema_slow.to(novelty.device)

        novelty_mean = novelty.detach().mean()
        if not torch.isfinite(novelty_mean):
            print(f"[Competence] WARNING: Non-finite novelty detected, using fallback")
            novelty_mean = torch.tensor(1.0, device=novelty.device)

        if not self._ema_initialized or self.ema_fast.item() < 1e-6:
            init_value = torch.clamp(novelty_mean, min=0.1, max=10.0)
            self.ema_fast = init_value.clone()
            self.ema_slow = init_value.clone()
            self._ema_initialized = True
            print(
                f"[Competence] Initialized EMAs: fast={self.ema_fast.item():.6f}, "
                f"slow={self.ema_slow.item():.6f}"
            )

        if not torch.isfinite(self.ema_fast) or not torch.isfinite(self.ema_slow):
            print(f"[Competence] ERROR: EMAs corrupted, reinitializing")
            self.ema_fast = torch.clamp(novelty_mean, min=0.1, max=10.0)
            self.ema_slow = self.ema_fast.clone()

        ema_fast_prev = self.ema_fast.clone()
        ema_slow_prev = self.ema_slow.clone()

        alpha_fast = self.config.alpha_fast
        alpha_slow = self.alpha_slow
        self.ema_fast = (1 - alpha_fast) * self.ema_fast + alpha_fast * novelty_mean
        self.ema_slow = (1 - alpha_slow) * self.ema_slow + alpha_slow * novelty_mean

        progress = ema_slow_prev - self.ema_fast
        anxiety = self.config.anxiety_penalty * torch.relu(self.ema_fast - ema_slow_prev)
        self._last_competence_breakdown = {
            "progress": progress.detach(),
            "anxiety": anxiety.detach(),
            "ema_fast_prev": ema_fast_prev.detach(),
            "ema_fast": self.ema_fast.detach(),
            "ema_slow_prev": ema_slow_prev.detach(),
            "ema_slow": self.ema_slow.detach(),
        }
        if hasattr(self, "_step_count") and self._step_count % 100 == 0:
            print(f"[Competence Diagnostic]")
            print(f"  Novelty: {novelty.mean():.6f}")
            print(
                "  EMA fast prev/current: "
                f"{ema_fast_prev:.6f} -> {self.ema_fast:.6f}"
            )
            print(
                "  EMA slow prev/current: "
                f"{ema_slow_prev:.6f} -> {self.ema_slow:.6f}"
            )
            print(f"  Progress: {progress.mean():.6f}")
            print(f"  Anxiety: {anxiety.mean():.6f}")
        batch_size = novelty.shape[0] if novelty.ndim > 0 else 1
        if progress.ndim == 0:
            progress_broadcast = progress.repeat(batch_size)
        else:
            progress_broadcast = progress
        if anxiety.ndim == 0:
            anxiety_broadcast = anxiety.repeat(batch_size)
        else:
            anxiety_broadcast = anxiety
        return progress_broadcast - anxiety_broadcast.to(device=novelty.device)

    def get_safety(self, observation_entropy: torch.Tensor) -> torch.Tensor:
        """Compute a safety penalty when observation entropy falls too low.

        Args:
            observation_entropy: Estimated entropy of the agent's observations.

        Returns:
            Negative tensor penalizing collapse below the entropy floor.
        """

        deficit = torch.relu(self.config.safety_entropy_floor - observation_entropy)
        severe_deficit = torch.relu(deficit - 0.05)
        penalty = deficit + 2.0 * severe_deficit.pow(2)
        return -penalty

    def get_survival(self, self_state: torch.Tensor | None) -> torch.Tensor:
        """Encourage maintaining vital stats such as health and food.

        Args:
            self_state: Optional tensor containing survival-related signals.

        Returns:
            Tensor of bonuses/penalties encouraging healthy self-state values.
        """

        if self_state is None:
            return torch.tensor(0.0, device=self.ema_fast.device)

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
        novelty: torch.Tensor,
        observation_entropy: torch.Tensor,
        action: torch.Tensor,
        latent: torch.Tensor,
        self_state: torch.Tensor | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Aggregate individual motivational signals into a single reward.

        Args:
            novelty: Epistemic novelty estimates for the current batch.
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

        r_comp = self.get_competence(novelty)
        r_emp = self.empowerment_estimator(action, latent)
        r_safe = self.get_safety(observation_entropy)
        r_survival = self.get_survival(self_state).to(device=action.device)
        r_explore = novelty
        batch = action.shape[0]
        if r_comp.ndim == 0:
            r_comp = r_comp.expand(batch)
        if r_safe.ndim == 0:
            r_safe = r_safe.expand(batch)
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
            "safety": self._scalers["safety"](r_safe),
            "survival": self._scalers["survival"](r_survival),
            "explore": r_explore,
        }
        intrinsic = (
            self.config.lambda_comp * normalized["competence"]
            + self.config.lambda_emp * normalized["empowerment"]
            + self.config.lambda_safety * normalized["safety"]
            + self.config.lambda_survival * normalized["survival"]
            + self.config.lambda_explore * normalized["explore"]
        )
        if return_components:
            raw = {
                "competence": r_comp.detach(),
                "empowerment": r_emp.detach(),
                "safety": r_safe.detach(),
                "survival": r_survival.detach(),
                "explore": r_explore.detach(),
            }
            return intrinsic, normalized, raw
        return intrinsic
