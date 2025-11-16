from __future__ import annotations

from dataclasses import dataclass, field

from dtc_agent.agents import ActorConfig, CriticConfig
from dtc_agent.cognition import WorkspaceConfig
from dtc_agent.memory import EpisodicBufferConfig
from dtc_agent.motivation import EmpowermentConfig, IntrinsicRewardConfig
from dtc_agent.world_model import DecoderConfig, DynamicsConfig, EncoderConfig

from .agent import Agent, CognitiveWaveConfig, StepResult
from .trainer import Trainer


@dataclass
class TrainingConfig:
    """Aggregate configuration for building the full training loop."""

    encoder: EncoderConfig
    decoder: DecoderConfig
    dynamics: DynamicsConfig
    world_model_ensemble: int
    workspace: WorkspaceConfig
    reward: IntrinsicRewardConfig
    empowerment: EmpowermentConfig
    episodic_memory: EpisodicBufferConfig
    rollout_capacity: int = 1024
    batch_size: int = 32
    optimizer_lr: float = 1e-3
    optimizer_empowerment_weight: float = 0.1
    actor: ActorConfig = field(
        default_factory=lambda: ActorConfig(latent_dim=0, action_dim=0)
    )
    critic: CriticConfig = field(default_factory=lambda: CriticConfig(latent_dim=0))
    dream_horizon: int | None = None
    dream_chunk_size: int = 5
    num_dream_chunks: int = 1
    discount_gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    critic_coef: float = 0.5
    world_model_coef: float = 1.0
    self_state_dim: int = 0
    device: str = "cpu"
    compile_modules: bool = False
    adaptive_entropy: bool = True
    adaptive_entropy_target: float = 1.0
    adaptive_entropy_scale: float = 5.0
    cognitive_wave: CognitiveWaveConfig = field(
        default_factory=CognitiveWaveConfig
    )

    @property
    def effective_dream_horizon(self) -> int:
        chunk_product = self.dream_chunk_size * self.num_dream_chunks
        if self.dream_horizon is not None:
            if chunk_product != self.dream_horizon:
                return chunk_product
            return self.dream_horizon
        return chunk_product


class TrainingLoop:
    """Facade exposing the agent and trainer subsystems."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.agent = Agent(config)
        self.trainer = Trainer(config, self.agent)
        self.batch_size = config.batch_size

    def step(
        self,
        observation: "torch.Tensor",
        action: "torch.Tensor | None" = None,
        next_observation: "torch.Tensor | None" = None,
        self_state: "torch.Tensor | None" = None,
        train: bool = False,
    ) -> StepResult:
        """Delegate inference to the agent and optionally stage rollouts."""

        result = self.agent.act(
            observation,
            action=action,
            self_state=self_state,
            train=train,
        )
        if train and next_observation is not None:
            self.trainer.store_transition(
                observation,
                result.action,
                next_observation,
                result.self_state,
            )
        return result

    def store_transition(
        self,
        observation: "torch.Tensor",
        action: "torch.Tensor",
        next_observation: "torch.Tensor",
        self_state: "torch.Tensor | None" = None,
    ) -> None:
        """Compatibility wrapper that forwards to :class:`Trainer`."""

        self.trainer.store_transition(observation, action, next_observation, self_state)

    def train_step(self) -> tuple[int, dict[str, float]] | None:
        """Run a single optimizer update if enough rollouts are available."""

        return self.trainer.train_step()

    def _compute_gae(
        self,
        rewards: "torch.Tensor",
        values: "torch.Tensor",
        next_value: "torch.Tensor",
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Compatibility shim forwarding to :class:`Trainer`."""

        return self.trainer._compute_gae(rewards, values, next_value)

