from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Iterable, Mapping

from omegaconf import OmegaConf

from dtc_agent.training import TrainingConfig
from dtc_agent.agents import ActorConfig, CriticConfig
from dtc_agent.world_model import DecoderConfig, DynamicsConfig, EncoderConfig
from dtc_agent.cognition import WorkspaceConfig
from dtc_agent.cognition.temporal_self import TemporalSelfConfig
from dtc_agent.motivation import EmpowermentConfig, IntrinsicRewardConfig
from dtc_agent.memory import EpisodicBufferConfig


def load_training_config(path: str | Path, overrides: Iterable[str] | None = None) -> TrainingConfig:
    """Load a :class:`TrainingConfig` from disk and apply optional overrides.

    Args:
        path: Path to the YAML configuration file.
        overrides: Optional iterable of ``key=value`` dotlist strings that are
            merged on top of the loaded configuration.

    Returns:
        Parsed :class:`TrainingConfig` instance with nested dataclasses
        materialized.

    Raises:
        TypeError: If the root of the configuration is not a mapping.
        KeyError: If any required configuration section is missing.
    """

    cfg = OmegaConf.load(Path(path))
    if overrides:
        override_conf = OmegaConf.from_dotlist(list(overrides))
        cfg = OmegaConf.merge(cfg, override_conf)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, Mapping):
        raise TypeError("Configuration root must be a mapping")

    encoder = _build_dataclass(EncoderConfig, _extract_section(resolved, "encoder"))
    decoder = _build_dataclass(DecoderConfig, _extract_section(resolved, "decoder"))
    dynamics = _build_dataclass(DynamicsConfig, _extract_section(resolved, "dynamics"))
    workspace = _build_dataclass(WorkspaceConfig, _extract_section(resolved, "workspace"))
    reward = _build_dataclass(
        IntrinsicRewardConfig, _extract_section(resolved, "reward")
    )
    empowerment = _build_dataclass(
        EmpowermentConfig, _extract_section(resolved, "empowerment")
    )
    episodic = _build_dataclass(
        EpisodicBufferConfig, _extract_section(resolved, "episodic_memory")
    )
    temporal_section = resolved.get("temporal_self")
    if isinstance(temporal_section, Mapping):
        temporal_self = _build_dataclass(TemporalSelfConfig, temporal_section)
    else:
        temporal_self = TemporalSelfConfig()
    actor_mapping = resolved.get("actor")
    if isinstance(actor_mapping, Mapping):
        actor_section = actor_mapping
    else:
        actor_section = {}
    critic_mapping = resolved.get("critic")
    if isinstance(critic_mapping, Mapping):
        critic_section = critic_mapping
    else:
        critic_section = {}
    actor_cfg = ActorConfig(
        latent_dim=0,
        action_dim=0,
        hidden_dim=actor_section.get("hidden_dim", 256),
        num_layers=actor_section.get("num_layers", 2),
        dropout=actor_section.get("dropout", 0.0),
    )
    critic_cfg = CriticConfig(
        latent_dim=0,
        hidden_dim=critic_section.get("hidden_dim", 256),
        num_layers=critic_section.get("num_layers", 2),
        dropout=critic_section.get("dropout", 0.0),
    )

    world_model_ensemble = resolved.get("world_model_ensemble")
    if world_model_ensemble is None:
        raise KeyError("world_model_ensemble is required in the configuration")
    rollout_capacity = resolved.get("rollout_capacity", 1024)
    batch_size = resolved.get("batch_size", 32)
    optimizer_lr = resolved.get("optimizer_lr", 1e-3)
    optimizer_empowerment_weight = resolved.get("optimizer_empowerment_weight", 0.1)
    dream_horizon = resolved.get("dream_horizon")
    dream_chunk_size = resolved.get("dream_chunk_size")
    num_dream_chunks = resolved.get("num_dream_chunks")
    if dream_chunk_size is None:
        if dream_horizon is not None:
            dream_chunk_size = dream_horizon
        else:
            dream_chunk_size = 5
    if num_dream_chunks is None:
        num_dream_chunks = 1
    base_dream_horizon = resolved.get(
        "base_dream_horizon", dream_chunk_size * num_dream_chunks
    )
    max_horizon_multiplier = resolved.get("max_horizon_multiplier", 8.0)
    boredom_threshold = resolved.get("boredom_threshold", 0.5)
    horizon_scaling_mode = resolved.get("horizon_scaling_mode", "sigmoid")
    discount_gamma = resolved.get("discount_gamma", 0.99)
    gae_lambda = resolved.get("gae_lambda", 0.95)
    entropy_coef = resolved.get("entropy_coef", 0.01)
    critic_coef = resolved.get("critic_coef", 0.5)
    world_model_coef = resolved.get("world_model_coef", 1.0)
    device = resolved.get("device", "cpu")
    compile_modules = bool(resolved.get("compile_modules", False))
    dream_noise_base_ratio = float(resolved.get("dream_noise_base_ratio", 0.1))
    dream_counterfactual_base_rate = float(
        resolved.get("dream_counterfactual_base_rate", 0.1)
    )
    self_state_dim = resolved.get("self_state_dim", 0)

    return TrainingConfig(
        encoder=encoder,
        decoder=decoder,
        dynamics=dynamics,
        world_model_ensemble=world_model_ensemble,
        workspace=workspace,
        reward=reward,
        empowerment=empowerment,
        episodic_memory=episodic,
        temporal_self=temporal_self,
        rollout_capacity=rollout_capacity,
        batch_size=batch_size,
        optimizer_lr=optimizer_lr,
        optimizer_empowerment_weight=optimizer_empowerment_weight,
        actor=actor_cfg,
        critic=critic_cfg,
        dream_horizon=dream_horizon,
        dream_chunk_size=dream_chunk_size,
        num_dream_chunks=num_dream_chunks,
        discount_gamma=discount_gamma,
        gae_lambda=gae_lambda,
        entropy_coef=entropy_coef,
        critic_coef=critic_coef,
        world_model_coef=world_model_coef,
        self_state_dim=self_state_dim,
        device=device,
        compile_modules=compile_modules,
        dream_noise_base_ratio=dream_noise_base_ratio,
        dream_counterfactual_base_rate=dream_counterfactual_base_rate,
        base_dream_horizon=int(base_dream_horizon),
        max_horizon_multiplier=float(max_horizon_multiplier),
        boredom_threshold=float(boredom_threshold),
        horizon_scaling_mode=str(horizon_scaling_mode),
    )


def _extract_section(resolved: Mapping[str, object], key: str) -> Mapping[str, object]:
    section = resolved.get(key)
    if not isinstance(section, Mapping):
        raise KeyError(f"Configuration missing required mapping: {key}")
    return section


def _build_dataclass(cls, data: Mapping[str, object]):
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    valid_fields = {field.name for field in fields(cls)}
    filtered = {key: value for key, value in data.items() if key in valid_fields}
    return cls(**filtered)
