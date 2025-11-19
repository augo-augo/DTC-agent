from __future__ import annotations
from dataclasses import dataclass, field
from contextlib import nullcontext
from collections import OrderedDict
import math
import threading
from typing import Any, Callable, ContextManager, ParamSpec, Protocol, TypeVar, cast

import torch
from torch import nn
from dtc_agent.agents import (
    ActorConfig,
    ActorNetwork,
    CriticConfig,
    CriticNetwork,
)
from dtc_agent.cognition import WorkspaceConfig, WorkspaceRouter
from dtc_agent.cognition.temporal_self import TemporalSelfConfig, TemporalSelfModule
from dtc_agent.memory import EpisodicBuffer, EpisodicBufferConfig, EpisodicSnapshot
from dtc_agent.motivation import (
    EmpowermentConfig,
    IntrinsicRewardConfig,
    IntrinsicRewardGenerator,
    InfoNCEEmpowermentEstimator,
    estimate_observation_entropy,
)
from dtc_agent.utils import sanitize_gradients, sanitize_tensor
from dtc_agent.world_model import (
    DecoderConfig,
    DynamicsConfig,
    EncoderConfig,
    WorldModelConfig,
    WorldModelEnsemble,
)
from .buffer import RolloutBuffer


P = ParamSpec("P")
T = TypeVar("T")


class _CallableModule(Protocol[P, T]):
    training: bool

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        ...

    def to(self, *args: Any, **kwargs: Any) -> "_CallableModule[P, T]":
        ...

    def train(self, mode: bool = ...) -> "_CallableModule[P, T]":
        ...


def _resolve_compile() -> Callable[[nn.Module], nn.Module]:
    compile_fn = getattr(torch, "compile", None)
    if not callable(compile_fn):
        return lambda module: module
    try:
        import torch._dynamo as _dynamo  # type: ignore[attr-defined]

        _dynamo.config.suppress_errors = True
    except Exception:
        pass
    def _compiler(module: nn.Module) -> nn.Module:
        try:
            return compile_fn(module)  # type: ignore[misc]
        except Exception:
            return module
    return _compiler


_maybe_compile = _resolve_compile()


def module_state_dict(module: nn.Module) -> OrderedDict[str, torch.Tensor]:
    """Return a clean state_dict even if ``module`` was torch.compile'd."""

    original = getattr(module, "_orig_mod", None)
    if isinstance(original, nn.Module):
        module = original
    return module.state_dict()


class _NullGradScaler:
    __slots__ = ()

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:  # pragma: no cover - parity with AMP API
        return None

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:  # pragma: no cover - parity with AMP API
        return None

    def state_dict(self) -> dict[str, float]:
        return {}

    def load_state_dict(self, state_dict: dict[str, float]) -> None:  # pragma: no cover - parity with AMP API
        return None


def _create_grad_scaler(device_type: str, enabled: bool):
    if device_type != "cuda":
        return _NullGradScaler()
    try:
        return torch.amp.GradScaler(enabled=enabled)  # type: ignore[attr-defined]
    except AttributeError:
        from torch.cuda.amp import GradScaler as LegacyGradScaler  # type: ignore[attr-defined]

        return LegacyGradScaler(enabled=enabled)


def _configure_tf32_precision(device: torch.device) -> None:
    """Configure TF32 behavior while avoiding mixed legacy/new backend APIs."""

    if device.type != "cuda":
        return

    configured_matmul = False
    matmul_backend = getattr(torch.backends.cuda, "matmul", None)
    if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
        try:
            matmul_backend.allow_tf32 = True
            configured_matmul = True
        except (TypeError, RuntimeError, AttributeError):
            configured_matmul = False

    if not configured_matmul and hasattr(torch._C, "_set_cublas_allow_tf32"):
        try:
            torch._C._set_cublas_allow_tf32(True)
            configured_matmul = True
        except AttributeError:
            configured_matmul = False

    if not configured_matmul and matmul_backend is not None:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
            configured_matmul = True
        except AttributeError:
            pass

    configured_cudnn = False
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
        try:
            cudnn_backend.allow_tf32 = True
            configured_cudnn = True
        except (TypeError, RuntimeError, AttributeError):
            configured_cudnn = False

    if not configured_cudnn and hasattr(torch._C, "_set_cudnn_allow_tf32"):
        try:
            torch._C._set_cudnn_allow_tf32(True)
            configured_cudnn = True
        except AttributeError:
            configured_cudnn = False

    if not configured_cudnn and cudnn_backend is not None:
        try:
            torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
            configured_cudnn = True
        except AttributeError:
            pass


class RunningMeanStd:
    """Track running mean and variance for streaming tensors."""

    def __init__(self, device: torch.device, epsilon: float = 1e-4) -> None:
        """Initialize running statistics on the specified device.

        Args:
            device: Device where the statistics should reside.
            epsilon: Initial count and minimum variance guard.
        """

        self.device = device
        self.epsilon = epsilon
        self.mean = torch.zeros(1, device=device)
        self.var = torch.ones(1, device=device)
        self.count = torch.tensor(epsilon, device=device)

    def update(self, x: torch.Tensor) -> None:
        """Update the running statistics with a new batch of samples.

        Args:
            x: Tensor of values whose flattened entries update the moments.
        """

        if x.numel() == 0:
            return
        values = x.detach().to(self.device, dtype=torch.float32).reshape(-1, 1)
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
        new_var = m2 / total_count
        self.mean = new_mean
        self.var = torch.clamp(new_var, min=self.epsilon)
        self.count = total_count


class RewardNormalizer:
    """Keeps intrinsic rewards within a bounded scale using running statistics."""

    def __init__(self, device: torch.device, clamp_value: float = 5.0, eps: float = 1e-6) -> None:
        """Create the normalizer with associated running statistics.

        Args:
            device: Device on which normalization is performed.
            clamp_value: Symmetric clamp applied after normalization.
            eps: Minimum variance used to prevent division by zero.
        """

        self.stats = RunningMeanStd(device=device)
        self.clamp_value = clamp_value
        self.eps = eps
        self.device = device

    def __call__(self, reward: torch.Tensor) -> torch.Tensor:
        """Normalize rewards to have stable scale across training steps.

        Args:
            reward: Tensor of intrinsic rewards to stabilize.

        Returns:
            Tensor of normalized rewards with clamped magnitude.
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


@dataclass
class LatentSnapshot:
    """Latent tensors captured during a policy evaluation step."""

    z_self: torch.Tensor
    slots: torch.Tensor
    latent_state: torch.Tensor


@dataclass
class StepResult:
    """Container summarizing the outputs of a single training step.

    Attributes list the tensors collected for analysis and logging.
    """

    action: torch.Tensor
    intrinsic_reward: torch.Tensor
    novelty: torch.Tensor
    observation_entropy: torch.Tensor
    slot_scores: torch.Tensor
    reward_components: dict[str, torch.Tensor] | None = None
    raw_reward_components: dict[str, torch.Tensor] | None = None
    competence_breakdown: dict[str, torch.Tensor] | None = None
    epistemic_novelty: torch.Tensor | None = None
    real_action_entropy: float | None = None
    latent_snapshot: LatentSnapshot | None = None


@dataclass
class TrainingConfig:
    """Aggregate configuration for building the full training loop.

    Attributes summarize all subsystem configurations and training hyper
    parameters.
    """

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
    policy_snapshot_interval: int = 100
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
    dream_noise_base_ratio: float = 0.1
    dream_counterfactual_base_rate: float = 0.1
    dream_from_memory_rate: float = 0.0
    base_dream_horizon: int = 15
    max_horizon_multiplier: float = 8.0
    boredom_threshold: float = 0.5
    horizon_scaling_mode: str = "sigmoid"
    temporal_self: TemporalSelfConfig = field(default_factory=TemporalSelfConfig)

    @property
    def effective_dream_horizon(self) -> int:
        """Return the effective horizon implied by chunk size and count."""

        chunk_product = self.dream_chunk_size * self.num_dream_chunks
        if self.dream_horizon is not None:
            if chunk_product != self.dream_horizon:
                return chunk_product
            return self.dream_horizon
        return chunk_product


class TrainingLoop:
    """High-level container wiring the major subsystems together."""

    def __init__(self, config: TrainingConfig) -> None:
        """Instantiate model components, buffers, and optimizers.

        Args:
            config: Fully specified training configuration dataclass.
        """

        self.config = config
        self.device = torch.device(config.device)
        _configure_tf32_precision(self.device)
        self.progress_momentum = config.workspace.progress_momentum
        self.action_cost_scale = config.workspace.action_cost_scale
        wm_config = WorldModelConfig(
            encoder=config.encoder,
            decoder=config.decoder,
            dynamics=config.dynamics,
            ensemble_size=config.world_model_ensemble,
        )
        world_model = WorldModelEnsemble(wm_config).to(self.device)
        if self.device.type == "cuda" and config.compile_modules:
            self.world_model = _maybe_compile(world_model)
        else:
            self.world_model = world_model
        self.workspace = WorkspaceRouter(config.workspace)
        self.memory = EpisodicBuffer(config.episodic_memory)
        empowerment = InfoNCEEmpowermentEstimator(config.empowerment).to(self.device)
        self.empowerment = empowerment
        self.reward = IntrinsicRewardGenerator(
            config.reward,
            empowerment_estimator=self.empowerment,
        )
        self.reward_normalizer = RewardNormalizer(device=self.device)
        self.temporal_self = TemporalSelfModule(config.temporal_self).to(self.device)
        self.current_temporal_state: dict[str, Any] | None = None
        # Policy dimensions derived from encoder/workspace layout.
        slot_dim = config.encoder.slot_dim
        policy_feature_dim = (
            slot_dim
            + slot_dim * config.workspace.broadcast_slots
            + config.episodic_memory.key_dim
            + self.temporal_self.field_dim
        )
        actor_cfg = ActorConfig(
            latent_dim=policy_feature_dim,
            action_dim=config.dynamics.action_dim,
            hidden_dim=config.actor.hidden_dim,
            num_layers=config.actor.num_layers,
            dropout=config.actor.dropout,
        )
        actor_net = ActorNetwork(actor_cfg).to(self.device)
        critic_net = CriticNetwork(
            CriticConfig(
                latent_dim=policy_feature_dim,
                hidden_dim=config.critic.hidden_dim,
                num_layers=config.critic.num_layers,
                dropout=config.critic.dropout,
            )
        ).to(self.device)
        if self.device.type == "cuda" and config.compile_modules:
            self.actor = _maybe_compile(actor_net)
            self.critic = _maybe_compile(critic_net)
        else:
            self.actor = actor_net
            self.critic = critic_net
        self.actor_eval = ActorNetwork(actor_cfg).to(self.device)
        self.actor_eval.load_state_dict(module_state_dict(self.actor))
        self.actor_eval.eval()

        self.self_state_dim = config.self_state_dim
        if self.self_state_dim > 0:
            self.self_state_encoder = nn.Linear(
                self.self_state_dim, slot_dim, bias=False
            ).to(self.device)
            self.self_state_predictor = nn.Linear(
                slot_dim, self.self_state_dim
            ).to(self.device)
        else:
            self.self_state_encoder = None
            self.self_state_predictor = None

        self._slot_baseline: torch.Tensor | None = None
        self._ucb_mean: torch.Tensor | None = None
        self._ucb_counts: torch.Tensor | None = None
        self._step_count: int = 0
        self._novelty_trace: torch.Tensor | None = None
        self._latest_self_state: torch.Tensor | None = None
        self._step_lock = threading.Lock()
        self._policy_snapshot_interval = max(1, config.policy_snapshot_interval)
        self._last_snapshot_sync = 0

        self.rollout_buffer = RolloutBuffer(capacity=config.rollout_capacity)
        self.batch_size = config.batch_size
        params: list[torch.nn.Parameter] = []
        params.extend(self.world_model.parameters())
        params.extend(self.empowerment.parameters())
        params.extend(self.actor.parameters())
        params.extend(self.critic.parameters())
        if self.self_state_encoder is not None:
            params.extend(self.self_state_encoder.parameters())
        if self.self_state_predictor is not None:
            params.extend(self.self_state_predictor.parameters())
        self.optimizer = torch.optim.Adam(params, lr=config.optimizer_lr)
        self.optimizer_empowerment_weight = config.optimizer_empowerment_weight
        # Blackwell/RTX 5090 workaround: force full precision to avoid
        # cuBLAS crashes triggered by FP16 kernels.
        # Force FP32 globally to avoid cublasGemmEx instability on Blackwell GPUs.
        self.autocast_enabled = False
        self.grad_scaler = _create_grad_scaler(self.device.type, self.autocast_enabled)
        self.novelty_tracker = RunningMeanStd(device=self.device)

    def _autocast_ctx(self) -> ContextManager[None]:
        if not self.autocast_enabled:
            return nullcontext()
        try:
            return torch.amp.autocast(device_type=self.device.type)
        except AttributeError:
            from torch.cuda.amp import autocast as legacy_autocast  # type: ignore[attr-defined]

            return legacy_autocast()

    def _compute_horizon_multiplier(self, normalized_deficit: float) -> float:
        normalized = max(0.0, min(1.0, float(normalized_deficit)))
        max_multiplier = max(1.0, float(self.config.max_horizon_multiplier))
        mode = str(getattr(self.config, "horizon_scaling_mode", "sigmoid")).lower()
        if mode == "linear":
            scaled = normalized
        else:
            k = 10.0
            midpoint = 0.5
            scaled = 1.0 / (1.0 + math.exp(-k * (normalized - midpoint)))
        return 1.0 + (max_multiplier - 1.0) * scaled

    def _call_with_fallback(
        self, attr: str, *args: P.args, module_override: nn.Module | None = None, **kwargs: P.kwargs
    ) -> T:
        module = cast(_CallableModule[P, T], module_override if module_override is not None else getattr(self, attr))
        amp_ctx = nullcontext()
        if attr == "world_model" and self.autocast_enabled and self.device.type == "cuda":
            try:
                amp_ctx = torch.amp.autocast(device_type=self.device.type, enabled=False)
            except AttributeError:  # pragma: no cover - legacy AMP
                from torch.cuda.amp import autocast as legacy_autocast

                amp_ctx = legacy_autocast(enabled=False)
        try:
            with amp_ctx:
                return module(*args, **kwargs)
        except Exception as err:
            message = str(err).lower()
            is_oom = isinstance(err, torch.cuda.OutOfMemoryError) or "out of memory" in message
            needs_fallback = is_oom or any(
                snippet in message
                for snippet in (
                    "symbolically trace a dynamo-optimized function",
                    "autotuner",
                    "triton",
                )
            )
            original_module = cast(
                _CallableModule[P, T] | None, getattr(module, "_orig_mod", None)
            )
            if not needs_fallback or original_module is None:
                raise
            if is_oom and torch.cuda.is_available():
                torch.cuda.empty_cache()
            original_module = original_module.to(self.device)
            original_module.train(module.training)
            setattr(self, attr, original_module)
            try:
                import torch._dynamo as _dynamo  # type: ignore[attr-defined]

                _dynamo.reset()
            except Exception:
                pass
            with amp_ctx:
                return cast(T, original_module(*args, **kwargs))

    def step(
        self,
        observation: torch.Tensor,
        action: torch.Tensor | None = None,
        next_observation: torch.Tensor | None = None,
        self_state: torch.Tensor | None = None,
        train: bool = False,
        actor_model: nn.Module | None = None,
        world_model: nn.Module | None = None,
    ) -> StepResult:
        """Perform a single interaction step and optional training update."""

        if not train and actor_model is None and hasattr(self, "_step_lock") and self._step_lock is not None:
            lock_ctx: ContextManager[Any] = self._step_lock  # type: ignore[assignment]
        else:
            lock_ctx = nullcontext()
        with lock_ctx:
            return self._step_impl(
                observation,
                action=action,
                next_observation=next_observation,
                self_state=self_state,
                train=train,
                actor_model=actor_model,
                world_model=world_model,
            )

    def _step_impl(
        self,
        observation: torch.Tensor,
        action: torch.Tensor | None = None,
        next_observation: torch.Tensor | None = None,
        self_state: torch.Tensor | None = None,
        train: bool = False,
        actor_model: nn.Module | None = None,
        world_model: nn.Module | None = None,
    ) -> StepResult:
        """Internal implementation of :meth:`step` without locking."""
        observation = observation.to(self.device, non_blocking=True)
        batch = observation.size(0)
        state_tensor: torch.Tensor | None
        if self.self_state_dim > 0:
            if self_state is None:
                state_tensor = torch.zeros(
                    batch,
                    self.self_state_dim,
                    device=self.device,
                    dtype=observation.dtype,
                )
            else:
                state_tensor = self_state.to(self.device, non_blocking=True)
                if state_tensor.ndim == 1:
                    state_tensor = state_tensor.unsqueeze(0)
                if state_tensor.size(0) != batch:
                    if state_tensor.size(0) == 1:
                        state_tensor = state_tensor.expand(batch, -1)
                    else:
                        raise ValueError("self_state batch dimension mismatch")
        else:
            state_tensor = None

        if state_tensor is not None:
            self._latest_self_state = state_tensor.detach()

        if (
            self.current_temporal_state is None
            or self.current_temporal_state["field_tensor"].size(0) != batch
        ):
            self.current_temporal_state = self.temporal_self.get_default_state(
                batch_size=batch,
                device=self.device,
                dtype=observation.dtype,
            )
        temporal_state_prev = self.current_temporal_state
        cognitive_mode = temporal_state_prev.get("cognitive_mode", "LEARNING")
        temporal_field = temporal_state_prev["field_tensor"]
        stimulus_deficit = float(temporal_state_prev.get("stimulus_deficit", 0.0))

        competence_breakdown: dict[str, torch.Tensor] | None = None
        epistemic_novelty: torch.Tensor | None = None
        current_real_entropy: float | None = None
        latent_snapshot: LatentSnapshot | None = None

        restore_world_model = False
        target_wm = world_model if world_model is not None else self.world_model
        if world_model is None and not train and self.world_model.training:
            self.world_model.eval()
            restore_world_model = True

        try:
            with torch.no_grad():
                with self._autocast_ctx():
                    amp_ctx_disable = nullcontext()
                    if self.autocast_enabled and self.device.type == "cuda":
                        try:
                            amp_ctx_disable = torch.amp.autocast(
                                device_type=self.device.type, enabled=False
                            )
                        except AttributeError:  # pragma: no cover - legacy fallback
                            from torch.cuda.amp import autocast as legacy_autocast

                            amp_ctx_disable = legacy_autocast(enabled=False)
                    with amp_ctx_disable:
                        latents = self._call_with_fallback("world_model", observation, module_override=world_model)
                    memory_context = self._get_memory_context(latents["z_self"])
                if action is not None:
                    action_for_routing = action.to(self.device, non_blocking=True)
                else:
                    action_for_routing = torch.zeros(
                        batch, self.config.dynamics.action_dim, device=self.device
                    )
                (
                    broadcast,
                    scores,
                    slot_novelty,
                    slot_progress,
                    slot_cost,
                ) = self._route_slots(
                    latents["slots"],
                    latents["z_self"],
                    action_for_routing,
                    state_tensor,
                    update_stats=True,
                    cognitive_mode=cognitive_mode,
                )
                features = self._assemble_features(
                    latents["z_self"], broadcast, memory_context, temporal_field
                )
                if action is None:
                    if actor_model is not None:
                        # Use the thread-local actor snapshot if provided (lock-free path)
                        action_dist = actor_model(features)
                    else:
                        actor_attr = "actor" if train else "actor_eval"
                        action_dist = self._call_with_fallback(actor_attr, features)
                    base_entropy = action_dist.entropy()
                    actor_boost = 1.0 + (
                        stimulus_deficit * self.config.temporal_self.actor_entropy_scale
                    )
                    if train and actor_boost > 1.0:
                        action = action_dist.rsample()
                        noise_scale = max(0.0, (actor_boost - 1.0) * 0.1)
                        if noise_scale > 0.0:
                            action = action + torch.randn_like(action) * noise_scale
                        action = action.clamp(-10.0, 10.0)
                    else:
                        action = action_dist.rsample()

                    real_action_entropy = base_entropy.mean()
                    if not hasattr(self, "_real_entropy_buffer"):
                        self._real_entropy_buffer: list[float] = []
                    entropy_value = float(real_action_entropy.item())
                    self._real_entropy_buffer.append(entropy_value)
                    if len(self._real_entropy_buffer) > 200:
                        self._real_entropy_buffer = self._real_entropy_buffer[-200:]
                    if len(self._real_entropy_buffer) >= 100:
                        recent_window = self._real_entropy_buffer[-100:]
                        avg_entropy = sum(recent_window) / len(recent_window)
                        if self._step_count % 100 == 0:
                            print(f"[Real Policy Entropy] Mean: {avg_entropy:.3f}")
                    current_real_entropy = entropy_value

                    (
                        broadcast,
                        scores,
                        slot_novelty,
                        slot_progress,
                        slot_cost,
                    ) = self._route_slots(
                        latents["slots"],
                        latents["z_self"],
                        action,
                        state_tensor,
                        update_stats=False,
                        cognitive_mode=cognitive_mode,
                    )
                    features = self._assemble_features(
                        latents["z_self"], broadcast, memory_context, temporal_field
                    )
                else:
                    action = action.to(self.device, non_blocking=True)

                latent_state = broadcast.mean(dim=1)
                latent_snapshot = LatentSnapshot(
                    z_self=latents["z_self"].detach(),
                    slots=latents["slots"].detach(),
                    latent_state=latent_state.detach(),
                )
                predictions = target_wm.predict_next_latents(latent_state, action)
                decoded = target_wm.decode_predictions(
                    predictions, use_frozen=True, sample=target_wm.training
                )
                novelty_mix = None
                if (
                    self.current_temporal_state is not None
                    and "novelty_mix" in self.current_temporal_state
                ):
                    novelty_mix = self.current_temporal_state["novelty_mix"]
                novelty = self.reward.get_novelty(
                    predictions, decoded, novelty_mix=novelty_mix
                ).to(self.device)
                observation_entropy = estimate_observation_entropy(observation)

                if not torch.isfinite(novelty).all():
                    finite = novelty[torch.isfinite(novelty)]
                    if finite.numel() > 0:
                        min_val = float(finite.min().item())
                        max_val = float(finite.max().item())
                        mean_val = float(finite.mean().item())
                    else:
                        min_val = max_val = mean_val = float('nan')
                    print(f"[STEP {self._step_count}] ERROR: NaN/Inf in novelty!")
                    print(f"  novelty stats: min={min_val}, max={max_val}, mean={mean_val}")
                    novelty = sanitize_tensor(novelty, replacement=0.0)

                epistemic_novelty = novelty

                if not torch.isfinite(observation_entropy).all():
                    finite = observation_entropy[torch.isfinite(observation_entropy)]
                    if finite.numel() > 0:
                        min_val = float(finite.min().item())
                        max_val = float(finite.max().item())
                    else:
                        min_val = max_val = float('nan')
                    print(f"[STEP {self._step_count}] ERROR: NaN/Inf in observation_entropy!")
                    print(f"  entropy stats: min={min_val}, max={max_val}")
                    observation_entropy = sanitize_tensor(observation_entropy, replacement=0.1)

                if not torch.isfinite(action).all():
                    print(f"[STEP {self._step_count}] ERROR: NaN/Inf in sampled action!")
                    action = sanitize_tensor(action, replacement=0.0)

                self.current_temporal_state = self.temporal_self(
                    temporal_state_prev, novelty
                )
                self.reward._step_count = self._step_count
                intrinsic_raw, norm_components, raw_components = self.reward.get_intrinsic_reward(
                    temporal_state=self.current_temporal_state,
                    novelty_batch=novelty,
                    observation_entropy=observation_entropy,
                    action=action,
                    latent=latent_state,
                    self_state=state_tensor,
                    return_components=True,
                )

                last_breakdown = self.temporal_self.last_competence_breakdown
                competence_breakdown = last_breakdown if last_breakdown is not None else {}
        finally:
            if restore_world_model:
                self.world_model.train()

        intrinsic = self.reward_normalizer(intrinsic_raw)
        reward_components = {key: value.detach() for key, value in norm_components.items()}
        raw_reward_components = {key: value.detach() for key, value in raw_components.items()}

        if train and next_observation is not None:
            self.store_transition(
                observation=observation,
                action=action,
                next_observation=next_observation,
                self_state=state_tensor,
            )
        return StepResult(
            action=action.detach(),
            intrinsic_reward=intrinsic.detach(),
            novelty=slot_novelty.detach(),
            observation_entropy=observation_entropy.detach(),
            slot_scores=scores.detach(),
            reward_components=reward_components,
            raw_reward_components=raw_reward_components,
            competence_breakdown=competence_breakdown,
            epistemic_novelty=epistemic_novelty.detach() if epistemic_novelty is not None else None,
            real_action_entropy=current_real_entropy,
            latent_snapshot=latent_snapshot,
        )

    def store_transition(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor,
        self_state: torch.Tensor | None = None,
    ) -> None:
        """Store a transition into the rollout buffer on CPU memory.

        Args:
            observation: Observation tensor at time ``t``.
            action: Action tensor taken at time ``t``.
            next_observation: Observation tensor at time ``t + 1``.
            self_state: Optional self-state tensor aligned with the transition.
        """
        obs_cpu = observation.detach().to("cpu", non_blocking=True).contiguous()
        act_cpu = action.detach().to("cpu", non_blocking=True).contiguous()
        next_cpu = next_observation.detach().to("cpu", non_blocking=True).contiguous()
        state_cpu = (
            self_state.detach().to("cpu", non_blocking=True).contiguous()
            if self_state is not None
            else None
        )
        if torch.cuda.is_available():
            obs_cpu = obs_cpu.pin_memory()
            act_cpu = act_cpu.pin_memory()
            next_cpu = next_cpu.pin_memory()
            if state_cpu is not None:
                state_cpu = state_cpu.pin_memory()

        batch_items = obs_cpu.shape[0]
        for idx in range(batch_items):
            self.rollout_buffer.push(
                obs_cpu[idx],
                act_cpu[idx],
                next_cpu[idx],
                state_cpu[idx] if state_cpu is not None else None,
            )

    def store_latent_transition(
        self,
        snapshot: LatentSnapshot | None,
        action: torch.Tensor,
        next_observation: torch.Tensor,
        world_model: nn.Module | None = None,
    ) -> None:
        """Persist latent transitions for episodic replay."""

        if snapshot is None:
            return

        batch = action.shape[0]
        if snapshot.z_self.size(0) != batch:
            raise ValueError("latent snapshot and action batch sizes must match")

        with torch.no_grad():
            with self._autocast_ctx():
                next_obs = next_observation.to(self.device, non_blocking=True)
                next_latents = self._call_with_fallback("world_model", next_obs, module_override=world_model)
        target_latent = next_latents["slots"].mean(dim=1)
        self._write_memory(snapshot, action=action, target_latent=target_latent)

    def _route_slots(
        self,
        slot_values: torch.Tensor,
        z_self: torch.Tensor,
        action: torch.Tensor,
        self_state: torch.Tensor | None,
        update_stats: bool,
        cognitive_mode: str = "LEARNING",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        slot_novelty = slot_values.var(dim=-1, unbiased=False)
        if update_stats:
            self.novelty_tracker.update(slot_novelty.mean(dim=-1))
        if self._slot_baseline is None:
            self._slot_baseline = slot_values.mean(dim=0).detach()

        if self._novelty_trace is None:
            slot_progress = torch.zeros_like(slot_novelty)
            if update_stats:
                self._novelty_trace = slot_novelty.detach()
        else:
            prev_trace = self._novelty_trace
            slot_progress = prev_trace - slot_novelty
            if update_stats:
                updated_trace = (
                    (1 - self.progress_momentum) * self._novelty_trace
                    + self.progress_momentum * slot_novelty.detach()
                )
                self._novelty_trace = updated_trace

        if update_stats:
            baseline_update = slot_values.mean(dim=0).detach()
            self._slot_baseline = (
                (1 - self.progress_momentum) * self._slot_baseline
                + self.progress_momentum * baseline_update
            )

        action_cost = torch.norm(action, dim=-1, keepdim=True) * self.action_cost_scale
        slot_cost = action_cost.expand(-1, slot_values.size(1))

        slot_norm = torch.nn.functional.normalize(slot_values, dim=-1)
        z_self_norm = torch.nn.functional.normalize(z_self, dim=-1)
        self_similarity = (
            slot_norm * z_self_norm.unsqueeze(1)
        ).sum(dim=-1).clamp(min=0.0)

        state_similarity = torch.zeros_like(self_similarity)
        if (
            self_state is not None
            and self.self_state_encoder is not None
            and self.self_state_dim > 0
        ):
            projected_state = self.self_state_encoder(
                self_state.to(self.device, non_blocking=True)
            )
            projected_state = torch.nn.functional.normalize(projected_state, dim=-1)
            state_similarity = (
                slot_norm * projected_state.unsqueeze(1)
            ).sum(dim=-1).clamp(min=0.0)

        self_mask = torch.clamp(self_similarity + state_similarity, min=0.0)

        batch_mean = slot_novelty.mean(dim=0).detach()
        if self._ucb_mean is None or self._ucb_counts is None:
            self._ucb_mean = batch_mean.clone()
            self._ucb_counts = torch.ones_like(batch_mean)
        elif update_stats:
            self._ucb_counts = self._ucb_counts + 1
            self._ucb_mean = self._ucb_mean + (batch_mean - self._ucb_mean) / self._ucb_counts
        assert self._ucb_mean is not None and self._ucb_counts is not None
        self._step_count += 1
        ucb_bonus = (
            self._ucb_mean.to(self.device)
            + self.config.workspace.ucb_beta
            * torch.sqrt(
                torch.log1p(torch.tensor(float(self._step_count), device=self.device))
                / self._ucb_counts.to(self.device)
            )
        )
        ucb = ucb_bonus.unsqueeze(0).expand(slot_values.size(0), -1)

        scores = self.workspace.score_slots(
            novelty=slot_novelty,
            progress=slot_progress,
            ucb=ucb,
            cost=slot_cost,
            self_mask=self_mask,
            cognitive_mode=cognitive_mode,
        )
        broadcast = self.workspace.broadcast(slot_values, scores=scores)
        return broadcast, scores, slot_novelty, slot_progress, slot_cost

    def _get_memory_context(self, keys: torch.Tensor) -> torch.Tensor:
        batch = keys.shape[0]
        if len(self.memory) == 0:
            return torch.zeros(batch, self.memory.config.key_dim, device=self.device)
        _, values = self.memory.read(keys)
        context = values[:, 0, :].to(self.device)
        return context

    def _write_memory(
        self,
        snapshot: LatentSnapshot,
        *,
        action: torch.Tensor | None = None,
        target_latent: torch.Tensor | None = None,
    ) -> None:
        key = snapshot.z_self.detach().to("cpu", non_blocking=True)
        context_value = snapshot.latent_state.detach().to("cpu", non_blocking=True)
        slots = snapshot.slots.detach().to("cpu", non_blocking=True)
        action_cpu = (
            action.detach().to("cpu", non_blocking=True) if action is not None else None
        )
        target_cpu = (
            target_latent.detach().to("cpu", non_blocking=True)
            if target_latent is not None
            else None
        )
        payload = EpisodicSnapshot(
            z_self=key, slots=slots, action=action_cpu, target_latent=target_cpu
        )
        self.memory.write(key, context_value, snapshot=payload)

    def refresh_policy_snapshot(self) -> None:
        """Update the inference policy copy with the latest trained weights."""

        with torch.no_grad():
            self.actor_eval.load_state_dict(module_state_dict(self.actor))
            self.actor_eval.eval()

    def _emergency_reset_if_corrupted(self) -> bool:
        """Reset critical parameters that have become non-finite.

        Returns:
            bool: ``True`` if a reset was required and performed, ``False`` otherwise.
        """

        # Empowerment temperature is a single learnable parameter that previously
        # caused NaNs to cascade through the training loop. Reset it aggressively.
        if hasattr(self.empowerment, "temperature"):
            temperature = getattr(self.empowerment, "temperature")
            if isinstance(temperature, torch.Tensor) and not torch.isfinite(temperature).all():
                print(f"[STEP {self._step_count}] 🚨 EMERGENCY: Resetting empowerment temperature")
                with torch.no_grad():
                    temperature.copy_(
                        torch.tensor(
                            self.config.empowerment.temperature,
                            device=temperature.device,
                            dtype=temperature.dtype,
                        )
                    )
                return True

        decoder = getattr(self.world_model, "decoder", None)
        if decoder is not None and hasattr(decoder, "log_std"):
            log_std = getattr(decoder, "log_std")
            if isinstance(log_std, torch.Tensor) and not torch.isfinite(log_std).all():
                print(f"[STEP {self._step_count}] 🚨 EMERGENCY: Resetting decoder log_std")
                with torch.no_grad():
                    log_std.copy_(
                        torch.full_like(log_std, self.config.decoder.init_log_std)
                    )
                return True

        return False

    def _check_parameter_health(self) -> bool:
        """Check all trainable parameters for non-finite values."""

        components: dict[str, nn.Module] = {
            "world_model": self.world_model,
            "empowerment": self.empowerment,
            "actor": self.actor,
            "critic": self.critic,
        }
        if self.self_state_encoder is not None:
            components["self_state_encoder"] = self.self_state_encoder
        if self.self_state_predictor is not None:
            components["self_state_predictor"] = self.self_state_predictor

        corrupted: list[str] = []
        for prefix, module in components.items():
            for name, param in module.named_parameters():
                if param.requires_grad and param.data is not None:
                    if not torch.isfinite(param.data).all():
                        corrupted.append(f"{prefix}.{name}")

        if corrupted:
            print(f"[STEP {self._step_count}] 🚨 CORRUPTED PARAMETERS:")
            for name in corrupted:
                print(f"  - {name}")
            return False
        return True

    def _assemble_features(
        self,
        z_self: torch.Tensor,
        broadcast: torch.Tensor,
        memory_context: torch.Tensor,
        temporal_field: torch.Tensor,
    ) -> torch.Tensor:
        broadcast_flat = broadcast.flatten(start_dim=1).to(dtype=z_self.dtype)
        temporal = temporal_field
        if temporal.size(0) != z_self.size(0):
            temporal = torch.zeros(
                z_self.size(0),
                temporal_field.size(-1),
                device=z_self.device,
                dtype=z_self.dtype,
            )
        else:
            temporal = temporal.to(device=z_self.device, dtype=z_self.dtype)
        memory_aligned = memory_context.to(device=z_self.device, dtype=z_self.dtype)
        return torch.cat([z_self, broadcast_flat, memory_aligned, temporal], dim=-1)

    def _optimize(self) -> tuple[int, dict[str, float]] | None:
        if len(self.rollout_buffer) < self.batch_size:
            return None

        if self._emergency_reset_if_corrupted():
            print("[TRAINING] Parameters were corrupted and reset, skipping this update")
            return None

        if not self._check_parameter_health():
            print("[TRAINING] Skipping update due to corrupted parameters")
            return None
        temporal_context = self.current_temporal_state
        if temporal_context is None:
            temporal_context = self.temporal_self.get_default_state(
                batch_size=self.batch_size,
                device=self.device,
                dtype=torch.float32,
            )
        lr_scale = float(temporal_context.get("learning_rate_scale", 1.0))
        current_lr = self.config.optimizer_lr * lr_scale
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr
        observations, actions, next_observations, self_states = self.rollout_buffer.sample(
            self.batch_size
        )
        observations = observations.to(self.device, non_blocking=True)
        actions = actions.to(self.device, non_blocking=True)
        next_observations = next_observations.to(self.device, non_blocking=True)
        if self_states is not None:
            self_states = self_states.to(self.device, non_blocking=True)

        replay_batch = max(1, self.batch_size // 2)
        replay_samples: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        if len(self.memory) > 0:
            try:
                replay_samples = self.memory.sample_transitions(
                    replay_batch, device=self.device
                )
            except ValueError:
                replay_samples = None

        self.optimizer.zero_grad(set_to_none=True)
        try:
            emp_diag: dict[str, float] | None = None
            replay_loss_value: torch.Tensor | None = None

            with self._autocast_ctx():
                latents = self._call_with_fallback("world_model", observations)

                for key, tensor in latents.items():
                    if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all():
                        print(
                            f"[STEP {self._step_count}] NaN in {key} from encoder! Skipping update."
                        )
                        return None
                memory_context = self._get_memory_context(latents["z_self"])
                broadcast, _, _, _, _ = self._route_slots(
                    latents["slots"],
                    latents["z_self"],
                    actions,
                    self_states,
                    update_stats=False,
                    cognitive_mode=temporal_context.get("cognitive_mode", "LEARNING"),
                )
                latent_state = broadcast.mean(dim=1)
                temporal_features = temporal_context.get("field_tensor")
                if (
                    temporal_features is None
                    or temporal_features.size(0) != latents["z_self"].size(0)
                ):
                    temporal_features = torch.zeros(
                        latents["z_self"].size(0),
                        self.temporal_self.field_dim,
                        device=self.device,
                        dtype=latents["z_self"].dtype,
                    )
                else:
                    temporal_features = temporal_features.to(
                        device=self.device, dtype=latents["z_self"].dtype
                    )
                features = self._assemble_features(
                    latents["z_self"], broadcast, memory_context, temporal_features
                )

                predictions = self.world_model.predict_next_latents(latent_state, actions)
                decoded = self.world_model.decode_predictions(
                    predictions,
                    use_frozen=False,
                    sample=self.world_model.training,
                )
                log_likelihoods = torch.stack(
                    [dist.log_prob(next_observations).mean() for dist in decoded]
                )
                frontier_loss = -log_likelihoods.mean()

                encoded_next = self._call_with_fallback("world_model", next_observations)
                mean_latents = torch.stack([pred.mean for pred in predictions], dim=0)
                predicted_latent = mean_latents.mean(dim=0)
                target_latent = encoded_next["slots"].mean(dim=1)
                latent_alignment = torch.nn.functional.mse_loss(predicted_latent, target_latent)
                frontier_loss = frontier_loss + 0.1 * latent_alignment

                replay_loss = torch.tensor(0.0, device=self.device)
                if replay_samples is not None:
                    replay_latent, replay_actions, replay_targets = replay_samples
                    replay_predictions = self.world_model.predict_next_latents(
                        replay_latent, replay_actions
                    )
                    replay_mean = torch.stack(
                        [dist.mean for dist in replay_predictions], dim=0
                    ).mean(dim=0)
                    replay_loss = torch.nn.functional.mse_loss(replay_mean, replay_targets)

                replay_loss_value = replay_loss

                if replay_samples is not None:
                    world_model_loss = 0.5 * (frontier_loss + replay_loss)
                else:
                    world_model_loss = frontier_loss
                world_model_loss = world_model_loss * self.config.world_model_coef

                self_state_loss = torch.tensor(0.0, device=self.device)
                if (
                    self_states is not None
                    and self.self_state_dim > 0
                    and self.self_state_predictor is not None
                ):
                    z_self_float = latents["z_self"].float()
                    predicted_state = self.self_state_predictor(z_self_float)
                    self_state_loss = torch.nn.functional.mse_loss(predicted_state, self_states)
                    self_state_loss = self.config.workspace.self_bias * self_state_loss

                if hasattr(self.empowerment, "get_queue_diagnostics"):
                    emp_diag = self.empowerment.get_queue_diagnostics()
                    if self._step_count % 1000 == 0:
                        print(
                            f"[Empowerment Queue] Size: {emp_diag['queue_size']}, "
                            f"Diversity: {emp_diag['queue_diversity']:.4f}"
                        )

                real_stimulus_deficit = float(
                    temporal_context.get("stimulus_deficit", 0.0)
                )
                dream_loss, actor_loss, critic_loss, dream_metrics = self._stable_dreaming(
                    latents, real_stimulus_deficit
                )
                total_loss = world_model_loss + actor_loss + critic_loss + dream_loss + self_state_loss

            loss_components = {
                "world_model": world_model_loss,
                "actor": actor_loss,
                "critic": critic_loss,
                "dream": dream_loss,
                "self_state": self_state_loss,
                "total": total_loss,
            }

            valid_step = True
            for name, loss_val in loss_components.items():
                if not torch.isfinite(loss_val):
                    print(
                        f"[STEP {self._step_count}] ⚠️ WARNING: Non-finite {name}_loss: {loss_val}"
                    )
                    loss_components[name] = torch.tensor(0.0, device=self.device)
                    valid_step = False

            world_model_loss = loss_components["world_model"]
            actor_loss = loss_components["actor"]
            critic_loss = loss_components["critic"]
            dream_loss = loss_components["dream"]
            self_state_loss = loss_components["self_state"]
            total_loss = loss_components["total"]

            if not valid_step:
                print("  Skipping gradient step due to NaN/Inf losses.")
                total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            else:
                # FIX: Blackwell/RTX 5090 workaround
                # The backward pass for Linear layers can trigger cublasGemmEx crashes in FP16.
                # We force the backward pass to run in FP32 by disabling autocast.
                if self.autocast_enabled and self.device.type == "cuda":
                    try:
                        backward_ctx = torch.amp.autocast(device_type=self.device.type, enabled=False)
                    except AttributeError:  # pragma: no cover - legacy AMP
                        from torch.cuda.amp import autocast as legacy_autocast
                        backward_ctx = legacy_autocast(enabled=False)
                else:
                    backward_ctx = nullcontext()

                with backward_ctx:
                    self.grad_scaler.scale(total_loss).backward()

                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)

                bad_grads = sanitize_gradients(self.world_model)
                bad_grads += sanitize_gradients(self.actor)
                bad_grads += sanitize_gradients(self.critic)
                bad_grads += sanitize_gradients(self.empowerment)
                if bad_grads > 0:
                    print(
                        f"[STEP {self._step_count}] WARNING: Sanitized {bad_grads} non-finite gradients"
                    )

                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()

            if (
                self._step_count - self._last_snapshot_sync
                >= self._policy_snapshot_interval
            ):
                self.refresh_policy_snapshot()
                self._last_snapshot_sync = self._step_count

            if self._step_count > 0 and self._step_count % 1000 == 0:
                self.world_model.refresh_frozen_decoder()
                if self._step_count % 5000 == 0:
                    print(f"[Step {self._step_count}] Refreshed frozen decoder")

            metrics: dict[str, float] = {
                "train/total_loss": float(total_loss.detach().cpu().item()),
                "train/world_model_loss": float(world_model_loss.detach().cpu().item()),
                "train/actor_loss": float(actor_loss.detach().cpu().item()),
                "train/critic_loss": float(critic_loss.detach().cpu().item()),
                "train/dream_loss_empowerment": float(dream_loss.detach().cpu().item()),
                "train/self_state_loss": float(self_state_loss.detach().cpu().item()),
                "temporal/stimulus_level": float(
                    temporal_context.get("stimulus_level", 0.0)
                ),
                "temporal/stimulus_deficit": float(
                    temporal_context.get("stimulus_deficit", 0.0)
                ),
                "temporal/learning_rate_scale": float(lr_scale),
            }
            if replay_loss_value is not None:
                metrics["train/world_model_replay_loss"] = float(
                    replay_loss_value.detach().cpu().item()
                )
            if emp_diag is not None:
                metrics["debug/empowerment_queue_size"] = float(emp_diag["queue_size"])
                metrics["debug/empowerment_queue_diversity"] = float(emp_diag["queue_diversity"])
            for key, value in dream_metrics.items():
                if isinstance(value, torch.Tensor):
                    metrics[key] = float(value.detach().cpu().item())
                else:
                    metrics[key] = float(value)
            return self._step_count, metrics
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                f"[STEP {self._step_count}] WARNING: CUDA OOM during optimization; skipping this step."
            )
            return None

    def _stable_dreaming(
        self,
        latents: dict[str, torch.Tensor],
        real_stimulus_deficit: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        dynamic_noise_ratio = (
            self.config.dream_noise_base_ratio
            + real_stimulus_deficit * self.config.temporal_self.dream_noise_scale
        )
        dynamic_noise_ratio = min(1.0, max(0.0, dynamic_noise_ratio))
        dynamic_counterfactual_rate = (
            self.config.dream_counterfactual_base_rate
            + real_stimulus_deficit
            * self.config.temporal_self.dream_counterfactual_scale
        )
        dynamic_counterfactual_rate = min(1.0, max(0.0, dynamic_counterfactual_rate))

        chunk_size = max(1, self.config.dream_chunk_size)
        base_horizon = max(chunk_size, int(self.config.base_dream_horizon))
        if real_stimulus_deficit <= self.config.boredom_threshold:
            horizon_multiplier = 1.0
        else:
            denom = max(1e-6, 1.0 - float(self.config.boredom_threshold))
            normalized_deficit = (real_stimulus_deficit - float(self.config.boredom_threshold)) / denom
            horizon_multiplier = self._compute_horizon_multiplier(normalized_deficit)
        dream_horizon = max(chunk_size, int(round(base_horizon * horizon_multiplier)))
        num_chunks = max(1, math.ceil(dream_horizon / chunk_size))

        initial_latents = {
            key: value.detach().clone()
            if isinstance(value, torch.Tensor)
            else value
            for key, value in latents.items()
        }
        memory_rate = float(getattr(self.config, "dream_from_memory_rate", 0.0))
        if memory_rate > 0.0 and len(self.memory) > 0:
            dream_batch = initial_latents["z_self"].shape[0]
            seed_count = max(1, int(round(dream_batch * memory_rate)))
            seed_count = min(dream_batch, seed_count)
            try:
                seed_z, seed_slots = self.memory.sample_uniform(
                    seed_count, device=self.device
                )
            except ValueError:
                seed_z = None
                seed_slots = None
            if seed_z is not None and seed_slots is not None:
                perm = torch.randperm(dream_batch, device=self.device)[:seed_count]
                initial_latents["z_self"][perm] = seed_z.to(
                    device=self.device, dtype=initial_latents["z_self"].dtype
                )
                slot_tensor = initial_latents["slots"]
                slot_tensor[perm] = seed_slots.to(
                    device=self.device, dtype=slot_tensor.dtype
                )

        # Latent drift injection removed per DTC 3.0 spec (Aleatoric Subtraction)
        # Rely on ensemble output noise (rsample) instead of input noise.
        memory_context = self._get_memory_context(initial_latents["z_self"]).detach()
        dream_self_state: torch.Tensor | None = None
        if self._latest_self_state is not None:
            dream_self_state = self._latest_self_state.to(
                self.device, non_blocking=True
            )
        current_latents = initial_latents

        entropies: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        log_probs: list[torch.Tensor] = []
        competence_terms = []
        empowerment_terms = []
        survival_terms = []
        intrinsic_terms = []
        explore_terms = []
        raw_explore_terms = []
        dream_novelty_values: list[float] = []

        dream_actions = []
        counterfactual_rates: list[torch.Tensor] = []
        latent_drift_norms: list[torch.Tensor] = []
        ensemble_noise_base = 0.05
        temporal_snapshot = self.temporal_self.snapshot()
        dream_temporal_state = self.temporal_self.get_default_state(
            batch_size=initial_latents["z_self"].size(0),
            device=self.device,
            dtype=initial_latents["z_self"].dtype,
        )
        last_entropy_boost = 1.0
        try:
            for _ in range(num_chunks):
                for _ in range(chunk_size):
                    dream_mode = dream_temporal_state.get("cognitive_mode", "LEARNING")
                    temporal_field = dream_temporal_state["field_tensor"]
                    stimulus_deficit_prev = float(
                        dream_temporal_state.get("stimulus_deficit", 0.0)
                    )
                    broadcast, _, _, _, _ = self._route_slots(
                        current_latents["slots"],
                        current_latents["z_self"],
                        torch.zeros(
                            current_latents["slots"].size(0),
                            self.config.dynamics.action_dim,
                            device=self.device,
                        ),
                        None,
                        update_stats=False,
                        cognitive_mode=dream_mode,
                    )
                    features = self._assemble_features(
                        current_latents["z_self"], broadcast, memory_context, temporal_field
                    )
                    action_dist = self._call_with_fallback("actor", features)
                    sampled_action = action_dist.rsample()
                    dream_log_prob = action_dist.log_prob(sampled_action)
                    dream_entropy = action_dist.entropy()

                    entropy_boost = 1.0 + (
                        stimulus_deficit_prev * self.config.temporal_self.dream_entropy_scale
                    )
                    last_entropy_boost = entropy_boost
                    base_noise_scale = 0.15
                    action_noise = torch.randn_like(sampled_action) * base_noise_scale
                    mutation_scale = 0.1 * entropy_boost
                    policy_mutation = torch.randn_like(sampled_action) * mutation_scale
                    mutated_action = sampled_action + action_noise + policy_mutation
                    counterfactual_mask = (
                        torch.rand(sampled_action.shape[0], device=sampled_action.device)
                        < dynamic_counterfactual_rate
                    )
                    wild_action = torch.randn_like(sampled_action) * 2.0
                    executed_action = torch.where(
                        counterfactual_mask.unsqueeze(-1), wild_action, mutated_action
                    )
                    executed_action = executed_action.clamp(-3.0, 3.0)
                    dream_actions.append(executed_action)
                    counterfactual_rates.append(counterfactual_mask.float().mean())

                    latent_state = broadcast.mean(dim=1)
                    latent_drift_norms.append(torch.tensor(0.0, device=latent_state.device))

                    # 1. Predict "Concept" (Latent State) directly
                    predictions = self.world_model.predict_next_latents(latent_state, executed_action)

                    # 2. Calculate "Conceptual Novelty" (Pure Epistemic Disagreement)
                    # We skip the expensive decoder entirely. Novelty is just ensemble variance.
                    # Each prediction is a Normal(mu, sigma). We look at var(mu).
                    stacked_means = torch.stack([p.mean for p in predictions])
                    novelty = stacked_means.var(dim=0).mean(dim=-1)

                    # 3. Latent Carryover (The "Aphantasia" Trick)
                    # Instead of decoding to pixels and re-encoding, we sample the consensus latent
                    # and assume it is truth.
                    if self.world_model.training:
                         # Sample from ensemble to maintain aleatoric uncertainty awareness
                        next_latent_mean = torch.stack([p.rsample() for p in predictions]).mean(dim=0)
                    else:
                        next_latent_mean = stacked_means.mean(dim=0)

                    # 4. "Phantom Slot" Expansion
                    # The Actor expects [Batch, Num_Slots, Dim]. We have [Batch, Dim].
                    # We broadcast the concept to all slots.
                    num_slots = current_latents["slots"].shape[1]
                    phantom_slots = next_latent_mean.unsqueeze(1).expand(-1, num_slots, -1)

                    # 5. Update State for next step
                    # Persist self-state (z_self) as we assume it's stable over the short dream horizon
                    current_latents = {
                        "z_self": current_latents["z_self"],
                        "slots": phantom_slots
                    }
                    
                    # Dummy entropy since we have no pixels
                    observation_entropy = torch.zeros_like(novelty)

                    dream_novelty_values.append(float(novelty.detach().mean().item()))

                    if not torch.isfinite(novelty).all() or not torch.isfinite(observation_entropy).all():
                        print(
                            f"[DREAM ERROR at step {self._step_count}] Non-finite dream values detected, skipping this dream chunk"
                        )
                        return (
                            torch.tensor(0.0, device=self.device),
                            torch.tensor(0.0, device=self.device),
                            torch.tensor(0.0, device=self.device),
                            {
                                k: torch.tensor(0.0, device=self.device)
                                for k in [
                                    "dream/intrinsic_reward",
                                    "dream/competence",
                                    "dream/empowerment",
                                    "dream/survival",
                                    "dream/policy_entropy",
                                    "dream/explore",
                                    "dream/explore_min",
                                    "dream/explore_max",
                                    "dream/explore_raw",
                                    "dream/explore_raw_min",
                                    "dream/explore_raw_max",
                                    "dream/avg_novelty",
                                    "dream/max_novelty_streak",
                                    "dream/horizon",
                                    "dream/horizon_multiplier",
                                ]
                            },
                        )
                    dream_temporal_state = self.temporal_self(
                        dream_temporal_state, novelty
                    )
                    self.reward._step_count = self._step_count
                    dream_reward, norm_components, raw_components = self.reward.get_intrinsic_reward(
                        temporal_state=dream_temporal_state,
                        novelty_batch=novelty,
                        observation_entropy=observation_entropy,
                        action=executed_action,
                        latent=latent_state,
                        self_state=dream_self_state,
                        return_components=True,
                    )
                    competence_terms.append(norm_components["competence"].detach().mean())
                    empowerment_terms.append(norm_components["empowerment"].detach().mean())
                    survival_terms.append(norm_components["survival"].detach().mean())
                    intrinsic_terms.append(dream_reward.detach().mean())
                    explore_terms.append(norm_components["explore"].detach().mean())
                    raw_explore_terms.append(raw_components["explore"].detach().mean())

                    normalized_reward = self.reward_normalizer(dream_reward)

                    critic_value = self._call_with_fallback("critic", features)
                    values.append(critic_value)
                    rewards.append(normalized_reward.detach())
                    log_probs.append(dream_log_prob)
                    entropies.append(dream_entropy)

                    # REMOVE THIS LINE:
                    # current_latents = self._call_with_fallback("world_model", predicted_obs)
                    memory_context = self._get_memory_context(current_latents["z_self"])

                current_latents = {key: value.detach() for key, value in current_latents.items()}
                memory_context = memory_context.detach()
        finally:
            self.temporal_self.restore(temporal_snapshot)

        final_broadcast, _, _, _, _ = self._route_slots(
            current_latents["slots"],
            current_latents["z_self"],
            torch.zeros(
                current_latents["slots"].size(0),
                self.config.dynamics.action_dim,
                device=self.device,
            ),
            None,
            update_stats=False,
            cognitive_mode=dream_temporal_state.get("cognitive_mode", "LEARNING"),
        )
        final_features = self._assemble_features(
            current_latents["z_self"], final_broadcast, memory_context, dream_temporal_state["field_tensor"]
        )
        next_value = self._call_with_fallback("critic", final_features).detach()

        rewards_tensor = torch.stack(rewards)
        values_tensor = torch.stack(values)
        log_probs_tensor = torch.stack(log_probs)
        entropies_tensor = torch.stack(entropies)

        advantages, returns = self._compute_gae(
            rewards_tensor, values_tensor, next_value
        )
        final_stimulus_deficit = float(
            dream_temporal_state.get("stimulus_deficit", 0.0)
        )
        actor_boost = 1.0 + (
            final_stimulus_deficit * self.config.temporal_self.actor_entropy_scale
        )
        dynamic_entropy_coef = self.config.entropy_coef * actor_boost
        if self.config.adaptive_entropy:
            current_avg_novelty = float(self.novelty_tracker.mean.clamp(min=0.0).item())
            if current_avg_novelty < self.config.adaptive_entropy_target:
                novelty_deficit = self.config.adaptive_entropy_target - current_avg_novelty
                boost_scale = 1.0 + (novelty_deficit * self.config.adaptive_entropy_scale)
                boost_scale = max(1.0, min(10.0, boost_scale))
                dynamic_entropy_coef *= boost_scale

        actor_loss = -(
            (advantages.detach() * log_probs_tensor).mean()
            + dynamic_entropy_coef * entropies_tensor.mean()
        )
        critic_loss = (
            self.config.critic_coef * 0.5 * (returns.detach() - values_tensor).pow(2).mean()
        )

        final_action = dream_actions[-1].detach()
        final_latent_state = final_broadcast.mean(dim=1).detach()
        empowerment_term = self.empowerment(final_action, final_latent_state).mean()
        empowerment_term = torch.nan_to_num(
            empowerment_term, nan=0.0, posinf=1e3, neginf=-1e3
        )
        dream_loss = -self.optimizer_empowerment_weight * empowerment_term
        dream_loss = dream_loss.clamp(min=-1e3, max=1e3)

        intrinsic_stack = torch.stack(intrinsic_terms)
        competence_stack = torch.stack(competence_terms)
        empowerment_stack = torch.stack(empowerment_terms)
        survival_stack = torch.stack(survival_terms)
        explore_stack = torch.stack(explore_terms)
        raw_explore_stack = torch.stack(raw_explore_terms)

        if dream_actions:
            divergence_values = torch.stack(
                [
                    (
                        action.detach()
                        - action.detach().mean(dim=0, keepdim=True)
                    )
                    .norm(dim=-1)
                    .mean()
                    for action in dream_actions
                ]
            )
        else:
            divergence_values = torch.tensor([0.0], device=self.device)

        if counterfactual_rates:
            counterfactual_tensor = torch.stack(counterfactual_rates)
        else:
            counterfactual_tensor = torch.tensor([0.0], device=self.device)

        if latent_drift_norms:
            latent_drift_tensor = torch.stack(latent_drift_norms)
        else:
            latent_drift_tensor = torch.tensor([0.0], device=self.device)

        if dream_novelty_values:
            avg_novelty_value = float(sum(dream_novelty_values) / len(dream_novelty_values))
            novelty_threshold = float(getattr(self.config.reward, "novelty_high", 1.0))
            current_streak = 0
            max_novelty_streak = 0
            for value in dream_novelty_values:
                if value >= novelty_threshold:
                    current_streak += 1
                    max_novelty_streak = max(max_novelty_streak, current_streak)
                else:
                    current_streak = 0
        else:
            avg_novelty_value = 0.0
            max_novelty_streak = 0

        dreaming_metrics = {
            "dream/intrinsic_reward": intrinsic_stack.mean().detach(),
            "dream/competence": competence_stack.mean().detach(),
            "dream/empowerment": empowerment_stack.mean().detach(),
            "dream/survival": survival_stack.mean().detach(),
            "dream/policy_entropy": entropies_tensor.mean().detach(),
            "dream/explore": explore_stack.mean().detach(),
            "dream/explore_min": explore_stack.min().detach(),
            "dream/explore_max": explore_stack.max().detach(),
            "dream/explore_raw": raw_explore_stack.mean().detach(),
            "dream/explore_raw_min": raw_explore_stack.min().detach(),
            "dream/explore_raw_max": raw_explore_stack.max().detach(),
            "dream/entropy_boost": torch.tensor(last_entropy_boost, device=self.device),
            "dream/actor_boost": torch.tensor(actor_boost, device=self.device),
            "dream/stimulus_deficit": torch.tensor(
                final_stimulus_deficit, device=self.device
            ),
            "dream/avg_novelty": torch.tensor(avg_novelty_value, device=self.device),
            "dream/max_novelty_streak": torch.tensor(
                float(max_novelty_streak), device=self.device
            ),
            "dream/horizon": torch.tensor(float(dream_horizon), device=self.device),
            "dream/horizon_multiplier": torch.tensor(
                float(horizon_multiplier), device=self.device
            ),
            "dream/action_divergence": divergence_values.mean().detach(),
            "dream/action_divergence_std": divergence_values.std(unbiased=False).detach(),
            "dream/counterfactual_rate": counterfactual_tensor.mean().detach(),
            "dream/latent_drift_norm": latent_drift_tensor.mean().detach(),
        }

        if self._step_count % 100 == 0:
            print(f"\n[Dream Diagnostic at step {self._step_count}]")
            print(
                f"  Rewards: mean={rewards_tensor.mean():.4f}, std={rewards_tensor.std():.6f}, min={rewards_tensor.min():.4f}, max={rewards_tensor.max():.4f}"
            )
            if dream_actions:
                action_norms = torch.stack([a.norm(dim=-1).mean() for a in dream_actions])
                print(
                    f"  Actions: mean_norm={action_norms.mean():.4f}, std={action_norms.std():.6f}"
                )
                print(
                    f"  Action divergence: mean={divergence_values.mean().item():.4f}, std={divergence_values.std(unbiased=False).item():.6f}"
                )
                print(
                    f"  Counterfactual rate: {counterfactual_tensor.mean().item():.4f}"
                )
                print(
                    f"  Latent drift norm: {latent_drift_tensor.mean().item():.6f}"
                )
            print(
                f"  Policy entropy: mean={entropies_tensor.mean():.4f}, std={entropies_tensor.std():.6f}"
            )
            if len(competence_terms) > 0:
                comp_tensor = torch.stack(competence_terms)
                print(
                    f"  Competence: mean={comp_tensor.mean():.4f}, std={comp_tensor.std():.6f}"
                )
            if len(survival_terms) > 0:
                survival_tensor = torch.stack(survival_terms)
                print(
                    f"  Survival: mean={survival_tensor.mean():.4f}, std={survival_tensor.std():.6f}"
                )
            if len(explore_terms) > 0:
                explore_tensor = torch.stack(explore_terms)
                print(
                    f"  Explore: mean={explore_tensor.mean():.4f}, std={explore_tensor.std():.6f}"
                )

        return dream_loss, actor_loss, critic_loss, dreaming_metrics
    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        horizon, batch = rewards.shape
        values_ext = torch.cat([values, next_value.unsqueeze(0)], dim=0)
        advantages = torch.zeros_like(rewards)
        last_advantage = torch.zeros(batch, device=self.device)
        for t in reversed(range(horizon)):
            delta = (
                rewards[t]
                + self.config.discount_gamma * values_ext[t + 1]
                - values_ext[t]
            )
            last_advantage = delta + (
                self.config.discount_gamma
                * self.config.gae_lambda
                * last_advantage
            )
            advantages[t] = last_advantage
        returns = advantages + values
        return advantages, returns

