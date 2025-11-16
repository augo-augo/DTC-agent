from __future__ import annotations

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
import threading
from typing import Any, Callable, ContextManager, ParamSpec, Protocol, TypeVar, cast, TYPE_CHECKING

import torch
from torch import nn
import torch.nn.functional as F

from dtc_agent.agents import ActorConfig, ActorNetwork, CriticConfig, CriticNetwork
from dtc_agent.cognition import WorkspaceRouter
from dtc_agent.memory import EpisodicBuffer
from dtc_agent.motivation import IntrinsicRewardGenerator, estimate_observation_entropy
from dtc_agent.utils import RewardNormalizer, RunningMeanStd, sanitize_tensor
from dtc_agent.world_model import (
    DecoderConfig,
    DynamicsConfig,
    EncoderConfig,
    WorldModelConfig,
    WorldModelEnsemble,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .loop import TrainingConfig


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
    except Exception:  # pragma: no cover - best effort configuration
        pass

    def _compiler(module: nn.Module) -> nn.Module:
        try:
            return compile_fn(module)  # type: ignore[misc]
        except Exception:
            return module

    return _compiler


_maybe_compile = _resolve_compile()


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
    if not configured_matmul and matmul_backend is not None and hasattr(
        matmul_backend, "fp32_precision"
    ):
        try:
            matmul_backend.fp32_precision = "tf32"
            configured_matmul = True
        except (TypeError, AttributeError):
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
    if not configured_cudnn and cudnn_backend is not None and hasattr(
        cudnn_backend, "fp32_precision"
    ):
        try:
            cudnn_backend.fp32_precision = "tf32"
            configured_cudnn = True
        except (TypeError, AttributeError):
            configured_cudnn = False
    conv_backend = getattr(cudnn_backend, "conv", None)
    if not configured_cudnn and conv_backend is not None and hasattr(
        conv_backend, "fp32_precision"
    ):
        try:
            conv_backend.fp32_precision = "tf32"
            configured_cudnn = True
        except (TypeError, AttributeError):
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



@dataclass
class StepResult:
    """Data container returned from ``Agent.act`` / ``TrainingLoop.step``."""

    action: torch.Tensor
    intrinsic_reward: torch.Tensor
    novelty: torch.Tensor
    observation_entropy: torch.Tensor
    slot_scores: torch.Tensor
    reward_components: dict[str, torch.Tensor] | None = None
    raw_reward_components: dict[str, torch.Tensor] | None = None
    training_loss: float | None = None
    training_metrics: dict[str, float] | None = None
    competence_breakdown: dict[str, torch.Tensor] | None = None
    epistemic_novelty: torch.Tensor | None = None
    real_action_entropy: float | None = None
    self_state: torch.Tensor | None = None
    action_index: torch.Tensor | None = None


@dataclass
class CognitiveWaveConfig:
    stimulus_history_window: int = 1000
    stimulus_ema_momentum: float = 0.1
    dream_entropy_scale: float = 10.0
    actor_entropy_scale: float = 5.0
    learning_rate_scale: float = 2.0


class CognitiveWaveController:
    """Balance internal dreaming and external exploration via stimulus tracking."""

    def __init__(self, config: CognitiveWaveConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.stimulus_history: deque[float] = deque(
            maxlen=max(1, config.stimulus_history_window)
        )
        self.env_stimulus_ema = RunningMeanStd(device=device, epsilon=1e-4)
        self._ema_value: torch.Tensor | None = None
        self._current_modifiers = self._default_modifiers()
        self._lock = threading.Lock()

    @property
    def current_modifiers(self) -> dict[str, float]:
        with self._lock:
            return dict(self._current_modifiers)

    def update(self, raw_environmental_novelty: float, step: int) -> None:
        del step
        novelty_tensor = torch.tensor(
            [raw_environmental_novelty], device=self.device, dtype=torch.float32
        )
        self.env_stimulus_ema.update(novelty_tensor)
        running_mean = self.env_stimulus_ema.get_mean().to(self.device)

        with self._lock:
            if self._ema_value is None:
                self._ema_value = running_mean.clone()
            else:
                momentum = float(self.config.stimulus_ema_momentum)
                momentum = max(0.0, min(1.0, momentum))
                self._ema_value = (
                    (1.0 - momentum) * self._ema_value + momentum * running_mean
                )
            self.stimulus_history.append(raw_environmental_novelty)

            if len(self.stimulus_history) < self.config.stimulus_history_window:
                self._current_modifiers = self._default_modifiers()
                return

            assert self._ema_value is not None
            stimulus_level = float(self._ema_value.item())
            stimulus_baseline = sum(self.stimulus_history) / len(self.stimulus_history)

            stimulus_deficit = max(0.0, stimulus_baseline - stimulus_level)
            stimulus_surplus = max(0.0, stimulus_level - stimulus_baseline)

            dream_boost = 1.0 + (stimulus_deficit * self.config.dream_entropy_scale)
            dream_boost = max(1.0, min(self.config.dream_entropy_scale, dream_boost))
            actor_boost = 1.0 + (stimulus_deficit * self.config.actor_entropy_scale)
            actor_boost = max(1.0, min(self.config.actor_entropy_scale, actor_boost))
            lr_scale = 1.0 + (stimulus_surplus * self.config.learning_rate_scale)
            lr_scale = max(1.0, min(self.config.learning_rate_scale, lr_scale))

            self._current_modifiers = {
                "dream_entropy_boost": dream_boost,
                "actor_entropy_scale": actor_boost,
                "learning_rate_scale": lr_scale,
                "stimulus_level": stimulus_level,
                "stimulus_deficit": stimulus_deficit,
            }

    def _default_modifiers(self) -> dict[str, float]:
        return {
            "dream_entropy_boost": 1.0,
            "actor_entropy_scale": 1.0,
            "learning_rate_scale": 1.0,
            "stimulus_level": 0.0,
            "stimulus_deficit": 0.0,
        }


class Agent:
    """Wrap the policy stack, memory routing, and inference logic."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        _configure_tf32_precision(self.device)
        self.progress_momentum = config.workspace.progress_momentum
        self.action_cost_scale = config.workspace.action_cost_scale

        # Clear CUDA cache before initializing models to maximize available memory
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            # Print initial memory stats
            print(f"[Memory] Initial GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")

        wm_config = WorldModelConfig(
            encoder=config.encoder,
            decoder=config.decoder,
            dynamics=config.dynamics,
            ensemble_size=config.world_model_ensemble,
        )
        # WorldModelEnsemble.to() now handles sequential loading with cache clearing
        world_model = WorldModelEnsemble(wm_config).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            print(f"[Memory] After WorldModel: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")

        if self.device.type == "cuda" and config.compile_modules:
            self.world_model = _maybe_compile(world_model)
        else:
            self.world_model = world_model

        self.workspace = WorkspaceRouter(config.workspace)
        self.memory = EpisodicBuffer(config.episodic_memory)
        self.reward: IntrinsicRewardGenerator | None = None
        self.reward_normalizer = RewardNormalizer(device=self.device)

        slot_dim = config.encoder.slot_dim
        policy_feature_dim = (
            slot_dim
            + slot_dim * config.workspace.broadcast_slots
            + config.episodic_memory.key_dim
        )

        # Load actor network
        print(f"[Memory] Loading actor network...")
        actor_net = ActorNetwork(
            ActorConfig(
                latent_dim=policy_feature_dim,
                action_dim=config.dynamics.action_dim,
                hidden_dim=config.actor.hidden_dim,
                num_layers=config.actor.num_layers,
                dropout=config.actor.dropout,
            )
        ).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            print(f"[Memory] After Actor: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")

        # Load critic network
        print(f"[Memory] Loading critic network...")
        critic_net = CriticNetwork(
            CriticConfig(
                latent_dim=policy_feature_dim,
                hidden_dim=config.critic.hidden_dim,
                num_layers=config.critic.num_layers,
                dropout=config.critic.dropout,
            )
        ).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            print(f"[Memory] After Critic: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")
        if self.device.type == "cuda" and config.compile_modules:
            self.actor = _maybe_compile(actor_net)
            self.critic = _maybe_compile(critic_net)
        else:
            self.actor = actor_net
            self.critic = critic_net

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
        self.autocast_enabled = self.device.type == "cuda"
        self.novelty_tracker = RunningMeanStd(device=self.device)
        self.cognitive_wave_controller = CognitiveWaveController(
            config.cognitive_wave, self.device
        )
        self._state_lock = threading.RLock()  # Protects shared state accessed by multiple threads

    @property
    def step_count(self) -> int:
        return self._step_count

    def get_latest_self_state(self) -> torch.Tensor | None:
        """Thread-safe read of the latest self state."""
        with self._state_lock:
            if self._latest_self_state is None:
                return None
            return self._latest_self_state.clone()

    def attach_reward_generator(self, reward: IntrinsicRewardGenerator) -> None:
        self.reward = reward

    def autocast_ctx(self) -> ContextManager[None]:
        if not self.autocast_enabled:
            return nullcontext()
        try:
            return torch.amp.autocast(device_type=self.device.type)
        except AttributeError:  # pragma: no cover - legacy CUDA fallback
            from torch.cuda.amp import autocast as legacy_autocast  # type: ignore[attr-defined]

            return legacy_autocast()

    def call_with_fallback(self, attr: str, *args: P.args, **kwargs: P.kwargs) -> T:
        module = cast(_CallableModule[P, T], getattr(self, attr))
        try:
            return module(*args, **kwargs)
        except Exception as err:
            message = str(err).lower()
            needs_fallback = any(
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
            original_module = original_module.to(self.device)
            original_module.train(module.training)
            setattr(self, attr, original_module)
            try:
                import torch._dynamo as _dynamo  # type: ignore[attr-defined]

                _dynamo.reset()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
            return cast(T, original_module(*args, **kwargs))

    def route_slots(
        self,
        slot_values: torch.Tensor,
        z_self: torch.Tensor,
        action: torch.Tensor,
        self_state: torch.Tensor | None,
        update_stats: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        slot_novelty = slot_values.var(dim=-1, unbiased=False)
        if update_stats:
            self.novelty_tracker.update(slot_novelty.mean(dim=-1))
        if self._slot_baseline is None:
            self._slot_baseline = slot_values.mean(dim=0).detach().cpu()

        if self._novelty_trace is None:
            slot_progress = torch.zeros_like(slot_novelty)
            if update_stats:
                self._novelty_trace = slot_novelty.detach().cpu()
        else:
            prev_trace = self._novelty_trace.to(self.device)
            slot_progress = prev_trace - slot_novelty
            if update_stats:
                updated_trace = (
                    (1 - self.progress_momentum) * self._novelty_trace
                    + self.progress_momentum * slot_novelty.detach().cpu()
                )
                self._novelty_trace = updated_trace

        if update_stats:
            baseline_update = slot_values.mean(dim=0).detach().cpu()
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
            self_state_device = self_state.to(self.device, non_blocking=True)
            # Synchronize to ensure self_state transfer is complete before encoder forward pass
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            projected_state = self.self_state_encoder(self_state_device)
            projected_state = torch.nn.functional.normalize(projected_state, dim=-1)
            state_similarity = (
                slot_norm * projected_state.unsqueeze(1)
            ).sum(dim=-1).clamp(min=0.0)

        self_mask = torch.clamp(self_similarity + state_similarity, min=0.0)

        batch_mean = slot_novelty.mean(dim=0).detach().cpu()
        with self._state_lock:
            if self._ucb_mean is None or self._ucb_counts is None:
                self._ucb_mean = batch_mean.clone()
                self._ucb_counts = torch.ones_like(batch_mean)
            elif update_stats:
                self._ucb_counts = self._ucb_counts + 1
                self._ucb_mean = self._ucb_mean + (batch_mean - self._ucb_mean) / self._ucb_counts
                self._step_count += 1
            ucb_mean_snapshot = self._ucb_mean.clone()
            ucb_counts_snapshot = self._ucb_counts.clone()
        assert ucb_mean_snapshot is not None and ucb_counts_snapshot is not None
        ucb_bonus = (
            ucb_mean_snapshot.to(self.device)
            + self.config.workspace.ucb_beta
            * torch.sqrt(
                torch.log1p(torch.tensor(float(self._step_count), device=self.device))
                / ucb_counts_snapshot.to(self.device)
            )
        )
        ucb = ucb_bonus.unsqueeze(0).expand(slot_values.size(0), -1)

        scores = self.workspace.score_slots(
            novelty=slot_novelty,
            progress=slot_progress,
            ucb=ucb,
            cost=slot_cost,
            self_mask=self_mask,
        )
        broadcast = self.workspace.broadcast(slot_values, scores=scores)
        return broadcast, scores, slot_novelty, slot_progress, slot_cost

    def get_memory_context(self, keys: torch.Tensor) -> torch.Tensor:
        batch = keys.shape[0]
        if len(self.memory) == 0:
            return torch.zeros(batch, self.memory.config.key_dim, device=self.device)
        _, values = self.memory.read(keys)
        context = values[:, 0, :].to(self.device)
        return context

    def write_memory(self, z_self: torch.Tensor, slots: torch.Tensor) -> None:
        key = z_self.detach().cpu()
        value = slots.mean(dim=1).detach().cpu()
        self.memory.write(key, value)

    def assemble_features(
        self,
        z_self: torch.Tensor,
        broadcast: torch.Tensor,
        memory_context: torch.Tensor,
    ) -> torch.Tensor:
        broadcast_flat = broadcast.flatten(start_dim=1)
        return torch.cat([z_self, broadcast_flat, memory_context], dim=-1)

    def act(
        self,
        observation: torch.Tensor,
        action: torch.Tensor | None = None,
        self_state: torch.Tensor | None = None,
        train: bool = False,
    ) -> StepResult:
        if self.reward is None:
            raise RuntimeError("Reward generator has not been attached to the agent")

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

        # Synchronize CUDA to ensure all async transfers are complete before model inference
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        if state_tensor is not None:
            with self._state_lock:
                self._latest_self_state = state_tensor.detach()

        competence_breakdown: dict[str, torch.Tensor] | None = None
        epistemic_novelty: torch.Tensor | None = None
        current_real_entropy: float | None = None
        selected_action_index: torch.Tensor | None = None

        restore_world_model = False
        if not train and self.world_model.training:
            self.world_model.eval()
            restore_world_model = True

        try:
            with torch.no_grad():
                with self.autocast_ctx():
                    latents = self.call_with_fallback("world_model", observation)
                    memory_context = self.get_memory_context(latents["z_self"])
                if action is not None:
                    action_for_routing = action.to(self.device, non_blocking=True)
                    # Synchronize to ensure action transfer is complete
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
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
                ) = self.route_slots(
                    latents["slots"],
                    latents["z_self"],
                    action_for_routing,
                    state_tensor,
                    update_stats=True,
                )
                features = self.assemble_features(latents["z_self"], broadcast, memory_context)
                if action is None:
                    action_dist = self.call_with_fallback("actor", features)
                    base_entropy = action_dist.entropy()
                    wave_mods = self.cognitive_wave_controller.current_modifiers
                    entropy_scale = wave_mods.get("actor_entropy_scale", 1.0)
                    sample_dist = action_dist
                    logits = getattr(action_dist, "logits", None)
                    if train and entropy_scale > 1.0 and logits is not None:
                        temperature = max(1.0, float(entropy_scale))
                        adjusted_logits = logits / temperature
                        sample_dist = torch.distributions.Categorical(logits=adjusted_logits)
                    action_indices = sample_dist.sample()
                    selected_action_index = action_indices.detach()
                    action = F.one_hot(
                        action_indices,
                        num_classes=self.config.dynamics.action_dim,
                    ).to(device=self.device, dtype=latents["z_self"].dtype)
                    if torch.isfinite(base_entropy).all():
                        entropy_value = float(base_entropy.mean().item())
                    else:
                        entropy_value = 0.0

                    with self._state_lock:
                        if not hasattr(self, "_real_entropy_history"):
                            self._real_entropy_history = deque(maxlen=100)
                        recent_window = self._real_entropy_history
                        recent_window.append(entropy_value)
                        if len(recent_window) == recent_window.maxlen:
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
                    ) = self.route_slots(
                        latents["slots"],
                        latents["z_self"],
                        action,
                        state_tensor,
                        update_stats=False,
                    )
                    features = self.assemble_features(
                        latents["z_self"], broadcast, memory_context
                    )
                else:
                    action = action.to(
                        device=self.device,
                        dtype=latents["z_self"].dtype,
                        non_blocking=True,
                    )
                    if action.ndim == 1:
                        action = action.unsqueeze(0)
                    if action.size(0) != batch:
                        if action.size(0) == 1:
                            action = action.expand(batch, -1)
                        else:
                            raise ValueError("action batch dimension mismatch")
                    if action.size(-1) == self.config.dynamics.action_dim:
                        selected_action_index = torch.argmax(action, dim=-1).detach()
                    # Synchronize to ensure action transfer is complete
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)

                latent_state = broadcast.mean(dim=1)
                predictions = self.world_model.predict_next_latents(latent_state, action)
                decoded = self.world_model.decode_predictions(predictions)
                novelty = self.reward.get_novelty(predictions, decoded).to(self.device)
                observation_entropy = estimate_observation_entropy(observation)

                if not torch.isfinite(novelty).all():
                    finite = novelty[torch.isfinite(novelty)]
                    if finite.numel() > 0:
                        min_val = float(finite.min().item())
                        max_val = float(finite.max().item())
                        mean_val = float(finite.mean().item())
                    else:
                        min_val = max_val = mean_val = float("nan")
                    print(f"[STEP {self._step_count}] ERROR: NaN/Inf in novelty!")
                    print(
                        f"  novelty stats: min={min_val}, max={max_val}, mean={mean_val}"
                    )
                    novelty = sanitize_tensor(novelty, replacement=0.0)

                if not torch.isfinite(observation_entropy).all():
                    finite = observation_entropy[torch.isfinite(observation_entropy)]
                    if finite.numel() > 0:
                        min_val = float(finite.min().item())
                        max_val = float(finite.max().item())
                    else:
                        min_val = max_val = float("nan")
                    print(
                        f"[STEP {self._step_count}] ERROR: NaN/Inf in observation_entropy!"
                    )
                    print(f"  entropy stats: min={min_val}, max={max_val}")
                    observation_entropy = sanitize_tensor(
                        observation_entropy, replacement=0.1
                    )

                if not torch.isfinite(action).all():
                    print(f"[STEP {self._step_count}] ERROR: NaN/Inf in sampled action!")
                    action = sanitize_tensor(action, replacement=0.0)

                self.reward._step_count = self._step_count
                intrinsic_raw, norm_components, raw_components = self.reward.get_intrinsic_reward(
                    novelty,
                    observation_entropy,
                    action,
                    latent_state,
                    self_state=state_tensor,
                    return_components=True,
                )

                raw_novelty_tensor = raw_components.get("explore")
                if raw_novelty_tensor is not None:
                    raw_novelty_mean = float(raw_novelty_tensor.mean().item())
                    self.cognitive_wave_controller.update(
                        raw_novelty_mean, self._step_count
                    )

                competence_breakdown = getattr(
                    self.reward, "_last_competence_breakdown", {}
                )
        finally:
            if restore_world_model:
                self.world_model.train()

        intrinsic = self.reward_normalizer(intrinsic_raw)
        reward_components = {key: value.detach() for key, value in norm_components.items()}
        raw_reward_components = {key: value.detach() for key, value in raw_components.items()}
        self.write_memory(latents["z_self"], broadcast)

        action_index_tensor = (
            selected_action_index.detach() if selected_action_index is not None else None
        )

        return StepResult(
            action=action.detach(),
            action_index=action_index_tensor,
            intrinsic_reward=intrinsic.detach(),
            novelty=slot_novelty.detach(),
            observation_entropy=observation_entropy.detach(),
            slot_scores=scores.detach(),
            reward_components=reward_components,
            raw_reward_components=raw_reward_components,
            training_loss=None,
            training_metrics=None,
            competence_breakdown=competence_breakdown,
            epistemic_novelty=epistemic_novelty.detach()
            if epistemic_novelty is not None
            else None,
            real_action_entropy=current_real_entropy,
            self_state=state_tensor.detach() if state_tensor is not None else None,
        )
