from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from dtc_agent.motivation import (
    InfoNCEEmpowermentEstimator,
    IntrinsicRewardGenerator,
    ensemble_epistemic_novelty,
    estimate_observation_entropy,
)
from dtc_agent.utils import sanitize_gradients

from .agent import Agent
from .buffer import RolloutBuffer

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .loop import TrainingConfig


class _NullGradScaler:
    __slots__ = ()

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        return None

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None

    def state_dict(self) -> dict[str, float]:
        return {}

    def load_state_dict(self, state_dict: dict[str, float]) -> None:
        return None


def _create_grad_scaler(device_type: str, enabled: bool):
    if device_type != "cuda":
        return _NullGradScaler()
    try:
        return torch.amp.GradScaler(enabled=enabled)  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover - legacy fallback
        from torch.cuda.amp import GradScaler as LegacyGradScaler  # type: ignore[attr-defined]

        return LegacyGradScaler(enabled=enabled)


class Trainer:
    """Own the optimization stack, reward modules, and rollout sampling."""

    def __init__(self, config: TrainingConfig, agent: Agent) -> None:
        self.config = config
        self.agent = agent
        self.device = agent.device
        self.rollout_buffer = RolloutBuffer(capacity=config.rollout_capacity)
        self.batch_size = config.batch_size
        self.autocast_enabled = agent.autocast_enabled

        self.empowerment = InfoNCEEmpowermentEstimator(config.empowerment).to(self.device)
        self.reward = IntrinsicRewardGenerator(
            config.reward,
            empowerment_estimator=self.empowerment,
            novelty_metric=ensemble_epistemic_novelty,
        )
        self.agent.attach_reward_generator(self.reward)

        params: list[torch.nn.Parameter] = []
        params.extend(self.agent.world_model.parameters())
        params.extend(self.empowerment.parameters())
        params.extend(self.agent.actor.parameters())
        params.extend(self.agent.critic.parameters())
        if self.agent.self_state_encoder is not None:
            params.extend(self.agent.self_state_encoder.parameters())
        if self.agent.self_state_predictor is not None:
            params.extend(self.agent.self_state_predictor.parameters())

        self.optimizer = torch.optim.Adam(params, lr=config.optimizer_lr)
        self.optimizer_empowerment_weight = config.optimizer_empowerment_weight
        self.grad_scaler = _create_grad_scaler(self.device.type, self.autocast_enabled)

    def store_transition(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor,
        self_state: torch.Tensor | None = None,
    ) -> None:
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

    def ready(self) -> bool:
        return len(self.rollout_buffer) >= self.batch_size

    def _emergency_reset_if_corrupted(self) -> bool:
        if hasattr(self.empowerment, "temperature"):
            temperature = getattr(self.empowerment, "temperature")
            if isinstance(temperature, torch.Tensor) and not torch.isfinite(temperature).all():
                print(f"[STEP {self.agent.step_count}] 🚨 EMERGENCY: Resetting empowerment temperature")
                with torch.no_grad():
                    temperature.copy_(
                        torch.tensor(
                            self.config.empowerment.temperature,
                            device=temperature.device,
                            dtype=temperature.dtype,
                        )
                    )
                return True

        decoder = getattr(self.agent.world_model, "decoder", None)
        if decoder is not None and hasattr(decoder, "log_std"):
            log_std = getattr(decoder, "log_std")
            if isinstance(log_std, torch.Tensor) and not torch.isfinite(log_std).all():
                print(f"[STEP {self.agent.step_count}] 🚨 EMERGENCY: Resetting decoder log_std")
                with torch.no_grad():
                    log_std.copy_(
                        torch.full_like(log_std, self.config.decoder.init_log_std)
                    )
                return True

        return False

    def _check_parameter_health(self) -> bool:
        components: dict[str, nn.Module] = {
            "world_model": self.agent.world_model,
            "empowerment": self.empowerment,
            "actor": self.agent.actor,
            "critic": self.agent.critic,
        }
        if self.agent.self_state_encoder is not None:
            components["self_state_encoder"] = self.agent.self_state_encoder
        if self.agent.self_state_predictor is not None:
            components["self_state_predictor"] = self.agent.self_state_predictor

        corrupted: list[str] = []
        for prefix, module in components.items():
            for name, param in module.named_parameters():
                if param.requires_grad and param.data is not None:
                    if not torch.isfinite(param.data).all():
                        corrupted.append(f"{prefix}.{name}")

        if corrupted:
            print(f"[STEP {self.agent.step_count}] 🚨 CORRUPTED PARAMETERS:")
            for name in corrupted:
                print(f"  - {name}")
            return False
        return True

    def train_step(self) -> tuple[int, dict[str, float]] | None:
        if not self.ready():
            return None

        if self._emergency_reset_if_corrupted():
            print("[TRAINING] Parameters were corrupted and reset, skipping this update")
            return None

        if not self._check_parameter_health():
            print("[TRAINING] Skipping update due to corrupted parameters")
            return None

        wave_modifiers = self.agent.cognitive_wave_controller.current_modifiers
        current_lr = self.config.optimizer_lr * wave_modifiers["learning_rate_scale"]
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

        self.optimizer.zero_grad(set_to_none=True)

        emp_diag: dict[str, float] | None = None

        with self.agent.autocast_ctx():
            latents = self.agent.call_with_fallback("world_model", observations)

            for key, tensor in latents.items():
                if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all():
                    print(
                        f"[STEP {self.agent.step_count}] NaN in {key} from encoder! Skipping update."
                    )
                    return None
            memory_context = self.agent.get_memory_context(latents["z_self"])
            broadcast, _, _, _, _ = self.agent.route_slots(
                latents["slots"],
                latents["z_self"],
                actions,
                self_states,
                update_stats=False,
            )
            latent_state = broadcast.mean(dim=1)
            features = self.agent.assemble_features(
                latents["z_self"], broadcast, memory_context
            )

            predictions = self.agent.world_model.predict_next_latents(latent_state, actions)
            decoded = self.agent.world_model.decode_predictions(predictions, use_frozen=False)
            log_likelihoods = torch.stack(
                [dist.log_prob(next_observations).mean() for dist in decoded]
            )
            world_model_loss = -log_likelihoods.mean()

            encoded_next = self.agent.call_with_fallback("world_model", next_observations)
            predicted_latent = torch.stack(predictions).mean(dim=0)
            target_latent = encoded_next["slots"].mean(dim=1)
            latent_alignment = torch.nn.functional.mse_loss(predicted_latent, target_latent)
            world_model_loss = (
                world_model_loss + 0.1 * latent_alignment
            ) * self.config.world_model_coef

            self_state_loss = torch.tensor(0.0, device=self.device)
            if (
                self_states is not None
                and self.agent.self_state_dim > 0
                and self.agent.self_state_predictor is not None
            ):
                z_self_float = latents["z_self"].float()
                predicted_state = self.agent.self_state_predictor(z_self_float)
                self_state_loss = torch.nn.functional.mse_loss(predicted_state, self_states)
                self_state_loss = self.config.workspace.self_bias * self_state_loss

            if hasattr(self.empowerment, "get_queue_diagnostics"):
                emp_diag = self.empowerment.get_queue_diagnostics()
                if self.agent.step_count % 1000 == 0:
                    print(
                        f"[Empowerment Queue] Size: {emp_diag['queue_size']}, "
                        f"Diversity: {emp_diag['queue_diversity']:.4f}"
                    )

            dream_loss, actor_loss, critic_loss, dream_metrics = self._stable_dreaming(
                latents, wave_modifiers
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

        for name, loss_val in loss_components.items():
            if not torch.isfinite(loss_val):
                print(f"[STEP {self.agent.step_count}] 🚨 Non-finite {name}_loss: {loss_val}")
                print("  Skipping optimization step to prevent parameter corruption")
                if self.agent.step_count % 1000 == 0:
                    torch.save(
                        {
                            "step": self.agent.step_count,
                            "world_model": self.agent.world_model.state_dict(),
                            "actor": self.agent.actor.state_dict(),
                            "critic": self.agent.critic.state_dict(),
                        },
                        f"nan_guard_step_{self.agent.step_count}.pt",
                    )
                return None

        self.grad_scaler.scale(total_loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.agent.world_model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), max_norm=1.0)

        bad_grads = sanitize_gradients(self.agent.world_model)
        bad_grads += sanitize_gradients(self.agent.actor)
        bad_grads += sanitize_gradients(self.agent.critic)
        bad_grads += sanitize_gradients(self.empowerment)
        if bad_grads > 0:
            print(
                f"[STEP {self.agent.step_count}] WARNING: Sanitized {bad_grads} non-finite gradients"
            )

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        if self.agent.step_count > 0 and self.agent.step_count % 1000 == 0:
            self.agent.world_model.refresh_frozen_decoder()
            if self.agent.step_count % 5000 == 0:
                print(f"[Step {self.agent.step_count}] Refreshed frozen decoder")

        metrics: dict[str, float] = {
            "train/total_loss": float(total_loss.detach().cpu().item()),
            "train/world_model_loss": float(world_model_loss.detach().cpu().item()),
            "train/actor_loss": float(actor_loss.detach().cpu().item()),
            "train/critic_loss": float(critic_loss.detach().cpu().item()),
            "train/dream_loss_empowerment": float(dream_loss.detach().cpu().item()),
            "train/self_state_loss": float(self_state_loss.detach().cpu().item()),
            "wave/stimulus_level": float(wave_modifiers["stimulus_level"]),
            "wave/stimulus_deficit": float(wave_modifiers["stimulus_deficit"]),
            "wave/learning_rate_scale": float(
                wave_modifiers["learning_rate_scale"]
            ),
        }
        if emp_diag is not None:
            metrics["debug/empowerment_queue_size"] = float(emp_diag["queue_size"])
            metrics["debug/empowerment_queue_diversity"] = float(emp_diag["queue_diversity"])
        for key, value in dream_metrics.items():
            if isinstance(value, torch.Tensor):
                metrics[key] = float(value.detach().cpu().item())
            else:
                metrics[key] = float(value)
        return self.agent.step_count, metrics

    def _stable_dreaming(
        self,
        latents: dict[str, torch.Tensor],
        wave_modifiers: dict[str, float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        chunk_size = max(1, self.config.dream_chunk_size)
        num_chunks = max(1, self.config.num_dream_chunks)

        initial_latents = {
            key: value.detach().clone()
            if isinstance(value, torch.Tensor)
            else value
            for key, value in latents.items()
        }
        memory_context = self.agent.get_memory_context(initial_latents["z_self"]).detach()
        dream_self_state: torch.Tensor | None = None
        latest_self_state = self.agent.get_latest_self_state()
        if latest_self_state is not None:
            dream_self_state = latest_self_state.to(
                self.device, non_blocking=True
            )
        current_latents = initial_latents

        entropies: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        log_probs: list[torch.Tensor] = []
        competence_terms = []
        empowerment_terms = []
        safety_terms = []
        survival_terms = []
        intrinsic_terms = []
        explore_terms = []
        raw_explore_terms = []

        dream_actions = []
        counterfactual_rates: list[torch.Tensor] = []
        latent_drift_norms: list[torch.Tensor] = []
        ensemble_noise_base = 0.05
        for _ in range(num_chunks):
            for _ in range(chunk_size):
                broadcast, _, _, _, _ = self.agent.route_slots(
                    current_latents["slots"],
                    current_latents["z_self"],
                    torch.zeros(
                        current_latents["slots"].size(0),
                        self.config.dynamics.action_dim,
                        device=self.device,
                    ),
                    None,
                    update_stats=False,
                )
                features = self.agent.assemble_features(
                    current_latents["z_self"], broadcast, memory_context
                )
                action_dist = self.agent.call_with_fallback("actor", features)
                sampled_action = action_dist.rsample()
                dream_log_prob = action_dist.log_prob(sampled_action)
                dream_entropy = action_dist.entropy()

                entropy_boost = wave_modifiers.get("dream_entropy_boost", 1.0)
                base_noise_scale = 0.15
                action_noise = torch.randn_like(sampled_action) * base_noise_scale
                mutation_scale = 0.1 * (1.0 + entropy_boost)
                policy_mutation = torch.randn_like(sampled_action) * mutation_scale
                mutated_action = sampled_action + action_noise + policy_mutation
                counterfactual_mask = (
                    torch.rand(sampled_action.shape[0], device=sampled_action.device) < 0.1
                )
                wild_action = torch.randn_like(sampled_action) * 2.0
                executed_action = torch.where(
                    counterfactual_mask.unsqueeze(-1), wild_action, mutated_action
                )
                executed_action = executed_action.clamp(-3.0, 3.0)
                dream_actions.append(executed_action)
                counterfactual_rates.append(counterfactual_mask.float().mean())

                latent_state = broadcast.mean(dim=1)
                if self.agent.world_model.training:
                    latent_drift = torch.randn_like(latent_state) * 0.05
                    latent_state = latent_state + latent_drift
                    latent_drift_norms.append(latent_drift.norm(dim=-1).mean())
                else:
                    latent_drift_norms.append(torch.tensor(0.0, device=latent_state.device))

                predictions = self.agent.world_model.predict_next_latents(
                    latent_state, executed_action
                )
                if self.agent.world_model.training:
                    predictions = [
                        pred
                        + torch.randn_like(pred)
                        * (ensemble_noise_base + 0.02 * float(idx))
                        for idx, pred in enumerate(predictions)
                    ]
                decoded = self.agent.world_model.decode_predictions(
                    predictions, use_frozen=False
                )
                novelty = self.reward.get_novelty(predictions, decoded)
                if self.agent.step_count % 100 == 0:
                    print(f"\n[Dream Novelty Diagnostic at step {self.agent.step_count}]")
                    print(
                        "  Novelty: "
                        f"mean={novelty.mean():.6f}, std={novelty.std():.6f}, "
                        f"min={novelty.min():.6f}, max={novelty.max():.6f}"
                    )
                    if latent_drift_norms:
                        print(
                            f"  Latent drift norm (latest): {latent_drift_norms[-1].item():.6f}"
                        )
                    print(
                        f"  Counterfactual rate (latest): {counterfactual_rates[-1].item():.3f}"
                    )
                predicted_obs = decoded[0].rsample()
                observation_entropy = estimate_observation_entropy(predicted_obs)

                if not torch.isfinite(novelty).all() or not torch.isfinite(observation_entropy).all():
                    print(
                        f"[DREAM ERROR at step {self.agent.step_count}] Non-finite dream values detected, skipping this dream chunk"
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
                                "dream/safety",
                                "dream/survival",
                                "dream/policy_entropy",
                                "dream/explore",
                                "dream/explore_min",
                                "dream/explore_max",
                                "dream/explore_raw",
                                "dream/explore_raw_min",
                                "dream/explore_raw_max",
                            ]
                        },
                    )
                self.reward._step_count = self.agent.step_count
                dream_reward, norm_components, raw_components = self.reward.get_intrinsic_reward(
                    novelty,
                    observation_entropy,
                    executed_action,
                    latent_state,
                    self_state=dream_self_state,
                    return_components=True,
                )
                competence_terms.append(norm_components["competence"].detach().mean())
                empowerment_terms.append(norm_components["empowerment"].detach().mean())
                safety_terms.append(norm_components["safety"].detach().mean())
                survival_terms.append(norm_components["survival"].detach().mean())
                intrinsic_terms.append(dream_reward.detach().mean())
                explore_terms.append(norm_components["explore"].detach().mean())
                raw_explore_terms.append(raw_components["explore"].detach().mean())

                normalized_reward = self.agent.reward_normalizer(dream_reward)

                critic_value = self.agent.call_with_fallback("critic", features)
                values.append(critic_value)
                rewards.append(normalized_reward)
                log_probs.append(dream_log_prob)
                entropies.append(dream_entropy)

                current_latents = self.agent.call_with_fallback("world_model", predicted_obs)
                memory_context = self.agent.get_memory_context(current_latents["z_self"])

            current_latents = {key: value.detach() for key, value in current_latents.items()}
            memory_context = memory_context.detach()

        final_broadcast, _, _, _, _ = self.agent.route_slots(
            current_latents["slots"],
            current_latents["z_self"],
            torch.zeros(
                current_latents["slots"].size(0),
                self.config.dynamics.action_dim,
                device=self.device,
            ),
            None,
            update_stats=False,
        )
        final_features = self.agent.assemble_features(
            current_latents["z_self"], final_broadcast, memory_context
        )
        next_value = self.agent.call_with_fallback("critic", final_features).detach()

        rewards_tensor = torch.stack(rewards)
        values_tensor = torch.stack(values)
        log_probs_tensor = torch.stack(log_probs)
        entropies_tensor = torch.stack(entropies)

        advantages, returns = self._compute_gae(
            rewards_tensor, values_tensor, next_value
        )
        dynamic_entropy_coef = (
            self.config.entropy_coef * wave_modifiers["actor_entropy_scale"]
        )
        if self.config.adaptive_entropy:
            novelty_mean = self.agent.novelty_tracker.get_mean()
            current_avg_novelty = float(
                novelty_mean.clamp(min=0.0).item()
            )
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
        dream_loss = -self.optimizer_empowerment_weight * empowerment_term

        intrinsic_stack = torch.stack(intrinsic_terms)
        competence_stack = torch.stack(competence_terms)
        empowerment_stack = torch.stack(empowerment_terms)
        safety_stack = torch.stack(safety_terms)
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

        dreaming_metrics = {
            "dream/intrinsic_reward": intrinsic_stack.mean().detach(),
            "dream/competence": competence_stack.mean().detach(),
            "dream/empowerment": empowerment_stack.mean().detach(),
            "dream/safety": safety_stack.mean().detach(),
            "dream/survival": survival_stack.mean().detach(),
            "dream/policy_entropy": entropies_tensor.mean().detach(),
            "dream/explore": explore_stack.mean().detach(),
            "dream/explore_min": explore_stack.min().detach(),
            "dream/explore_max": explore_stack.max().detach(),
            "dream/explore_raw": raw_explore_stack.mean().detach(),
            "dream/explore_raw_min": raw_explore_stack.min().detach(),
            "dream/explore_raw_max": raw_explore_stack.max().detach(),
            "dream/wave_entropy_boost": float(wave_modifiers["dream_entropy_boost"]),
            "dream/wave_actor_entropy_scale": float(
                wave_modifiers["actor_entropy_scale"]
            ),
            "dream/action_divergence": divergence_values.mean().detach(),
            "dream/action_divergence_std": divergence_values.std(unbiased=False).detach(),
            "dream/counterfactual_rate": counterfactual_tensor.mean().detach(),
            "dream/latent_drift_norm": latent_drift_tensor.mean().detach(),
        }

        if self.agent.step_count % 100 == 0:
            print(f"\n[Dream Diagnostic at step {self.agent.step_count}]")
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
