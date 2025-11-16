import math
import torch
import torch.nn.functional as F

from dtc_agent.config import load_training_config
from dtc_agent.training import TrainingLoop


def test_compute_gae_shapes() -> None:
    config = load_training_config("configs/testing.yaml")
    loop = TrainingLoop(config)
    horizon = config.effective_dream_horizon
    batch = 4
    rewards = torch.randn(horizon, batch)
    values = torch.randn(horizon, batch)
    next_value = torch.randn(batch)

    advantages, returns = loop.trainer._compute_gae(rewards, values, next_value)

    assert advantages.shape == (horizon, batch)
    assert returns.shape == (horizon, batch)


def test_stable_dreaming_outputs_are_finite() -> None:
    config = load_training_config("configs/testing.yaml")
    loop = TrainingLoop(config)
    batch = 2
    observations = torch.rand(batch, *config.encoder.observation_shape).to(
        loop.agent.device
    )
    latents = loop.agent.call_with_fallback("world_model", observations)
    wave_modifiers = loop.agent.cognitive_wave_controller.current_modifiers

    dream_loss, actor_loss, critic_loss, metrics = loop.trainer._stable_dreaming(
        latents, wave_modifiers
    )

    for loss in (dream_loss, actor_loss, critic_loss):
        assert loss.ndim == 0
        assert torch.isfinite(loss)
    assert isinstance(metrics, dict)
    for key in ("dream/explore", "dream/explore_raw", "dream/explore_min", "dream/explore_max"):
        assert key in metrics
    for value in metrics.values():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all()
        else:
            assert math.isfinite(float(value))


def test_optimize_backpropagates_to_all_modules() -> None:
    torch.manual_seed(0)
    config = load_training_config("configs/testing.yaml")
    loop = TrainingLoop(config)
    obs_shape = config.encoder.observation_shape
    action_dim = config.dynamics.action_dim

    for _ in range(loop.batch_size):
        observation = torch.rand(*obs_shape)
        action_index = torch.randint(0, action_dim, (1,))
        action = F.one_hot(action_index, num_classes=action_dim).float().squeeze(0)
        next_observation = torch.rand(*obs_shape)
        self_state = torch.rand(config.self_state_dim)
        loop.trainer.rollout_buffer.push(
            observation, action, next_observation, self_state
        )

    result = loop.train_step()
    assert result is not None
    step_index, metrics = result
    assert isinstance(step_index, int)
    assert "train/total_loss" in metrics
    for key in ("dream/explore", "dream/explore_raw", "dream/explore_min", "dream/explore_max"):
        assert key in metrics
    for value in metrics.values():
        assert math.isfinite(float(value))
    def _has_grad(module: torch.nn.Module) -> bool:
        grads = [param.grad for param in module.parameters() if param.requires_grad]
        return len(grads) > 0 and all(g is not None and torch.isfinite(g).all() for g in grads)

    assert _has_grad(loop.agent.world_model)
    assert _has_grad(loop.agent.actor)
    assert _has_grad(loop.agent.critic)


