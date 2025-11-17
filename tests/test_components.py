import torch

from dtc_agent.config import load_training_config
from dtc_agent.training.loop import RunningMeanStd, RewardNormalizer, TrainingLoop


def test_compute_gae_matches_manual_calculation() -> None:
    config = load_training_config("configs/testing.yaml")
    loop = TrainingLoop(config)
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[0.5], [0.6], [0.7]])
    next_value = torch.tensor([0.8])

    advantages, returns = loop._compute_gae(rewards, values, next_value)

    gamma = config.discount_gamma
    lam = config.gae_lambda
    values_ext = torch.cat([values, next_value.unsqueeze(0)], dim=0)
    manual_advantages = torch.zeros_like(rewards)
    running = torch.zeros(rewards.size(1))
    for t in reversed(range(rewards.size(0))):
        delta = rewards[t] + gamma * values_ext[t + 1] - values_ext[t]
        running = delta + gamma * lam * running
        manual_advantages[t] = running
    manual_returns = manual_advantages + values

    assert torch.allclose(advantages, manual_advantages, atol=1e-6)
    assert torch.allclose(returns, manual_returns, atol=1e-6)


def test_running_mean_std_and_reward_normalizer_track_statistics() -> None:
    device = torch.device("cpu")
    stats = RunningMeanStd(device=device)
    stats.update(torch.ones(10))
    stats.update(torch.zeros(10))

    assert torch.isclose(stats.mean, torch.tensor([0.5], device=device), atol=1e-3).all()
    assert torch.isclose(stats.var, torch.tensor([0.25], device=device), atol=1e-3).all()

    normalizer = RewardNormalizer(device=device)
    normalized_ones = normalizer(torch.ones(10))
    normalized_zeros = normalizer(torch.zeros(10))

    assert torch.allclose(normalized_ones.mean(), torch.tensor(0.0), atol=1e-3)
    assert torch.allclose(normalized_zeros.mean(), torch.tensor(-1.0), atol=1e-3)
    assert torch.isclose(normalizer.stats.mean, torch.tensor([0.5]), atol=1e-3).all()
    assert torch.isclose(normalizer.stats.var, torch.tensor([0.25]), atol=1e-3).all()
