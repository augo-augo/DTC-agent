import sys
import types

import torch

sys.modules.setdefault("faiss", types.SimpleNamespace())

from dtc_agent.motivation import (
    ensemble_epistemic_novelty,
    estimate_observation_entropy,
)


def test_estimate_observation_entropy_shape() -> None:
    observation = torch.randn(4, 3, 16, 16)
    entropy = estimate_observation_entropy(observation)
    assert entropy.shape == (4,)


def test_entropy_increases_with_variance() -> None:
    low_var = torch.zeros(4, 3, 16, 16)
    high_var = torch.randn(4, 3, 16, 16)
    entropy_low = estimate_observation_entropy(low_var)
    entropy_high = estimate_observation_entropy(high_var)
    assert torch.all(entropy_high >= entropy_low - 1e-6)


def test_ensemble_epistemic_novelty_responds_to_disagreement() -> None:
    batch = 4
    latent_dim = 8
    pred1 = torch.randn(batch, latent_dim)
    pred2 = pred1 + torch.randn(batch, latent_dim) * 0.5
    pred3 = pred1 + torch.randn(batch, latent_dim) * 0.5

    predicted_latents = [pred1, pred2, pred3]

    decoded_means = [
        torch.randn(batch, 3, 8, 8),
        torch.randn(batch, 3, 8, 8) + 0.2,
        torch.randn(batch, 3, 8, 8) - 0.2,
    ]
    decoded_dists = [
        torch.distributions.Normal(mean, torch.ones_like(mean) * 0.5)
        for mean in decoded_means
    ]

    novelty = ensemble_epistemic_novelty(predicted_latents, decoded_dists)

    assert novelty.shape == (batch,)
    assert torch.all(novelty > 0)
    assert torch.all(torch.isfinite(novelty))
