from __future__ import annotations

import torch


def estimate_observation_entropy(observation: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Estimate observation entropy with a numerically stable helper.

    Args:
        observation: Batch of observations ``[batch, channels, height, width]``.
        eps: Minimum variance added during entropy estimation.

    Returns:
        Tensor of entropy estimates for each observation.
    """

    from dtc_agent.utils.stability import estimate_observation_entropy_stable

    return estimate_observation_entropy_stable(observation, eps=eps)
