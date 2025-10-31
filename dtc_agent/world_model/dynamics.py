from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DynamicsConfig:
    """Hyperparameters for a single dynamics model in the ensemble.

    Attributes:
        latent_dim: Dimensionality of the latent state.
        action_dim: Dimensionality of the action input.
        hidden_dim: Width of the intermediate hidden representation.
        dropout: Dropout probability applied within the transition MLP.
    """

    latent_dim: int
    action_dim: int
    hidden_dim: int = 256
    dropout: float = 0.1


class DynamicsModel(nn.Module):
    """Simple GRU-based dynamics model stub."""

    def __init__(self, config: DynamicsConfig) -> None:
        """Initialize the dynamics model components.

        Args:
            config: Dynamics hyper-parameters shared across ensemble members.
        """

        super().__init__()
        self.config = config
        self.input_layer = nn.Linear(config.latent_dim + config.action_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.transition = nn.GRUCell(config.hidden_dim, config.latent_dim)
        self.register_buffer("_noise_scale", torch.tensor(0.1))

    def forward(self, latent_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict the next latent state given the previous latent state and action.

        Args:
            latent_state: Current latent state tensor ``[batch, latent_dim]``.
            action: Action tensor ``[batch, action_dim]`` taken at the state.

        Returns:
            Tensor containing the predicted next latent state for each batch item.

        Raises:
            ValueError: If the batch dimensions of ``latent_state`` and ``action`` differ.
        """
        if latent_state.shape[0] != action.shape[0]:
            raise ValueError("latent_state and action batch dimensions must match")
        joint = torch.cat([latent_state, action], dim=-1)
        target_dtype = latent_state.dtype
        hidden = torch.relu(self.input_layer(joint))
        hidden = self.dropout(hidden)
        hidden_float = hidden.float()
        latent_state_float = latent_state.float()
        device_type = hidden_float.device.type
        autocast_fn = getattr(torch.amp, "autocast", None)
        if autocast_fn is not None and device_type in ("cuda", "cpu"):
            autocast_ctx = autocast_fn(device_type=device_type, enabled=False)
        else:
            autocast_ctx = nullcontext()
        with autocast_ctx:
            next_latent = self.transition(hidden_float, latent_state_float)
        if self.training:
            noise = torch.randn_like(next_latent) * self._noise_scale
            next_latent = next_latent + noise
        return next_latent.to(target_dtype)
