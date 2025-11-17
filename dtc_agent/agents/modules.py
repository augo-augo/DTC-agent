from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class ActorConfig:
    """Configuration for constructing an :class:`ActorNetwork`.

    Attributes:
        latent_dim: Dimensionality of the concatenated feature input.
        action_dim: Number of continuous action dimensions produced.
        hidden_dim: Width of the intermediate MLP layers.
        num_layers: Number of hidden layers in the shared backbone MLP.
        dropout: Dropout rate applied after hidden activations.
    """

    latent_dim: int
    action_dim: int
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.0


@dataclass
class CriticConfig:
    """Configuration for constructing a :class:`CriticNetwork`.

    Attributes:
        latent_dim: Dimensionality of the aggregated latent features.
        hidden_dim: Width of the shared critic MLP layers.
        num_layers: Number of hidden layers in the critic backbone.
        dropout: Dropout rate applied after hidden activations.
    """

    latent_dim: int
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.0


def _build_mlp(input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> nn.Sequential:
    """Construct a feed-forward network used by the actor and critic backbones.

    Args:
        input_dim: Size of the first linear layer input.
        hidden_dim: Width for all hidden linear layers.
        num_layers: Number of hidden layers to instantiate.
        dropout: Dropout probability applied after each activation.

    Returns:
        ``nn.Sequential`` MLP containing ``num_layers`` linear blocks.
    """

    layers: list[nn.Module] = []
    for i in range(num_layers):
        layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.SiLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class ActorNetwork(nn.Module):
    """Actor producing Gaussian action distributions conditioned on GW broadcast and memory context."""

    def __init__(self, config: ActorConfig) -> None:
        """Initialize the policy network.

        Args:
            config: Hyper-parameters describing the backbone and output heads.
        """

        super().__init__()
        self.config = config
        self.backbone = _build_mlp(
            input_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        self.mean_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.log_std = nn.Parameter(torch.zeros(config.action_dim))

    def forward(self, features: torch.Tensor) -> torch.distributions.Distribution:
        """Compute a factorized Normal policy distribution.

        Args:
            features: Latent features describing the policy context.

        Returns:
            ``torch.distributions.Independent`` Normal over ``action_dim`` actions.
        """

        features_float = features.float()
        hidden = self.backbone(features_float)
        mean = self.mean_head(hidden)
        std = torch.exp(self.log_std).clamp(min=1e-4, max=10.0)
        return torch.distributions.Independent(
            torch.distributions.Normal(mean, std),
            1,
        )


class CriticNetwork(nn.Module):
    """Critic estimating state value from aggregated latent features."""

    def __init__(self, config: CriticConfig) -> None:
        """Initialize the value function network.

        Args:
            config: Hyper-parameters describing the backbone architecture.
        """

        super().__init__()
        self.config = config
        self.backbone = _build_mlp(
            input_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        self.value_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Estimate the scalar value for the provided latent features.

        Args:
            features: Latent representation of the current policy context.

        Returns:
            Tensor containing the value prediction for each batch element.
        """

        features_float = features.float()
        hidden = self.backbone(features_float)
        return self.value_head(hidden).squeeze(-1)
