from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn

from .decoder import DecoderConfig, SharedDecoder
from .dynamics import DynamicsConfig, DynamicsModel
from .encoder import EncoderConfig, SlotAttentionEncoder


@dataclass
class WorldModelConfig:
    """Configuration for constructing the world-model ensemble.

    Attributes:
        encoder: Configuration for the shared encoder.
        decoder: Configuration for the shared decoder.
        dynamics: Configuration for each dynamics model.
        ensemble_size: Number of dynamics models to instantiate.
    """

    encoder: EncoderConfig
    decoder: DecoderConfig
    dynamics: DynamicsConfig
    ensemble_size: int = 5


class WorldModelEnsemble(nn.Module):
    """Bundle shared encoder/decoder with an ensemble of dynamics models."""

    def __init__(self, config: WorldModelConfig) -> None:
        """Construct encoder, decoder, and ensemble of dynamics models.

        Args:
            config: Configuration describing each component and ensemble size.
        """

        super().__init__()
        self.config = config
        self.encoder = SlotAttentionEncoder(config.encoder)
        self.decoder = SharedDecoder(config.decoder)
        self.frozen_decoder = copy.deepcopy(self.decoder)
        for param in self.frozen_decoder.parameters():
            param.requires_grad_(False)
        self.dynamics_models = nn.ModuleList()
        for i in range(config.ensemble_size):
            torch.manual_seed(42 + i * 1000)
            model = DynamicsModel(config.dynamics)
            with torch.no_grad():
                for param in model.parameters():
                    if param.requires_grad:
                        param.add_(torch.randn_like(param) * 0.05)
            self.dynamics_models.append(model)

        torch.manual_seed(42)

    @torch.no_grad()
    def refresh_frozen_decoder(self) -> None:
        """Synchronize the frozen observer head with the trainable decoder."""
        self.frozen_decoder.load_state_dict(self.decoder.state_dict())

    def forward(self, observation: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode an observation into latent slots.

        Args:
            observation: Observation batch ``[batch, channels, height, width]``.

        Returns:
            Mapping with keys ``"z_self"`` and ``"slots"``.
        """
        return self.encoder(observation)

    def predict_next_latents(
        self, latent_state: torch.Tensor, action: torch.Tensor
    ) -> List[torch.Tensor]:
        """Run the ensemble forward to obtain next-state predictions.

        Args:
            latent_state: Latent state tensor ``[batch, latent_dim]``.
            action: Action tensor ``[batch, action_dim]``.

        Returns:
            List of predicted next latent states from each ensemble member.
        """
        return [model(latent_state, action) for model in self.dynamics_models]

    def decode_predictions(
        self, predicted_latents: Iterable[torch.Tensor], use_frozen: bool = True
    ) -> List[torch.distributions.Distribution]:
        """Decode predicted latent states into observation distributions.

        Args:
            predicted_latents: Sequence of latent predictions to decode.
            use_frozen: Whether to use the frozen decoder copy for stability.

        Returns:
            List of observation distributions for each latent input.
        """
        decoder = self.frozen_decoder if use_frozen else self.decoder
        return [decoder(latent) for latent in predicted_latents]

    def get_epistemic_disagreement(
        self, predicted_latents: List[torch.Tensor]
    ) -> torch.Tensor:
        """Measure epistemic uncertainty as ensemble disagreement.

        Args:
            predicted_latents: List of latent predictions from the ensemble.

        Returns:
            Tensor quantifying disagreement; zero when fewer than two models.
        """
        if not predicted_latents:
            param = next(self.parameters(), None)
            device = param.device if param is not None else torch.device("cpu")
            return torch.zeros(0, device=device)

        if len(predicted_latents) < 2:
            return torch.zeros(predicted_latents[0].shape[0], device=predicted_latents[0].device)

        stacked = torch.stack(predicted_latents, dim=0)
        std_per_dim = stacked.std(dim=0)
        disagreement = std_per_dim.mean(dim=-1)
        return disagreement
