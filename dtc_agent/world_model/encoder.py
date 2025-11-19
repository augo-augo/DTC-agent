from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class EncoderConfig:
    """Configuration container for the shared encoder.

    Attributes:
        observation_shape: Shape ``(channels, height, width)`` of inputs.
        slot_dim: Dimensionality of each output slot.
        num_slots: Number of object slots to produce.
        cnn_channels: Feature channels for the CNN backbone.
        kernel_size: Convolution kernel size used in the backbone.
        slot_iterations: Number of Slot Attention refinement steps.
        mlp_hidden_size: Width of the Slot Attention MLP.
        epsilon: Numerical stability term for attention normalization.
    """

    observation_shape: tuple[int, int, int]
    slot_dim: int
    num_slots: int
    cnn_channels: Sequence[int] = field(default_factory=lambda: (32, 64, 128))
    kernel_size: int = 5
    slot_iterations: int = 3
    mlp_hidden_size: int = 128
    epsilon: float = 1e-6


class _ConvBackbone(nn.Module):
    """Simple CNN feature extractor with positional embeddings."""

    def __init__(self, in_channels: int, channels: Sequence[int], kernel_size: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = in_channels
        for hidden in channels:
            layers.extend(
                [
                    nn.Conv2d(
                        current,
                        hidden,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm2d(hidden),
                    nn.ReLU(inplace=True),
                ]
            )
            current = hidden
        layers.append(nn.Conv2d(current, current, kernel_size=1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class _PositionalEmbedding(nn.Module):
    """Adds sine-cosine positional encoding to flattened features."""

    def __init__(self, num_channels: int, height: int, width: int) -> None:
        super().__init__()
        self.register_buffer(
            "pos", self._build_embedding(num_channels, height, width), persistent=False
        )

    @staticmethod
    def _build_embedding(num_channels: int, height: int, width: int) -> torch.Tensor:
        if num_channels % 4 != 0:
            raise ValueError("Position embedding channels must be divisible by 4")
        y_range = torch.linspace(-1.0, 1.0, steps=height)
        x_range = torch.linspace(-1.0, 1.0, steps=width)
        yy, xx = torch.meshgrid(y_range, x_range, indexing="ij")
        dim_t = torch.arange(num_channels // 4, dtype=torch.float32)
        dim_t = 1.0 / (10000 ** (dim_t / (num_channels // 4)))

        pos_y = yy[..., None] * dim_t
        pos_x = xx[..., None] * dim_t
        pe = torch.stack(
            [torch.sin(pos_y), torch.cos(pos_y), torch.sin(pos_x), torch.cos(pos_x)],
            dim=-1,
        )
        return pe.view(height * width, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.pos.shape[-1]:
            raise ValueError("Mismatched positional embedding dimension")
        if x.shape[1] != self.pos.shape[0]:
            raise ValueError("Mismatched positional embedding spatial size")
        return x + self.pos


class _SlotAttention(nn.Module):
    """Slot Attention implementation following Locatello et al., 2020."""

    def __init__(
        self,
        num_slots: int,
        dim: int,
        iters: int,
        mlp_hidden_size: int,
        epsilon: float,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.epsilon = epsilon

        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

        self.slot_mu = nn.Parameter(torch.zeros(1, 1, dim))
        self.slot_sigma = nn.Parameter(torch.ones(1, 1, dim))

        self.project_q = nn.Linear(dim, dim, bias=False)
        self.project_k = nn.Linear(dim, dim, bias=False)
        self.project_v = nn.Linear(dim, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("Slot Attention inputs must be [batch, num_inputs, dim]")
        if inputs.shape[-1] != self.dim:
            raise ValueError(
                f"Slot Attention inputs last dimension must equal slot dim ({self.dim})"
            )
        if self.num_slots <= 0:
            raise ValueError("Slot Attention requires num_slots to be positive")
        if inputs.shape[1] <= 0:
            raise ValueError("Slot Attention requires at least one input token")

        device_type = inputs.device.type
        if device_type == "cuda":
            try:
                autocast_ctx = torch.amp.autocast(
                    device_type=device_type, enabled=False
                )
            except AttributeError:  # pragma: no cover - legacy torch fallback
                from torch.cuda.amp import autocast as legacy_autocast  # type: ignore[attr-defined]

                autocast_ctx = legacy_autocast(enabled=False)
        else:
            autocast_ctx = nullcontext()

        with autocast_ctx:
            b, _, d = inputs.shape
            # Force float32 BEFORE LayerNorm to prevent FP16 overflow in variance calc
            inputs = inputs.to(dtype=torch.float32)
            inputs = self.norm_inputs(inputs)

            mu = self.slot_mu.expand(b, self.num_slots, -1).to(dtype=torch.float32)
            sigma = F.softplus(self.slot_sigma).clamp(min=0.1, max=2.0).to(dtype=torch.float32)
            slots = (mu + sigma * torch.randn_like(mu)).to(dtype=torch.float32)

            k = self.project_k(inputs).contiguous().to(dtype=torch.float32)
            v = self.project_v(inputs).to(dtype=torch.float32)

            for _ in range(self.iters):
                slots_prev = slots
                slots = self.norm_slots(slots).to(dtype=torch.float32)

                q = self.project_q(slots).to(dtype=torch.float32)
                # Force q^T contiguous so matmul can pick stable dense GEMM kernels.
                q_t = q.transpose(1, 2).contiguous()
                try:
                    # Use bmm to enforce strict batch-matrix-multiply path
                    # This avoids broken cublasSgemmStridedBatched on Blackwell (5090)
                    dots = torch.bmm(k, q_t) * (d ** -0.5)
                except RuntimeError as err:
                    if "cublas" in str(err).lower():
                        raise RuntimeError(
                            "Slot Attention attention scores failed on CUDA. "
                            f"inputs shape={inputs.shape}, num_slots={self.num_slots}. "
                            "A zero dimensional configuration (e.g. encoder.num_slots=0) "
                            "or empty spatial tokens will trigger this."
                        ) from err
                    raise

                attn = dots.softmax(dim=1) + self.epsilon
                attn = attn / attn.sum(dim=2, keepdim=True)

                # Force attn^T contiguous to fix strided memory access on Blackwell GPUs
                updates = torch.matmul(attn.transpose(1, 2).contiguous(), v)

                slots = self.gru(
                    updates.reshape(-1, d),
                    slots_prev.reshape(-1, d),
                )
                slots = slots.view(b, self.num_slots, d).to(dtype=torch.float32)
                slots = slots + self.mlp(self.norm_mlp(slots)).to(dtype=torch.float32)

            return slots


class SlotAttentionEncoder(nn.Module):
    """Slot Attention encoder producing object-centric slots and a dedicated self-state."""

    def __init__(self, config: EncoderConfig) -> None:
        """Initialize the encoder components described by ``config``.

        Args:
            config: Encoder hyper-parameters shared across the model.
        """

        super().__init__()
        self.config = config
        if config.slot_dim <= 0:
            raise ValueError(f"slot_dim must be positive, got {config.slot_dim}")
        if config.num_slots <= 0:
            raise ValueError(f"num_slots must be positive, got {config.num_slots}")
        if not config.cnn_channels:
            raise ValueError("cnn_channels must not be empty")
        if config.cnn_channels[-1] <= 0:
            raise ValueError(
                f"cnn_channels last entry must be positive, got {config.cnn_channels[-1]}"
            )
        c, h, w = config.observation_shape
        self.backbone = _ConvBackbone(c, config.cnn_channels, config.kernel_size)
        feature_dim = config.cnn_channels[-1]
        self.positional = _PositionalEmbedding(feature_dim, h, w)
        self.pre_slots = nn.Linear(feature_dim, config.slot_dim)
        self.slot_attention = _SlotAttention(
            num_slots=config.num_slots,
            dim=config.slot_dim,
            iters=config.slot_iterations,
            mlp_hidden_size=config.mlp_hidden_size,
            epsilon=config.epsilon,
        )
        self.self_state = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, config.slot_dim),
        )

    def forward(self, observation: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode a batch of observations into latent slots.

        Args:
            observation: Observation batch ``[batch, channels, height, width]``.

        Returns:
            Mapping containing the self slot (``z_self``) and object slots
            (``slots``).

        Raises:
            ValueError: If ``observation`` does not have 4 dimensions.
        """
        if observation.ndim != 4:
            raise ValueError("observation must be [batch, channels, height, width]")
        features = self.backbone(observation)
        batch, channels, height, width = features.shape
        flat = features.view(batch, channels, height * width).permute(0, 2, 1).contiguous()
        flat = self.positional(flat)
        flat = self.pre_slots(flat)
        slots = self.slot_attention(flat)
        z_self = self.self_state(features)
        return {"z_self": z_self, "slots": slots}
