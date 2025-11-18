import re

import pytest
import torch

from dtc_agent.world_model.encoder import _SlotAttention


def _slot_attention(dim: int = 8) -> _SlotAttention:
    return _SlotAttention(num_slots=2, dim=dim, iters=1, mlp_hidden_size=16, epsilon=1e-6)


def test_slot_attention_rejects_invalid_rank() -> None:
    slot_attention = _slot_attention()
    bad_inputs = torch.zeros(1, 2, 3, 4)

    message = re.escape("Slot Attention inputs must be [batch, num_inputs, dim]")
    with torch.autograd.set_grad_enabled(False):
        with pytest.raises(ValueError, match=message):
            slot_attention(bad_inputs)


def test_slot_attention_rejects_mismatched_dim() -> None:
    slot_attention = _slot_attention(dim=4)
    bad_inputs = torch.zeros(1, 1, 5)

    message = re.escape("Slot Attention inputs last dimension must equal slot dim (4)")
    with torch.autograd.set_grad_enabled(False):
        with pytest.raises(ValueError, match=message):
            slot_attention(bad_inputs)


def test_slot_attention_rejects_empty_sequence() -> None:
    slot_attention = _slot_attention()
    bad_inputs = torch.zeros(1, 0, 8)

    message = re.escape("Slot Attention requires at least one input token")
    with torch.autograd.set_grad_enabled(False):
        with pytest.raises(ValueError, match=message):
            slot_attention(bad_inputs)


def test_slot_attention_rejects_non_finite_inputs() -> None:
    slot_attention = _slot_attention(dim=1)
    bad_inputs = torch.tensor([[[float("nan")]]])

    message = re.escape("Slot Attention inputs must be finite (no NaN or Inf values)")
    with torch.autograd.set_grad_enabled(False):
        with pytest.raises(ValueError, match=message):
            slot_attention(bad_inputs)
