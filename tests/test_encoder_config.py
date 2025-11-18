import pytest

from dtc_agent.world_model.encoder import EncoderConfig, SlotAttentionEncoder


def _base_config(**overrides: object) -> EncoderConfig:
    base = dict(
        observation_shape=(3, 32, 32),
        slot_dim=32,
        num_slots=4,
        cnn_channels=(32, 64, 32),
    )
    base.update(overrides)
    return EncoderConfig(**base)


def test_slot_dim_must_be_positive() -> None:
    config = _base_config(slot_dim=0)
    with pytest.raises(ValueError, match="slot_dim must be positive"):
        SlotAttentionEncoder(config)


def test_cnn_channels_must_have_positive_tail() -> None:
    config = _base_config(cnn_channels=(16, -8))
    with pytest.raises(ValueError, match="cnn_channels last entry must be positive"):
        SlotAttentionEncoder(config)


def test_feature_dim_and_slot_dim_must_match() -> None:
    config = _base_config(slot_dim=64, cnn_channels=(32, 64, 32))
    with pytest.raises(ValueError, match="pre_slots input dimension does not match"):
        SlotAttentionEncoder(config)
