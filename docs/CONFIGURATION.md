# Configuration Guide

This project uses OmegaConf-driven YAML configuration files located in `configs/` to
control the behaviour of the `TrainingLoop` and the surrounding subsystems.

## File Layout

- `configs/default.yaml` - canonical parameters for standard development runs.
- `configs/testing.yaml` - lighter configuration with smaller slot/action spaces for fast smoke tests.

Both files expose the same keys:

| Section | Purpose |
| ------- | ------- |
| `encoder` | Slot-attention encoder settings (CNN channels, slot iterations, observation shape). |
| `decoder` | Parameters for the deconvolutional `SharedDecoder` (channel stack, activations, variance controls). |
| `dynamics` | Dimensions for each ensemble dynamics member. |
| `workspace` | Global workspace routing weights, broadcast count, progress momentum, UCB exploration factors, and action cost scaling. |
| `reward` | Coefficients driving competence/empowerment/safety signals. |
| `empowerment` | InfoNCE empowerment estimator dimensions, queue depth, and temperature. |
| `episodic_memory` | Capacity and key dimensionality for the FAISS-backed episodic buffer. |
| `rollout_capacity` | Number of transitions retained in the replay buffer. |
| `batch_size` | Minibatch size drawn from the rollout buffer during optimization. |
| `optimizer_lr` | Adam learning rate shared by the world model, actor, critic, and empowerment heads. |
| `optimizer_empowerment_weight` | Weighting applied to empowerment maximisation inside reconstruction and dreaming losses. |
| `actor` | Policy-network hidden size, depth, and dropout (state/action dimensionality inferred automatically). |
| `critic` | Value-network hidden size, depth, and dropout. |
| `dream_chunk_size` | Length of each imagined rollout segment processed before detaching gradients. |
| `num_dream_chunks` | Number of chunks composed to form the full imagined trajectory (effective horizon = chunk_size × num_chunks). |
| `discount_gamma` | Discount factor used for imagined returns. |
| `gae_lambda` | GAE smoothing factor for advantage estimates. |
| `entropy_coef` | Entropy regularisation weight added to the actor loss. |
| `critic_coef` | Weighting applied to the critic regression term. |
| `world_model_coef` | Global scale applied to reconstruction + latent-alignment losses. |
| `world_model_ensemble` | Number of models in the dynamics ensemble. |
| `device` | Default runtime device. |

## CLI Usage

```bash
python -m dtc_agent.training --config configs/default.yaml
```

Override individual values without editing the file by passing OmegaConf dot-list
values:

```bash
python -m dtc_agent.training \
  --config configs/testing.yaml \
  --override encoder.slot_dim=48 \
  --override reward.lambda_emp=0.4 \
  --override empowerment.temperature=0.05
```

Use `--device` to quickly switch devices at runtime:

```bash
python -m dtc_agent.training --device cuda:0
```

If new parameters are added to any dataclass, extend the corresponding YAML section to
keep the configuration in sync.
