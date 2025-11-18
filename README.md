# DTC: An Intrinsically Motivated Agent for Emergent Curriculum Learning

**Author:** augo-augo (Independent Researcher)  
**Version:** 3.0 (November 2025)

[![Paper](https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b.svg)](https://arxiv.org)
[![Code](https://img.shields.io/badge/GitHub-DTC--agent-blue.svg)](https://github.com/augo-augo/DTC-agent)
[![Training Logs](https://img.shields.io/badge/W%26B-Training%20Logs-orange.svg)](https://wandb.ai/bsingleton-art-scgwt/scgwt-crafter/runs/5dz500gm?nw=nwuserbsingletonart)

---

## Abstract

This repository contains the official implementation of the **DTC (Dual-Timescale Competence)** agent, a model-based reinforcement learning agent designed for open-ended learning without external rewards. Version 3.0 introduces a *spectral intrinsic motivation stack* that combines **Learning Progress** with a **Clean Curiosity** signal derived from probabilistic dynamics, and adds **episodic dream seeding** to rehearse long-term skills. These components feed a **Cognitive Wave Controller** that modulates policy entropy to maintain perpetual discovery.

When evaluated in the open-ended Crafter environment, the DTC agent demonstrates the emergence of a multi-stage developmental curriculum, progressing from simple survival skills to a "boredom trough," followed by the compositional discovery of more complex behaviors like agriculture and tool use. The architecture now explicitly distinguishes epistemic from aleatoric uncertainty, preventing "white-noise addiction," and uses episodic teleportation to fight catastrophic forgetting without external rewards.

---

## Core Mechanisms

The agent's behavior is an emergent property of three core mechanisms:

### 1. Dual-Timescale Learning Progress (The "Boredom" Signal)

The primary reward component is $R_{LP}$, derived from two EMAs of *prediction error* ($E_{fast,t}$, $E_{slow,t}$) tracked over novelty batches:

- **`error_fast`** ($\alpha=0.3$): Captures the instantaneous surprise being experienced
- **`error_slow`** ($\alpha=0.01$): Captures the background mastery level

The reward is the *rectified derivative* of understanding:

$$
R_{LP,t} = \mathrm{ReLU}\left(\frac{E_{slow,t-1} - E_{fast,t}}{E_{slow,t-1} + \varepsilon}\right)
$$

- **Discovery:** When error spikes, $E_{fast} > E_{slow}$ and $R_{LP} = 0$ (no punishment)
- **Mastery:** As the model improves, $E_{fast}$ drops below $E_{slow}$ and $R_{LP} > 0$, creating a strong signal to keep practicing until boredom returns

This dynamic is visible in the agent's training logs where competence oscillates as tasks are imagined, mastered, and discarded during dreamed rollouts.

### 2. Cognitive Wave (The "Exploration" Driver)

The agent acts on "boredom" via the `CognitiveWaveController`. This controller tracks the overall `stimulus_level` (a running mean of novelty). When this level drops, indicating mastery and stagnation, the controller boosts exploration pressure by scaling:

- **`dream_entropy_scale`** (×10.0)
- **`actor_entropy_scale`** (×5.0)

This forces the agent to explore more diverse, novel policies in its imagination, allowing it to break out of local optima and discover the next set of learnable skills. This mechanism is the direct cause of the agent's developmental breakthroughs.

![Cognitive Wave Entropy Boost](data/wave_entropy_boost.png)  
*Figure 1: The Cognitive Wave Controller boosts dream entropy during periods of low environmental stimulus (boredom), enabling breakthrough discoveries.*

### 3. Spectral Curiosity & Episodic Rehearsal

Version 3.0 replaces the legacy "safety" hacks with principled mechanisms:

- **Clean Novelty:** Each dynamics model predicts a Normal distribution. Ensemble disagreement ($\mathrm{Var}(\mu)$) is reduced by the predicted aleatoric noise ($\mathbb{E}[\sigma^2]$), rewarding only learnable ignorance. The agent no longer chases stochastic chaos.
- **Probabilistic Dream Seeding:** An episodic FAISS buffer stores latent snapshots. A configurable fraction of every dream batch is *teleported* to recalled states, ensuring maintenance rehearsal and long-range exploration.
- **Survival Bias:** A lightweight survival term keeps the agent slightly "hungry," guaranteeing forward momentum without hand-crafted safety floors.

---

## Key Results: Emergent Developmental Stages in Crafter

Driven only by these internal signals—**without any external rewards or curriculum engineering**—the agent's training run reveals a clear, multi-stage curriculum.

### Phase I & II: Skill Mastery and "Boredom Trough" (~0k - 140k steps)

The agent first masters basic skills like `wake_up`. Once mastered (~80k steps), the DTC "boredom" signal kicks in, and the agent *stops* performing the skill. This is not catastrophic forgetting; it is **graceful skill deprecation** sustained by episodic rehearsal (dream seeding keeps the skill latent even while it is not actively practiced).

![Wake Up Achievement](data/wake_up.png)  
*Figure 2: The agent masters `wake_up` in early training, then abandons it as the competence reward drops to zero.*

### Phase III & IV: "Agricultural Revolution" and Second Cycle (~140k+ steps)

The exploratory pressure built during the "Boredom Trough" (Figure 1) leads to a cascade of compositional discoveries:

1. The agent discovers `collect_sapling` (~140k steps)
2. This immediately unlocks the repeatable, high-level strategy of `place_plant` (farming)
3. This new paradigm supports other advanced skills like combat and resource gathering

![Collect Sapling](data/collect_sapling.png)  
*Figure 3: Discovery of `collect_sapling` marks the beginning of the "agricultural revolution."*

![Place Plant](data/place_plant.png)  
*Figure 4: The agent masters `place_plant` (farming), a compositional skill requiring sapling collection.*

![Make Wood Sword](data/make_wood_sword.png)  
*Figure 5: Tool-making behaviors like `make_wood_sword` emerge after establishing resource pipelines.*

![Defeat Zombie](data/defeat_zombie.png)  
*Figure 6: Combat skills appear naturally once the agent has weapons and confidence.*

4. After mastering this new set of skills (~220k steps), the agent again becomes "bored," and the cycle repeats

---

## Why This Matters

Most RL agents require carefully shaped reward functions and curriculum design. The DTC agent demonstrates that complex, structured behavior can emerge from simple intrinsic drives, suggesting a path toward more autonomous, self-directed AI systems. The "hardened" pathology fixes make this approach actually reliable in practice.

---

## Getting Started

This project is a standard Python package, configured with `omegaconf` and built on `torch`.

### 1. Clone and Install (Conda Recommended)

For GPU-accelerated training, install the core CUDA-enabled dependencies with `conda` before using `pip` for the remaining Python packages.

```bash
# Clone the repository
git clone https://github.com/augo-augo/DTC-agent.git
cd DTC-agent

# Create a conda environment with PyTorch and FAISS that match your CUDA toolkit
conda create -n dtc python=3.10 pytorch-cuda=11.8 pytorch faiss-gpu -c pytorch -c conda-forge

# Activate the environment
conda activate dtc

# Install the package in editable mode (installs remaining dependencies from pyproject.toml)
pip install -e .
```

### 2. Run Tests

Ensure all components are functioning correctly by running the `pytest` suite:

```bash
pytest
```

### 3. Run a Test Agent

You can run a lightweight version of the agent on your CPU using the `testing.yaml` config:

```bash
python -m dtc_agent.training --config configs/testing.yaml --device cpu
```

### 4. Run the Full Experiment

The full experiment (as seen in the W&B logs) uses the `default.yaml` config and requires a CUDA-enabled GPU:

```bash
# This requires a CUDA-enabled GPU and W&B setup
python -m dtc_agent.training --config configs/default.yaml --device cuda
```

---


## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## Contact

For questions or collaboration inquiries, please open an issue on GitHub or contact the author directly.
