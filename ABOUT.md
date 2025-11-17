**Technical Specification & Implementation Plan: SC-GWT v2.0 (Hardened)**

## **1. Executive Summary**

This document outlines the architecture for **SC-GWT v2.0**, a model-based reinforcement learning (RL) agent designed for general-purpose artificial intelligence. The agent's core drive is not an external reward but a set of **intrinsic motivations** based on resolving **epistemic uncertainty** (competence) and maximizing **influence** (empowerment).

It is a **Self-Centric** system because it uses a **Global Workspace (GW)** architecture where internal, self-preservation signals are given a hard-coded bias, allowing them to override other cognitive processes.

This **v2.0 (Hardened)** specification incorporates critical fixes to address known pathologies in this class of agent, including:

  * **Latent Space Drift:** Solved by grounding novelty in observation-space predictions.
  * **Wireheading (The "Dark Room" Problem):** Solved by a multi-part intrinsic reward (Competence + Empowerment + Safety) and a "frozen observer" head.
* **Set-point Instability:** Solved with dual-timescale "Goldilocks" reward signals.
* **Superstition:** Solved with ensemble-based consistency checking and counterfactual probes.
* **Catastrophic Forgetting:** Solved with an episodic memory buffer (kNN).

### **Implementation Kickoff (October 21, 2025)**

Initial Python scaffolding for the SC-GWT prototype now exists under the `dtc_agent/` package with:

- world model ensemble stubs (encoder, dynamics, decoder, ensemble wrapper, frozen observer sync)
- intrinsic reward generator with competence/empowerment/safety placeholders and JSD metric shim
- cognition workspace router handling top-k broadcasts into the global workspace
- FAISS-backed episodic buffer for non-parametric recall (with eviction TODO)
- prototype training loop wiring the subsystems plus a CLI entry point returning a dummy intrinsic reward

Immediate next steps:

1. Wire the training loop to a real environment rollout source and expand the world-model objective beyond the Gaussian stub.
2. Harden the empowerment estimator with contrastive negatives and evaluate stability across seeds.
3. Add regression tests covering observation entropy, workspace salience metrics, and optimizer behaviour.

- Replaced the decoder placeholder with a configurable deconvolutional generator (learnable variance, selectable activations).
- Introduced OmegaConf-driven YAML configs (`configs/default.yaml`, `configs/testing.yaml`) and a loader exposed via `dtc_agent.config.load_training_config`.
- Updated the CLI (`python -m dtc_agent.training`) to accept `--config`, `--override`, and runtime device switching for rapid experiment setup.
- Added observation entropy estimation and slot salience heuristics to drive WorkspaceRouter cost/novelty inputs.
- Added a rollout buffer, optimizer stubs, and unified step result reporting in the training harness.
- Introduced actor/critic modules with Stable Dreaming (GAE-driven imagined rollouts and policy/value losses).
- Landed initial pytest scaffolding covering entropy metrics and rollout sampling.
- Swapped in a slot-attention encoder with configurable CNN backbone and iterative slots.
- Added an InfoNCE empowerment estimator with a latent replay queue and config-driven hyperparameters.
- Added `docs/CONFIGURATION.md` documenting configuration usage and override patterns.
- Wired the Crafter survival runner (`dtc_agent/training/__main__.py`) into Weights & Biases with per-step vitals, optimizer diagnostics, and episode video logging for experiment tracking.
- Hardened early training stability with intrinsic-reward normalization and mixed-precision optimisation (reward clipping + AMP with grad scaling) to pre-empt NaN or OOM failures.
- Introduced an exploration bonus term derived from ensemble disagreement to encourage broader search without destabilising competence/empowerment balances.
- Normalized each intrinsic drive (competence/empowerment/safety/explore) with running statistics so reward weights remain balanced across training phases.

## **2. Core Principles**

1.  **The Agent is a Prediction Engine:** The agent's primary function is to build a generative **World Model (WM)** of itself and its environment to predict the future.
2.  **Motivation is Intrinsic:** All behavior is driven by the singular goal of improving this World Model. The agent is rewarded for:
      * **Competence:** Reducing its own *ignorance* (epistemic uncertainty).
      * **Empowerment:** Seeking states where it has maximum control and optionality.
3.  **Memory is the Model (Mostly):** Long-term, general knowledge is encoded in the parametric weights of the World Model. Short-term, specific facts are held in a non-parametric episodic buffer.
4.  **Cognition is Gated:** A **Global Workspace** acts as a low-bandwidth attentional bottleneck, forcing the agent to serialize its focus on the *most salient* information.
5.  **Self-Preservation is Emergent & Biased:** The agent learns that it cannot predict the world if its own systems are failing. This drive is "bootstrapped" by a hard-coded attentional bias for critical self-state variables (e.g., integrity, energy).

## **3. Formal Architecture Specification (v2.0)**

The agent is composed of four main subsystems.

### **Component I: The World Model (WM) Ensemble**

This is the agent's "simulator" and perceptual system. It is composed of a shared encoder/decoder and an ensemble of dynamics models.

  * **1. Shared Encoder (with Slot Attention):**
      * **Purpose:** To take a raw observation $o_t$ and decompose it into a meaningful, object-centric latent state $z_t$.
      * **Mechanism:** A CNN backbone followed by a **Slot Attention** module.
      * **Output ($S_{actual}$):** The full latent state $z_t = \{z_{self}, s_1, s_2, \dots, s_m\}$, where $z_{self}$ is a learned internal state and $s_i$ are $m$ object "slots."
  * **2. Ensemble Dynamics Models:**
      * **Purpose:** To predict the future.
      * **Mechanism:** An ensemble of $k$ independent dynamics models (e.g., GRUs, Transformers). Each $f^{(i)}$ predicts the next latent state: $\hat{z}_t^{(i)} = f^{(i)}(z_{t-1}, a_{t-1})$.
      * **Output ($S_{predicted}$):** A set of $k$ differing latent state predictions.
  * **3. Shared Decoder:**
      * **Purpose:** To project latent states back into reality for grounding and novelty calculation.
      * **Mechanism:** A generative model (e.g., Transposed CNN) that can recompose the latent slots $s_i$ into a full observation prediction.
      * **Output:** A distribution over the observation space, $p_\theta(o_t\mid \hat z_t^{(i)})$.

### **Component II: The Reward & Novelty Subsystem**

This subsystem calculates the final intrinsic reward $R_{intr}$ that trains the agent.

  * **1. Epistemic Novelty Calculation (Fix \#1, \#8):**
      * **Purpose:** To robustly measure agent *ignorance* without latent drift.
      * **Mechanism:** We calculate novelty in **observation space**, not latent space. We use the **Jensen-Shannon Divergence (JSD)** between the $k$ predicted observations from the *shared decoder*.
        $$N_{epi,t} \leftarrow \mathrm{JSD}\big(\{p_\theta(o_t\mid \hat z_t^{(i)})\}_{i=1}^k\big)$$
      * **Hardening:** This calculation is performed using a **"frozen observer"** (a periodic copy of the decoder) that the policy cannot "game" via its actions.
  * **2. Competence Reward (Fix \#2):**
      * **Purpose:** To reward the agent for learning, while preventing oscillations or anxiety.
      * **Mechanism:** A **dual-timescale "Goldilocks"** signal. We maintain a "fast" EMA $\bar{N}_{fast}$ of novelty. The reward is the progress against this baseline, *minus* a penalty for overwhelming novelty (chaos).
        $$\bar{N}_{fast,t} = (1-\alpha_{\text{fast}})\bar{N}_{fast, t-1} + \alpha_{\text{fast}} N_{epi,t}$$
        $$R_{comp,t} = \big(\bar{N}_{fast, t-1} - N_{epi,t}\big) - \kappa\cdot\max(0,\,N_{epi,t}-N_{high})$$
  * **3. Empowerment Reward (Fix \#3):**
      * **Purpose:** To provide a stable "drive to influence" that prevents the "dark room" problem.
      * **Mechanism:** We replace the unstable Mutual Information $I(A;S')$ with a stable **InfoNCE contrastive estimator** $f_\psi(a, z')$.
        $$R_{emp,t} \;\approx\; \log \frac{\exp f_\psi(a_t, z’{t})}{\sum{j=0}^{K}\exp f_\psi(a^-_j, z’{t})}$$
  * **4. Safety Reward (Fix \#8):**
      * **Purpose:** A final, hard-coded penalty against sensor-blinding.
      * **Mechanism:** Penalize any drop in sensory entropy below a minimum threshold.
        $$R_{safety} = -\lambda_{sens}\,\max(0, H_{min} - H[o_t])$$
  * **Total Intrinsic Reward:**
    $$R_{intr, t} = \lambda_{comp}R_{comp} + \lambda_{emp}R_{emp} + R_{safety}$$

### **Component III: The Cognitive Control Subsystem**

This manages attention (what to think about) and episodic memory (what to remember).

  * **1. Global Workspace (GW) Router (Fix \#4):**
      * **Purpose:** To select the $k$ most salient "slots" of information for system-wide processing.
      * **Mechanism:** A biased Top-K selection. An "Attraction Score" is calculated for each slot $s_i$ based on multiple factors:
        $$A^{(i)} = w_1 N^{(i)}_{epi} + w_2 \Delta N^{(i)} + w_3 \text{UCB}^{(i)} - w_4 \text{Cost}^{(i)} + \text{Bias}_{self}^{(i)}$$
      * This score balances:
          * $N^{(i)}_{epi}$: Novelty (how poorly is this slot understood?)
          * $\Delta N^{(i)}$: Learning Progress (am I getting better at this?)
          * $\text{UCB}^{(i)}$: Exploration Bonus (is it time to re-check this?)
          * $\text{Cost}^{(i)}$: Action Cost (what will it cost to probe this?)
          * $\text{Bias}_{self}^{(i)}$: **Self-Centric Bias** (is this internal signal an *emergency*?)
      * **Output:** The $GW_t$, a $k$-slot vector, is broadcast to the Actor and Episodic Buffer.
  * **2. Episodic Buffer (Fix \#5):**
      * **Purpose:** To enable one-shot learning and prevent catastrophic forgetting.
      * **Mechanism:** A Key-Value store (e.g., a FAISS index for kNN).
      * **Write:** At each step, a key (projected from $GW_t$) and a value (the recent trajectory data) are written to the buffer.
      * **Read:** The Actor queries the buffer using the *current* $GW_t$ to retrieve the $k$-Nearest-Neighbor experiences, which are fed in as context for action.
      * **Maintenance:** Entries have a Time-To-Live (TTL) and are pruned unless re-retrieved, proving their relevance.

### **Component IV: The Actor-Critic (AC) System**

This is the agent's decision-making and self-evaluation system.

  * **1. The Actor (Policy) $\pi(a_t | \cdot)$:**
      * **Purpose:** To select the optimal action.
      * **Input:** The $GW_t$ broadcast, the full self-state $z_{self, t}$, and the retrieved memories from the Episodic Buffer.
      * **Output:** A distribution over actions $a_t$.
  * **2. The Critic (Value) $V_\phi(z_t)$:**
      * **Purpose:** To provide a stable baseline for learning by predicting the expected future $R_{intr}$.
      * **Input:** The latent state $z_t$.
      * **Output:** A scalar value (expected future return).

## **4. The Unified Learning Process**

The agent is trained in a two-phase loop: collecting real experience and training on "dreamed" experience.

### **A. Unified Loss Function**

The entire system is trained by minimizing a single, multi-part loss function:
$$\mathcal{L}_{\text{total}} = \lambda_{wm}\mathcal{L}_{WM} + \lambda_{\text{actor}}\mathcal{L}_{\text{actor}} + \lambda_{\text{critic}}\mathcal{L}_{\text{critic}} + \lambda_{emp}\mathcal{L}_{emp}$$

  * **$\mathcal{L}_{WM}$ (World Model Loss):** The ELBO loss, which includes:
      * **Reconstruction Loss:** $\mathcal{L}_{\text{recon}} = -\mathbb{E}[ \log p(o_t | z_t) ]$
      * **Dynamics Loss:** $\mathcal{L}_{\text{dyn}} = \mathbb{E}[ D_{KL}[p(z_t | o_t, \dots) \parallel p(\hat{z}_t | \dots)] ]$
  * **$\mathcal{L}_{\text{actor}}$ (Actor Loss):**
      * Trained on **Generalized Advantage Estimation (GAE)**, which is the "shotgun" eligibility trace. The advantage $A_t$ is the (discounted) sum of TD-errors, where the TD-error is:
        $$A_t = R_{intr, t} + \gamma V_\phi(z_{t+1}) - V_\phi(z_t)$$
      * $\mathcal{L}_{\text{actor}} = -\mathbb{E}[ A_t \cdot \log \pi(a_t | z_t) ]$
  * **$\mathcal{L}_{\text{critic}}$ (Critic Loss):**
      * A simple regression loss to predict the value.
        $$\mathcal{L}_{\text{critic}} = \mathbb{E}\left[ \left( (R_{intr, t} + \gamma V_{\text{target}}(z_{t+1})) - V_\phi(z_t) \right)^2 \right]$$
  * **$\mathcal{L}_{emp}$ (Empowerment Loss):**
      * The loss function for the InfoNCE estimator $f_\psi$.

### **B. The Training Loop (Stable Dreaming)**

1.  **Phase 1: Collect Experience (Real-time)**
      * The agent uses its full processing loop (Steps 1-5) to select an action $a_t$ in the real environment.
      * The transition $(o_t, a_t, o_{t+1}, R_{intr, t})$ is stored in a main `ReplayBuffer`.
      * The episodic data is stored in the `EpisodicBuffer_M`.
2.  **Phase 2: Train Components (Offline)**
      * **Train World Model:** A batch of *real* data is sampled from `ReplayBuffer` to train the WM via the $\mathcal{L}_{WM}$ loss.
      * **Train Actor-Critic (Dreaming):**
          * A batch of starting states $z_t$ is sampled from the `ReplayBuffer`.
          * The WM "dreams" (imagines) a set of short rollouts (e.g., $H=10$) starting from those states, using the Actor's policy.
          * The hardened $R_{intr}$ signals are calculated for this *imagined* trajectory.
          * The Actor and Critic are trained *entirely* on these cheap, dreamed rollouts via $\mathcal{L}_{\text{actor}}$ and $\mathcal{L}_{\text{critic}}$.
      * **Stability (Fix \#6):** Training is a mix of real and imagined data (e.g., 1:4 ratio) to keep the WM anchored to reality.
      * **Superstition Filtering (Fix \#7):** The WM is also trained on a **consistency loss**, forcing it to make robust predictions during counterfactual probes and domain randomization. This filters out spurious correlations.

## **5. Implementation & Staging Plan**

### **A. Core Tech Stack**

  * **Language:** Python 3.10+
  * **Frameworks:** PyTorch or JAX (for ensemble handling and custom gradients).
  * **Libraries:** FAISS (for kNN buffer), `numpy`, `wandb` (for logging).

### **B. Component Implementation (Pseudocode Skeletons)**

```python
# --- 1. World Model Component ---
class WM_Ensemble(nn.Module):
    def __init__(self, k_ensemble, m_slots):
        self.encoder = SlotAttentionEncoder(m_slots)
        self.dynamics_models = nn.ModuleList([DynamicsModel() for _ in range(k_ensemble)])
        self.decoder = SharedDecoder(m_slots)
        self.frozen_decoder = copy.deepcopy(self.decoder)

    def get_novelty(self, z_t_minus_1, a_t_minus_1):
        # 1. Get k prior predictions
        priors_z = [model(z_t_minus_1, a_t_minus_1) for model in self.dynamics_models]
        # 2. Decode using *frozen* head
        predicted_obs_dists = [self.frozen_decoder(z) for z in priors_z]
        # 3. Calculate JSD
        N_epi = decoder_variance_novelty(predicted_obs_dists)
        return N_epi
    
    def get_elbo_loss(self, batch):
        # ... ELBO loss calculation ...
        pass

# --- 2. Reward Generation Component ---
class IntrinsicRewardGenerator:
    def __init__(self, alpha_fast, N_high, kappa):
        self.ema_fast = 0.0
        self.alpha_fast = alpha_fast
        self.N_high = N_high
        self.kappa = kappa
        
        self.R_emp_estimator = InfoNCE_Estimator()
        self.R_safety_thresh = 0.1 # H_min

    def get_R_comp(self, N_epi_t):
        R_progress = self.ema_fast - N_epi_t
        self.ema_fast = (1 - self.alpha_fast) * self.ema_fast + self.alpha_fast * N_epi_t
        
        R_anxiety_penalty = self.kappa * max(0, N_epi_t - self.N_high)
        return R_progress - R_anxiety_penalty

    def get_R_safety(self, obs_t):
        entropy = calculate_entropy(obs_t)
        return -max(0, self.R_safety_thresh - entropy)

    def get_full_reward(self, N_epi_t, obs_t, a_t, z_prime_t):
        R_comp = self.get_R_comp(N_epi_t)
        R_safety = self.get_R_safety(obs_t)
        R_emp = self.R_emp_estimator(a_t, z_prime_t)
        
        R_intr = (lambda_comp * R_comp + 
                  lambda_emp * R_emp + 
                  lambda_safety * R_safety)
        return R_intr

# --- 3. Cognitive Control Components ---
class WorkspaceRouter:
    def get_attraction_scores(self, per_slot_novelty, per_slot_progress, ...):
        # A = w1*N + w2*dN + w3*UCB - w4*Cost + Bias_self
        pass # Returns [B, m_slots] scores

    def get_gw_broadcast(self, z_t, scores, k):
        # Top-k selection
        pass # Returns [B, k, slot_dim]

class EpisodicBuffer_kNN:
    def __init__(self, capacity, key_dim):
        self.index = faiss.IndexFlatL2(key_dim)
        self.store = {} # Stores values
    
    def write(self, key, value):
        # ... Add key to index, add value to store ...
        pass

    def read(self, query_key, k):
        # ... Search index, retrieve values from store ...
        pass
```

### **C. Phased Curriculum (Education Plan)**

The agent cannot be trained on its full objective from scratch.

1.  **Phase 1: The "Playpen" (0-1M steps)**
      * **Environment:** Simple physics sandbox (e.g., Crafter, MiniGrid).
      * **Goal:** Train the World Model.
      * **Active Losses:** $\mathcal{L}_{WM}$ (high weight), $\mathcal{L}_{emp}$. The Actor is trained *only* on $R_{emp}$.
      * **Metric:** $\downarrow \mathcal{L}_{WM}$ (World Model loss). The agent should learn to move and interact just to maximize its control.
2.  **Phase 2: The "School" (1M-10M steps)**
      * **Environment:** Add simple, solvable puzzles (e.g., key-and-door).
      * **Goal:** Train the competence drive.
      * **Active Losses:** Enable $\mathcal{L}_{actor}$ and $\mathcal{L}_{critic}$ trained on the full $R_{intr}$ (with a high $\lambda_{comp}$).
      * **Metric:** $\uparrow R_{comp}$. The agent should show a clear drive to solve puzzles to reduce novelty.
3.  **Phase 3: The "World" (10M+ steps)**
      * **Environment:** Complex, open-ended 3D simulation or language-based world.
      * **Goal:** Observe emergent, self-driven, stable behavior.
      * **Metrics:** Stable, non-zero $R_{comp}$ and $R_{emp}$. Does the agent explore, practice, and maintain itself without explicit instruction? Does it avoid wireheading pathologies?

