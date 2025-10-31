from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class WorkspaceConfig:
    """Hyper-parameters controlling the GW slot routing heuristics.

    Attributes:
        broadcast_slots: Number of slots to forward to the global workspace.
        self_bias: Bias applied to slots resembling the agent's self state.
        novelty_weight: Weight for novelty in the routing score.
        progress_weight: Weight for slot-level progress in the score.
        cost_weight: Weight penalizing frequently used or costly slots.
        progress_momentum: Momentum for updating novelty progress traces.
        action_cost_scale: Scaling factor converting action magnitude to cost.
        ucb_weight: Weight applied to the UCB bonus when scoring slots.
        ucb_beta: Exploration strength for the UCB bonus term.
    """

    broadcast_slots: int
    self_bias: float
    novelty_weight: float
    progress_weight: float
    cost_weight: float
    progress_momentum: float = 0.1
    action_cost_scale: float = 1.0
    ucb_weight: float = 0.2
    ucb_beta: float = 1.0


class WorkspaceRouter:
    """Implements a lightweight routing bottleneck for the global workspace."""

    def __init__(self, config: WorkspaceConfig) -> None:
        """Store the configuration used to score and select slots.

        Args:
            config: Parameters describing the GW routing heuristic.
        """

        self.config = config

    def score_slots(
        self,
        novelty: torch.Tensor,
        progress: torch.Tensor,
        ucb: torch.Tensor,
        cost: torch.Tensor,
        self_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute slot scores by combining novelty, progress, UCB, and cost.

        Args:
            novelty: Per-slot novelty estimates ``[batch, num_slots]``.
            progress: Slot progress metrics ``[batch, num_slots]``.
            ucb: Upper-confidence bonus ``[batch, num_slots]``.
            cost: Usage cost penalty ``[batch, num_slots]``.
            self_mask: Mask indicating overlap with the agent's self state.

        Returns:
            Tensor of per-slot scores favouring novel yet inexpensive slots.
        """
        score = (
            self.config.novelty_weight * novelty
            + self.config.progress_weight * progress
            + self.config.ucb_weight * ucb
            - self.config.cost_weight * cost
        )
        score = score + self.config.self_bias * self_mask
        return score

    def broadcast(self, slots: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Select the top-``k`` slots according to their routing scores.

        Args:
            slots: Slot tensor of shape ``[batch, num_slots, slot_dim]``.
            scores: Scalar scores ``[batch, num_slots]`` produced by ``score_slots``.

        Returns:
            Tensor containing the ``broadcast_slots`` highest-scoring slots.
        """
        k = self.config.broadcast_slots
        topk = torch.topk(scores, k=k, dim=1).indices
        return torch.gather(
            slots, dim=1, index=topk.unsqueeze(-1).expand(-1, -1, slots.shape[-1])
        )
