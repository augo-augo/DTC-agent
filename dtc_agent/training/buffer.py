from __future__ import annotations

import random
from collections import deque
from typing import Deque
import threading

import torch


class RolloutBuffer:
    """Simple FIFO buffer for on-policy rollouts."""

    def __init__(self, capacity: int) -> None:
        """Initialize the buffer.

        Args:
            capacity: Maximum number of transitions retained in the buffer.

        Raises:
            ValueError: If ``capacity`` is not strictly positive.
        """

        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._storage: Deque[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]] = deque(
            maxlen=capacity
        )
        self._lock = threading.Lock()

    def push(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor,
        self_state: torch.Tensor | None = None,
    ) -> None:
        """Insert a transition into the buffer.

        Args:
            observation: Observation tensor at time ``t``.
            action: Action tensor taken at time ``t``.
            next_observation: Observation tensor at time ``t + 1``.
            self_state: Optional auxiliary agent state associated with the
                transition. ``None`` indicates that no self-state is tracked.
        """

        with self._lock:
            self._storage.append((observation, action, next_observation, self_state))

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Sample a batch of transitions uniformly without replacement.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of batched tensors ``(observations, actions, next_observations,
            self_state)``. The ``self_state`` element is ``None`` when no
            auxiliary state is stored in the buffer.

        Raises:
            ValueError: If ``batch_size`` is non-positive or exceeds the number
                of stored transitions.
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        with self._lock:
            if len(self._storage) < batch_size:
                raise ValueError("Not enough samples in buffer for requested batch size")
            batch = random.sample(self._storage, batch_size)
        observations, actions, next_observations, self_states = zip(*batch)
        state_tensor: torch.Tensor | None
        if self_states[0] is None:
            if any(state is not None for state in self_states[1:]):
                raise ValueError("Inconsistent self_state entries in rollout buffer")
            state_tensor = None
        else:
            if any(state is None for state in self_states):
                raise ValueError("Inconsistent self_state entries in rollout buffer")
            state_tensor = torch.stack(self_states)
        obs_tensor = torch.stack(observations)
        act_tensor = torch.stack(actions)
        next_tensor = torch.stack(next_observations)
        if torch.cuda.is_available():
            obs_tensor = obs_tensor.pin_memory()
            act_tensor = act_tensor.pin_memory()
            next_tensor = next_tensor.pin_memory()
            if state_tensor is not None:
                state_tensor = state_tensor.pin_memory()
        return (
            obs_tensor,
            act_tensor,
            next_tensor,
            state_tensor,
        )

    def __len__(self) -> int:
        """Return the number of stored transitions."""

        with self._lock:
            return len(self._storage)
