from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import threading

import numpy as np
import torch
import faiss


@dataclass
class EpisodicEntry:
    key: torch.Tensor
    context: torch.Tensor
    z_self: torch.Tensor
    slots: torch.Tensor
    action: torch.Tensor | None = None
    target_latent: torch.Tensor | None = None


@dataclass
class EpisodicSnapshot:
    z_self: torch.Tensor
    slots: torch.Tensor
    action: torch.Tensor | None = None
    target_latent: torch.Tensor | None = None


@dataclass
class EpisodicBufferConfig:
    """Configuration for the approximate episodic recall buffer.

    Attributes:
        capacity: Maximum number of entries retained in memory.
        key_dim: Dimensionality of the FAISS key vectors.
    """

    capacity: int
    key_dim: int


class EpisodicBuffer:
    """Thread-safe FAISS index for approximate episodic recall."""

    def __init__(self, config: EpisodicBufferConfig) -> None:
        """Initialize the buffer and underlying FAISS index.

        Args:
            config: Parameters governing capacity and index dimensionality.
        """

        self.config = config
        self._lock = threading.RLock()
        self._cpu_index = faiss.IndexFlatL2(config.key_dim)
        self.values: Dict[int, EpisodicEntry] = {}
        self.next_id = 0

    def __len__(self) -> int:
        """Return the number of stored episodic entries."""

        with self._lock:
            return len(self.values)

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        snapshot: EpisodicSnapshot | None = None,
    ) -> None:
        """Insert key/value pairs into the episodic memory.

        Args:
            key: Tensor of shape ``[batch, key_dim]`` used for FAISS indexing.
            value: Tensor payload associated with each key.
            snapshot: Optional latent/action metadata retained for replay.

        Raises:
            ValueError: If ``key`` is not a 2-D tensor with ``key_dim`` columns.
        """

        with self._lock:
            if key.ndim != 2:
                raise ValueError("key must be shape [batch, key_dim]")
            if key.shape[1] != self.config.key_dim:
                raise ValueError("key dimension mismatch")
            if len(self) >= self.config.capacity:
                self._evict_oldest()
            batch = key.shape[0]
            key_cpu = key.detach().to(dtype=torch.float32, device="cpu").contiguous()
            key_np = np.ascontiguousarray(key_cpu.numpy())
            value_cpu = value.detach().to(device="cpu").contiguous()
            if snapshot is not None:
                z_self_cpu = snapshot.z_self.detach().to(device="cpu").contiguous()
                slots_cpu = snapshot.slots.detach().to(device="cpu").contiguous()
                action_cpu = (
                    snapshot.action.detach().to(device="cpu").contiguous()
                    if snapshot.action is not None
                    else None
                )
                target_cpu = (
                    snapshot.target_latent.detach().to(device="cpu").contiguous()
                    if snapshot.target_latent is not None
                    else None
                )
            else:
                z_self_cpu = None
                slots_cpu = None
                action_cpu = None
                target_cpu = None
            self._cpu_index.add(key_np)
            for idx in range(batch):
                context_tensor = value_cpu[idx]
                entry = EpisodicEntry(
                    key=key_cpu[idx],
                    context=context_tensor,
                    z_self=z_self_cpu[idx] if z_self_cpu is not None else context_tensor.clone(),
                    slots=slots_cpu[idx] if slots_cpu is not None else context_tensor.unsqueeze(0),
                    action=action_cpu[idx] if action_cpu is not None else None,
                    target_latent=target_cpu[idx] if target_cpu is not None else None,
                )
                self.values[self.next_id] = entry
                self.next_id += 1

    def read(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve approximate nearest neighbours for each query vector.

        Args:
            query: Tensor of shape ``[batch, key_dim]`` to match against the
                index.
            k: Number of neighbours to retrieve for each query.

        Returns:
            Tuple ``(distances, values)`` where ``distances`` contains squared
            L2 distances and ``values`` contains the recalled payloads.

        Raises:
            ValueError: If ``query`` is not a 2-D tensor with ``key_dim``
                columns.
        """

        with self._lock:
            if query.ndim != 2:
                raise ValueError("query must be shape [batch, key_dim]")
            target_device = query.device

            query_np = query.detach().cpu().numpy()
            query_np = np.ascontiguousarray(query_np, dtype=np.float32)

            distances, indices = self._cpu_index.search(query_np, k)
        fallback = torch.zeros(self.config.key_dim, device="cpu")
        retrieved_cpu = [
            self.values[idx].context if idx in self.values else fallback
            for idx in indices.flatten()
        ]
        stacked_cpu = torch.stack(retrieved_cpu).view(query.shape[0], k, -1)
        values = stacked_cpu.to(target_device)
        distances_tensor = torch.from_numpy(distances).to(target_device)
        return distances_tensor, values

    def sample_uniform(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample latent snapshots uniformly from stored entries."""

        with self._lock:
            if not self.values:
                raise ValueError("Episodic buffer is empty")
            ids = list(self.values.keys())
            replace = len(ids) < batch_size
            sampled = np.random.choice(ids, size=batch_size, replace=replace)
            z_self_list = []
            slots_list = []
            for idx in sampled:
                entry = self.values[idx]
                z_self_list.append(entry.z_self)
                slots_list.append(entry.slots)
            z_self_tensor = torch.stack(z_self_list).to(device)
            slots_tensor = torch.stack(slots_list).to(device)
            return z_self_tensor, slots_tensor

    def sample_transitions(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample latent transitions containing (state, action, next_state) tuples."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        with self._lock:
            valid_ids = [
                idx
                for idx, entry in self.values.items()
                if entry.action is not None and entry.target_latent is not None
            ]
            if not valid_ids:
                raise ValueError("No episodic transitions with stored actions are available")
            replace = len(valid_ids) < batch_size
            sampled = np.random.choice(valid_ids, size=batch_size, replace=replace)
            latent_state_list: List[torch.Tensor] = []
            action_list: List[torch.Tensor] = []
            target_list: List[torch.Tensor] = []
            for idx in sampled:
                entry = self.values[idx]
                assert entry.action is not None and entry.target_latent is not None
                latent_state_list.append(entry.context)
                action_list.append(entry.action)
                target_list.append(entry.target_latent)
            latent_tensor = torch.stack(latent_state_list)
            action_tensor = torch.stack(action_list)
            target_tensor = torch.stack(target_list)
            if torch.cuda.is_available():
                latent_tensor = latent_tensor.pin_memory()
                action_tensor = action_tensor.pin_memory()
                target_tensor = target_tensor.pin_memory()
        return (
            latent_tensor.to(device, non_blocking=True),
            action_tensor.to(device, non_blocking=True),
            target_tensor.to(device, non_blocking=True),
        )

    def _evict_oldest(self) -> None:
        with self._lock:
            if not self.values:
                return
            oldest_idx = min(self.values)
            del self.values[oldest_idx]
            # Lazy rebuild: Only rebuild if we've drifted too far
            # The index will contain "ghost keys" (indices that are no longer in self.values)
            # which read() must handle by checking self.values.
            if self._cpu_index.ntotal > self.config.capacity * 2:
                self._rebuild_index_locked()

    def _rebuild_index_locked(self) -> None:
        """Rebuild the FAISS index from currently stored keys."""

        new_index = faiss.IndexFlatL2(self.config.key_dim)
        if self.values:
            # Sort by ID to maintain temporal order in the index (optional but good for debugging)
            sorted_ids = sorted(self.values.keys())
            key_stack = torch.stack([self.values[idx].key for idx in sorted_ids])
            key_np = np.ascontiguousarray(key_stack.numpy(), dtype=np.float32)
            new_index.add(key_np)
            
            # CRITICAL: We must re-map the IDs in self.values to match the new index (0..N-1)
            # Otherwise, the new index returns 0..N-1, but self.values has old large IDs.
            new_values = {}
            for new_id, old_id in enumerate(sorted_ids):
                new_values[new_id] = self.values[old_id]
            self.values = new_values
            self.next_id = len(self.values)
            
        self._cpu_index = new_index
