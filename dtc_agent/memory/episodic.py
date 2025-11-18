from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import threading

import numpy as np
import torch
import faiss


@dataclass
class EpisodicEntry:
    context: torch.Tensor
    z_self: torch.Tensor
    slots: torch.Tensor


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
        snapshot: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """Insert key/value pairs into the episodic memory.

        Args:
            key: Tensor of shape ``[batch, key_dim]`` used for FAISS indexing.
            value: Tensor payload associated with each key.

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
                z_self_snapshot, slots_snapshot = snapshot
                z_self_cpu = z_self_snapshot.detach().to(device="cpu").contiguous()
                slots_cpu = slots_snapshot.detach().to(device="cpu").contiguous()
            else:
                z_self_cpu = None
                slots_cpu = None
            self._cpu_index.add(key_np)
            for idx in range(batch):
                context_tensor = value_cpu[idx]
                if z_self_cpu is not None and slots_cpu is not None:
                    entry = EpisodicEntry(
                        context=context_tensor,
                        z_self=z_self_cpu[idx],
                        slots=slots_cpu[idx],
                    )
                else:
                    entry = EpisodicEntry(
                        context=context_tensor,
                        z_self=context_tensor.clone(),
                        slots=context_tensor.unsqueeze(0),
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

    def _evict_oldest(self) -> None:
        with self._lock:
            if not self.values:
                return
            oldest_idx = min(self.values)
            del self.values[oldest_idx]
        # FAISS does not support removing individual entries from IndexFlatL2; this will be
        # revisited when swapping to an IVF index or a rolling rebuild strategy.
