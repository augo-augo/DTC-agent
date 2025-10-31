from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import threading

import numpy as np
import torch
import faiss


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
        self._gpu_resources: faiss.StandardGpuResources | None = None
        self._pending_keys: List[np.ndarray] = []
        self._trained = False
        self._using_gpu = False
        self._nlist = max(1, min(4096, max(1, config.capacity // 8)))
        self._train_threshold = min(config.capacity, max(self._nlist * 2, 32))
        self.index = self._cpu_index
        self._init_index()
        self.values: Dict[int, torch.Tensor] = {}
        self.next_id = 0

    def __len__(self) -> int:
        """Return the number of stored episodic entries."""

        with self._lock:
            return len(self.values)

    def write(self, key: torch.Tensor, value: torch.Tensor) -> None:
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
            self._cpu_index.add(key_np)
            if self._using_gpu:
                self._add_gpu_keys(key_np)
            for idx in range(batch):
                self.values[self.next_id] = value_cpu[idx]
                self.next_id += 1
            if self._using_gpu and not self._trained:
                total_pending = sum(arr.shape[0] for arr in self._pending_keys)
                if total_pending >= self._train_threshold:
                    self._finalize_gpu_index()

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
            backend = self.index if (self._using_gpu and self._trained) else self._cpu_index

            query_np = query.detach().cpu().numpy()
            query_np = np.ascontiguousarray(query_np, dtype=np.float32)

            distances, indices = backend.search(query_np, k)
            fallback = torch.zeros_like(query[0]).cpu()
            retrieved_cpu = [self.values.get(idx, fallback) for idx in indices.flatten()]
            stacked_cpu = torch.stack(retrieved_cpu).view(query.shape[0], k, -1)
            values = stacked_cpu.to(target_device)
            distances_tensor = torch.from_numpy(distances).to(target_device)
            return distances_tensor, values

    def _init_index(self) -> None:
        self.index = self._cpu_index
        self._using_gpu = False
        self._trained = True
        self._pending_keys.clear()

    def _add_gpu_keys(self, key_array: np.ndarray) -> None:
        pass

    def _finalize_gpu_index(self) -> None:
        pass

    def _evict_oldest(self) -> None:
        with self._lock:
            if not self.values:
                return
            oldest_idx = min(self.values)
            del self.values[oldest_idx]
        # FAISS does not support removing individual entries from IndexFlatL2; this will be
        # revisited when swapping to an IVF index or a rolling rebuild strategy.
