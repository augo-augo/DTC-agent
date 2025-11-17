from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _pairwise_l2(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    diff = query[:, None, :] - database[None, :, :]
    return np.sum(diff * diff, axis=2).astype(np.float32)


def _empty_results(batch: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    distances = np.full((batch, k), np.inf, dtype=np.float32)
    indices = -np.ones((batch, k), dtype=np.int64)
    return distances, indices


class _IndexFlatL2:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors = np.empty((0, dim), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError("vectors must be 2D with matching dimensionality")
        if self._vectors.size == 0:
            self._vectors = vectors.copy()
        else:
            self._vectors = np.concatenate([self._vectors, vectors], axis=0)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError("queries must be 2D with matching dimensionality")
        if k <= 0:
            raise ValueError("k must be positive")
        if self._vectors.size == 0:
            return _empty_results(queries.shape[0], k)
        distances = _pairwise_l2(queries, self._vectors)
        top_k_idx = np.argsort(distances, axis=1)[:, :k]
        batch_indices = np.arange(queries.shape[0])[:, None]
        top_k_distances = distances[batch_indices, top_k_idx]
        return top_k_distances, top_k_idx.astype(np.int64)


class _StandardGpuResources:
    pass


class _FaissStub(SimpleNamespace):
    def __init__(self) -> None:
        super().__init__(IndexFlatL2=_IndexFlatL2, StandardGpuResources=_StandardGpuResources)


sys.modules.setdefault("faiss", _FaissStub())
