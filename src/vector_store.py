from __future__ import annotations

import os
from typing import List, Tuple

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.person_ids: List[str] = []

    def add(self, embedding: np.ndarray, person_id: str) -> None:
        if embedding.shape[-1] != self.dim:
            raise ValueError(f"Embedding dim {embedding.shape[-1]} != {self.dim}")
        vector = embedding.astype("float32").reshape(1, -1)
        self.index.add(vector)
        self.person_ids.append(person_id)

    def search(self, embedding: np.ndarray, top_k: int = 1) -> Tuple[List[str], List[float]]:
        if self.index.ntotal == 0:
            raise ValueError("Vector index is empty. Enroll first.")
        vector = embedding.astype("float32").reshape(1, -1)
        scores, indices = self.index.search(vector, top_k)
        ids = []
        sims = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            ids.append(self.person_ids[idx])
            sims.append(float(score))
        return ids, sims

    def save(self, index_path: str, meta_path: str) -> None:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            for person_id in self.person_ids:
                f.write(f"{person_id}\n")

    @classmethod
    def load(cls, index_path: str, meta_path: str) -> "VectorStore":
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("Index or metadata not found. Enroll first.")
        store = cls()
        store.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            store.person_ids = [line.strip() for line in f if line.strip()]
        if store.index.ntotal != len(store.person_ids):
            raise ValueError(
                "Index and metadata size mismatch. "
                "Delete database/ and re-run enrollment."
            )
        return store
