from typing import List, Tuple

import numpy as np

from .vector_store import VectorStore


def find_best_match(store: VectorStore, query_embedding: np.ndarray) -> Tuple[str, float]:
    ids, scores = store.search(query_embedding, top_k=1)
    if not ids:
        raise ValueError("No matches found in vector store.")
    return ids[0], scores[0]


def find_top_matches(
    store: VectorStore, query_embedding: np.ndarray, top_k: int = 5
) -> Tuple[List[str], List[float]]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1.")
    ids, scores = store.search(query_embedding, top_k=top_k)
    if not ids:
        raise ValueError("No matches found in vector store.")
    return ids, scores
