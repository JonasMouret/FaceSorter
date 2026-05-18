from __future__ import annotations
from typing import List
import numpy as np
from numpy.linalg import norm


def cluster_embeddings(embeddings: List[np.ndarray], threshold: float = 0.50) -> List[int]:
    """
    Cluster face embeddings using union-find on cosine similarity.

    Two faces are merged into the same cluster when their cosine similarity
    is >= threshold.  Each isolated face forms its own cluster.

    Returns a list of 0-based integer labels, one per embedding.
    """
    n = len(embeddings)
    if n == 0:
        return []

    normalized = [e / (norm(e) + 1e-8) for e in embeddings]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            sim = float(np.dot(normalized[i], normalized[j]))
            if sim >= threshold:
                union(i, j)

    root_to_label: dict[int, int] = {}
    labels: List[int] = []
    for i in range(n):
        root = find(i)
        if root not in root_to_label:
            root_to_label[root] = len(root_to_label)
        labels.append(root_to_label[root])

    return labels
