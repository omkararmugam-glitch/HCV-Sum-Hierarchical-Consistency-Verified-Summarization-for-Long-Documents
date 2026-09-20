"""Sentence-graph centrality, shared by the centrality investigation and multi-document salience.

Moved here from scripts/probe_centrality.py (FINDINGS 13.4), which now imports it. Graphs are built from
L2-normalised sentence embeddings:
  knn_graph   each sentence linked to its k most similar sentences, symmetrised, weight = cosine
              (edge attribute ``dist`` = 1 - cosine, for shortest paths / betweenness)
  dense_graph every pair with cosine >= floor (LexRank-style)
PageRank on such a graph is LexRank / TextRank (Erkan & Radev 2004; Mihalcea & Tarau 2004).
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def knn_graph(emb: np.ndarray, k: int) -> nx.Graph:
    sims = emb @ emb.T
    np.fill_diagonal(sims, -np.inf)
    n = len(emb)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in np.argsort(-sims[i])[: min(k, n - 1)]:
            w = max(float(sims[i, j]), 1e-6)
            g.add_edge(i, int(j), weight=w, dist=1.0 - w)
    return g


def dense_graph(emb: np.ndarray, floor: float = 0.1) -> nx.Graph:
    sims = emb @ emb.T
    n = len(emb)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= floor:
                w = float(sims[i, j])
                g.add_edge(i, j, weight=w, dist=1.0 - w)
    return g


def centrality(g: nx.Graph, measure: str = "pagerank") -> dict[int, float]:
    """``pagerank`` (hubs of dense groups) or ``betweenness`` (bridges between groups)."""
    if g.number_of_nodes() == 0:
        return {}
    if measure == "pagerank":
        return nx.pagerank(g, weight="weight")
    if measure == "betweenness":
        return nx.betweenness_centrality(g, weight="dist")
    raise ValueError(f"unknown centrality measure {measure!r}")
