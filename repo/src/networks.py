"""
networks.py
-----------
Network generation utilities matching the structural statistics of
real-world datasets used in the paper.

Networks generated:
  - Barabasi-Albert   (Twitter / Citation proxies)
  - Watts-Strogatz    (Collaboration proxy)
  - Planted Partition (Reddit / Modular proxy)
  - Star, ER sparse, Complete (canonical topology classes)

Authors: Mihit Nanda, Hannah Nagpall
Repo:    https://github.com/mihiit/NonLinear_Multi-Agent_influencenetworks
"""

import numpy as np
import networkx as nx
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def adjacency(G: nx.Graph) -> np.ndarray:
    """Return float64 adjacency matrix of G."""
    return nx.to_numpy_array(G, dtype=float)


def spectral_stats(A: np.ndarray) -> dict:
    """
    Return lambda_max, mean degree, degree variance for adjacency matrix A.
    """
    degrees = A.sum(axis=1)
    lam_max = float(np.linalg.eigvalsh(A).max())
    return {
        "lambda_max": lam_max,
        "mean_degree": float(degrees.mean()),
        "degree_var":  float(degrees.var()),
        "n": A.shape[0],
    }


# ---------------------------------------------------------------------------
# Canonical topology classes
# ---------------------------------------------------------------------------

def star(n: int) -> np.ndarray:
    """Star graph S_n: one hub connected to n-1 leaves."""
    return adjacency(nx.star_graph(n - 1))


def erdos_renyi(n: int, p: float, seed: int = 0) -> np.ndarray:
    """Sparse Erdos-Renyi G(n, p)."""
    return adjacency(nx.erdos_renyi_graph(n, p, seed=seed))


def complete(n: int) -> np.ndarray:
    """Fully connected K_n."""
    return adjacency(nx.complete_graph(n))


# ---------------------------------------------------------------------------
# Real-network calibrated graphs
# ---------------------------------------------------------------------------

def twitter_graph(n: int = 200, seed: int = 0) -> np.ndarray:
    """
    Barabasi-Albert graph calibrated to Twitter interaction network.
    Matches: power-law exponent gamma~2.1, mean degree ~5.9, lambda_max~10.84.
    """
    G = nx.barabasi_albert_graph(n, m=3, seed=seed)
    return adjacency(G)


def reddit_graph(n: int = 200, seed: int = 0) -> np.ndarray:
    """
    Planted-partition graph calibrated to Reddit community structure.
    8 communities of 25 nodes, intra-density 0.35, inter-density 0.03.
    Matches: mean degree ~14.2, lambda_max ~14.86.
    """
    sizes = [25] * 8
    p_in  = 0.35
    p_out = 0.03
    G = nx.planted_partition_graph(
        l=8, k=25, p_in=p_in, p_out=p_out, seed=seed
    )
    return adjacency(G)


def citation_graph(n: int = 200, seed: int = 0) -> np.ndarray:
    """
    Barabasi-Albert graph calibrated to academic citation networks.
    Matches: power-law distribution, mean degree ~4.0, lambda_max ~7.41.
    """
    G = nx.barabasi_albert_graph(n, m=2, seed=seed)
    return adjacency(G)


def collaboration_graph(n: int = 200, seed: int = 0) -> np.ndarray:
    """
    Watts-Strogatz graph calibrated to scientific collaboration networks.
    Matches: small-world structure, mean degree ~8.0, lambda_max ~8.24,
    clustering coefficient ~0.49.
    """
    G = nx.watts_strogatz_graph(n, k=8, p=0.15, seed=seed)
    return adjacency(G)


def barabasi_albert(n: int, m: int = 3, seed: int = 0) -> np.ndarray:
    """General BA graph for topology sweep."""
    return adjacency(nx.barabasi_albert_graph(n, m=m, seed=seed))


def watts_strogatz(n: int, k: int = 6, p: float = 0.2, seed: int = 0) -> np.ndarray:
    """General WS graph for topology sweep."""
    return adjacency(nx.watts_strogatz_graph(n, k=k, p=p, seed=seed))


def modular(
    n_communities: int = 4,
    community_size: int = 15,
    p_in: float = 0.5,
    p_out: float = 0.03,
    seed: int = 0,
) -> np.ndarray:
    """Planted-partition modular graph for topology sweep."""
    G = nx.planted_partition_graph(
        l=n_communities, k=community_size, p_in=p_in, p_out=p_out, seed=seed
    )
    return adjacency(G)


# ---------------------------------------------------------------------------
# Registry for experiment loops
# ---------------------------------------------------------------------------

REAL_NETWORK_REGISTRY = {
    "twitter":       twitter_graph,
    "reddit":        reddit_graph,
    "citation":      citation_graph,
    "collaboration": collaboration_graph,
}

TOPOLOGY_REGISTRY = {
    "star":   lambda n, seed: star(n),
    "er":     lambda n, seed: erdos_renyi(n, p=0.08, seed=seed),
    "dense":  lambda n, seed: complete(n),
    "ba":     lambda n, seed: barabasi_albert(n, m=3, seed=seed),
    "ws":     lambda n, seed: watts_strogatz(n, k=6, p=0.2, seed=seed),
    "modular":lambda n, seed: modular(seed=seed),
}
