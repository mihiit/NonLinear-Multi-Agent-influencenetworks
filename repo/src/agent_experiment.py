"""
agent_experiment.py
-------------------
AI agent misinformation experiment (Section 10 of the paper).

Simulates n=30 language-model-style agents exchanging beliefs
across three network architectures:
  - Hub (Barabasi-Albert)     : exposure-driven transient instability
  - Dense (K_n)               : reinforcement-driven persistence
  - Modular (planted partition): community-contained propagation

The source node receives a persistent external misinformation injection
at every step (modelling a compromised or adversarial agent).

Authors: Mihit Nanda, Hannah Nagpall
Repo:    https://github.com/mihiit/NonLinear_Multi-Agent_influencenetworks
"""

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .dynamics import step
from .networks import adjacency


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class AgentRunResult:
    """Results for one network architecture over T rounds."""
    network_name:  str
    A:             np.ndarray
    X:             np.ndarray        # (T+1, n) belief trajectory
    source_node:   int
    mean_belief:   np.ndarray        # (T+1,) mean over all agents
    alpha:         float
    p:             float
    r:             float


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def build_agent_networks(n: int = 30, seed: int = 0) -> Dict[str, Tuple[np.ndarray, int]]:
    """
    Build the three agent network architectures used in the paper.

    Returns
    -------
    dict mapping name -> (adjacency_matrix, source_node_index)
    """
    rng = np.random.default_rng(seed)

    # Hub: BA graph, source = highest-degree node
    G_hub  = nx.barabasi_albert_graph(n, m=2, seed=seed)
    A_hub  = adjacency(G_hub)
    src_hub = int(np.argmax(A_hub.sum(axis=1)))

    # Dense: complete graph K_n, source = node 0
    A_dense = adjacency(nx.complete_graph(n))
    src_dense = 0

    # Modular: 3 communities of 10, source = one per community
    G_mod  = nx.planted_partition_graph(l=3, k=10, p_in=0.6, p_out=0.05, seed=seed)
    A_mod  = adjacency(G_mod)
    src_mod = 0    # node 0 is in community 1

    return {
        "hub":     (A_hub,   src_hub),
        "dense":   (A_dense, src_dense),
        "modular": (A_mod,   src_mod),
    }


def run_agent_experiment(
    n: int = 30,
    alpha: float = 0.12,
    p: float = 0.6,
    r: float = 0.03,
    T: int = 120,
    seed: int = 0,
) -> Dict[str, AgentRunResult]:
    """
    Run the AI agent misinformation experiment across all three architectures.

    Parameters
    ----------
    n     : number of agents
    alpha : influence strength
    p     : susceptibility scaling
    r     : fact-checking / self-correction rate
    T     : number of communication rounds
    seed  : random seed

    Returns
    -------
    dict mapping architecture name -> AgentRunResult
    """
    rng      = np.random.default_rng(seed)
    networks = build_agent_networks(n=n, seed=seed)
    results  = {}

    for name, (A, src) in networks.items():
        X = np.zeros((T + 1, n))
        X[0] = rng.uniform(0, 0.01, size=n)
        X[0, src] = 1.0       # source starts fully misinformed

        for t in range(T):
            x_next      = step(X[t], A, alpha, p, r)
            x_next[src] = 1.0  # persistent injection: source never recovers
            X[t + 1]    = x_next

        results[name] = AgentRunResult(
            network_name = name,
            A            = A,
            X            = X,
            source_node  = src,
            mean_belief  = X.mean(axis=1),
            alpha        = alpha,
            p            = p,
            r            = r,
        )

    return results


def early_late_summary(
    results: Dict[str, AgentRunResult],
    early_t: int = 20,
    late_t: int = 120,
) -> Dict[str, Dict[str, float]]:
    """
    Extract early and late mean belief levels for each architecture.

    Returns
    -------
    dict mapping name -> {"early": float, "late": float}
    """
    summary = {}
    for name, res in results.items():
        summary[name] = {
            "early": float(res.mean_belief[min(early_t, len(res.mean_belief) - 1)]),
            "late":  float(res.mean_belief[min(late_t,  len(res.mean_belief) - 1)]),
        }
    return summary
