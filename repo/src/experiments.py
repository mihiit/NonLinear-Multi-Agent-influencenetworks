"""
experiments.py
--------------
Monte Carlo experiment framework: sweeps over alpha, topology, and
parameters, averaging over multiple graph realisations and trials.

Reproduces:
  - Phase diagram (Fig. 2)
  - Spectral breakdown (Fig. 4 / Sec. 6)
  - Real-network experiments (Fig. 8 / Sec. 7)
  - Extended topology study (Fig. 9 / Sec. 8)
  - Ablation study (Sec. 11)

Authors: Mihit Nanda, Hannah Nagpall
Repo:    https://github.com/mihiit/NonLinear_Multi-Agent_influencenetworks
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .dynamics import simulate, steady_state, transient_growth_rate
from .networks import TOPOLOGY_REGISTRY, REAL_NETWORK_REGISTRY, spectral_stats


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    """Stores results of a single (topology, alpha) sweep."""
    alpha_values:  np.ndarray
    x_star_mean:   np.ndarray          # steady-state mean over trials
    x_star_ci:     np.ndarray          # 95% CI half-width
    g_mean:        np.ndarray          # transient growth rate mean
    g_ci:          np.ndarray          # 95% CI half-width
    topology_name: str = ""
    stats:         dict = field(default_factory=dict)   # spectral stats


# ---------------------------------------------------------------------------
# Core sweep function
# ---------------------------------------------------------------------------

def alpha_sweep(
    A_fn: Callable[[int, int], np.ndarray],
    alpha_values: np.ndarray,
    p: float = 0.5,
    r: float = 0.02,
    n: int = 60,
    T: int = 200,
    n_trials: int = 100,
    n_graphs: int = 20,
    topology_name: str = "",
) -> SweepResult:
    """
    Sweep over alpha values; average x* and g over n_trials x n_graphs.

    Parameters
    ----------
    A_fn         : callable(n, seed) -> adjacency matrix
    alpha_values : 1-D array of alpha values to sweep
    p, r         : model parameters
    n            : number of nodes
    T            : simulation steps
    n_trials     : Monte Carlo trials per (graph, alpha)
    n_graphs     : number of graph realisations
    topology_name: label for the result

    Returns
    -------
    SweepResult
    """
    n_alpha   = len(alpha_values)
    all_xstar = np.zeros((n_alpha, n_graphs * n_trials))
    all_g     = np.zeros((n_alpha, n_graphs * n_trials))

    for gi in range(n_graphs):
        A = A_fn(n, seed=gi)
        for ai, alpha in enumerate(alpha_values):
            for ti in range(n_trials):
                X = simulate(A, alpha, p, r, T=T, seed=gi * n_trials + ti)
                all_xstar[ai, gi * n_trials + ti] = steady_state(X)
                all_g[ai, gi * n_trials + ti]     = transient_growth_rate(X)

    # Compute mean and 95% CI
    ci_factor = 1.96 / np.sqrt(n_graphs * n_trials)
    A_sample  = A_fn(n, seed=0)

    return SweepResult(
        alpha_values  = alpha_values,
        x_star_mean   = all_xstar.mean(axis=1),
        x_star_ci     = all_xstar.std(axis=1) * ci_factor,
        g_mean        = all_g.mean(axis=1),
        g_ci          = all_g.std(axis=1) * ci_factor,
        topology_name = topology_name,
        stats         = spectral_stats(A_sample),
    )


# ---------------------------------------------------------------------------
# Phase diagram
# ---------------------------------------------------------------------------

def phase_diagram(
    A_fn: Callable,
    alpha_grid: np.ndarray,
    r_grid: np.ndarray,
    p: float = 0.5,
    n: int = 60,
    T: int = 200,
    n_trials: int = 20,
    n_graphs: int = 5,
) -> np.ndarray:
    """
    Compute 2-D phase diagram x*(alpha, r) over a grid.

    Returns
    -------
    Z : (len(r_grid), len(alpha_grid)) array of steady-state values
    """
    Z = np.zeros((len(r_grid), len(alpha_grid)))
    for ri, r in enumerate(r_grid):
        for ai, alpha in enumerate(alpha_grid):
            vals = []
            for gi in range(n_graphs):
                A = A_fn(n, seed=gi)
                for ti in range(n_trials):
                    X = simulate(A, alpha, p, r, T=T, seed=gi * n_trials + ti)
                    vals.append(steady_state(X))
            Z[ri, ai] = float(np.mean(vals))
    return Z


# ---------------------------------------------------------------------------
# Convenience: run all topology sweeps
# ---------------------------------------------------------------------------

def run_topology_sweep(
    alpha_values: np.ndarray,
    topologies: Optional[List[str]] = None,
    n: int = 60,
    p: float = 0.5,
    r: float = 0.02,
    T: int = 200,
    n_trials: int = 100,
    n_graphs: int = 20,
) -> Dict[str, SweepResult]:
    """
    Run alpha sweep for multiple topology classes.

    Parameters
    ----------
    topologies : list of keys from TOPOLOGY_REGISTRY; defaults to all

    Returns
    -------
    dict mapping topology name -> SweepResult
    """
    if topologies is None:
        topologies = list(TOPOLOGY_REGISTRY.keys())
    results = {}
    for name in topologies:
        A_fn = TOPOLOGY_REGISTRY[name]
        results[name] = alpha_sweep(
            A_fn, alpha_values, p=p, r=r, n=n,
            T=T, n_trials=n_trials, n_graphs=n_graphs,
            topology_name=name,
        )
    return results


def run_real_network_sweep(
    alpha_values: np.ndarray,
    datasets: Optional[List[str]] = None,
    n: int = 200,
    p: float = 0.5,
    r: float = 0.02,
    T: int = 200,
    n_trials: int = 100,
    n_graphs: int = 20,
) -> Dict[str, SweepResult]:
    """
    Run alpha sweep for real-network calibrated datasets.
    """
    if datasets is None:
        datasets = list(REAL_NETWORK_REGISTRY.keys())
    results = {}
    for name in datasets:
        A_fn = REAL_NETWORK_REGISTRY[name]
        results[name] = alpha_sweep(
            A_fn, alpha_values, p=p, r=r, n=n,
            T=T, n_trials=n_trials, n_graphs=n_graphs,
            topology_name=name,
        )
    return results
