"""
generate_figures.py
-------------------
Reproduce all figures from the paper with a single command:

    python generate_figures.py

Figures are saved to figures/ directory.

Authors: Mihit Nanda, Hannah Nagpall
Repo:    https://github.com/mihiit/NonLinear_Multi-Agent_influencenetworks
"""

import os
import sys
import numpy as np

# Ensure repo root is on path when run as script
sys.path.insert(0, os.path.dirname(__file__))

from src.experiments      import run_topology_sweep, run_real_network_sweep
from src.agent_experiment import run_agent_experiment
from src.dynamics         import simulate
from src.networks         import erdos_renyi
from src.plotting         import (plot_topology_sweep, plot_real_network,
                                   plot_agent_experiment, plot_lyapunov)

os.makedirs("figures", exist_ok=True)

# ── Parameters (matching paper) ─────────────────────────────────────────────
ALPHA = np.arange(0, 1.01, 0.02)
P     = 0.5
R     = 0.02
N     = 60
T     = 200
TRIALS    = 100
GRAPHS    = 20


def main():
    print("=" * 60)
    print("Generating paper figures...")
    print("=" * 60)

    # Fig 1 – topology sweep (all 6 topologies)
    print("\n[1/4] Topology sweep (6 classes)...")
    topo_results = run_topology_sweep(
        ALPHA, n=N, p=P, r=R, T=T,
        n_trials=TRIALS, n_graphs=GRAPHS,
    )
    plot_topology_sweep(topo_results, save_path="figures/fig_topology_sweep.jpg")

    # Fig 2 – real-network calibrated experiments
    print("\n[2/4] Real-network experiments...")
    real_results = run_real_network_sweep(
        ALPHA, n=200, p=P, r=R, T=T,
        n_trials=TRIALS, n_graphs=GRAPHS,
    )
    plot_real_network(real_results, save_path="figures/fig_real_networks.jpg")

    # Fig 3 – AI agent misinformation
    print("\n[3/4] AI agent misinformation experiment...")
    agent_results = run_agent_experiment(n=30, alpha=0.12, p=0.6, r=0.03, T=120)
    plot_agent_experiment(agent_results, save_path="figures/fig_agent_experiment.jpg")

    # Fig 4 – Lyapunov convergence
    print("\n[4/4] Lyapunov convergence...")
    A_test = erdos_renyi(n=50, p=0.12, seed=42)
    alpha_stable   = 0.01
    alpha_unstable = 0.5
    X_stable   = simulate(A_test, alpha_stable,   P, R, T=100, seed=0)
    X_unstable = simulate(A_test, alpha_unstable, P, R, T=100, seed=0)
    plot_lyapunov(X_stable, X_unstable, save_path="figures/fig_lyapunov.jpg")

    print("\n" + "=" * 60)
    print("All figures saved to figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
