"""
NonLinear Multi-Agent Influence Networks
========================================
Source package for:
  Nanda & Nagpall, "Decoupling Transient Instability and Steady-State
  Persistence in Nonlinear Multi-Agent Influence Networks"

GitHub: https://github.com/mihiit/NonLinear_Multi-Agent_influencenetworks
"""

from .dynamics         import simulate, steady_state, transient_growth_rate, step
from .networks         import (twitter_graph, reddit_graph, citation_graph,
                                collaboration_graph, star, erdos_renyi, complete,
                                barabasi_albert, watts_strogatz, modular,
                                TOPOLOGY_REGISTRY, REAL_NETWORK_REGISTRY, spectral_stats)
from .theory           import (theory_summary, spectral_threshold, mean_field_threshold,
                                decoupling_gap, decoupling_gap_lower_bound,
                                lyapunov_constant, lyapunov_regime_satisfied,
                                persistence_lower_bound)
from .experiments      import alpha_sweep, run_topology_sweep, run_real_network_sweep, phase_diagram
from .agent_experiment import run_agent_experiment, early_late_summary

__all__ = [
    "simulate", "steady_state", "transient_growth_rate", "step",
    "twitter_graph", "reddit_graph", "citation_graph", "collaboration_graph",
    "star", "erdos_renyi", "complete", "barabasi_albert", "watts_strogatz", "modular",
    "TOPOLOGY_REGISTRY", "REAL_NETWORK_REGISTRY", "spectral_stats",
    "theory_summary", "spectral_threshold", "mean_field_threshold",
    "decoupling_gap", "decoupling_gap_lower_bound",
    "lyapunov_constant", "lyapunov_regime_satisfied", "persistence_lower_bound",
    "alpha_sweep", "run_topology_sweep", "run_real_network_sweep", "phase_diagram",
    "run_agent_experiment", "early_late_summary",
]
