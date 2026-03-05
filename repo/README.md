# Nonlinear Multi-Agent Influence Networks

**Decoupling Transient Instability and Steady-State Persistence in Nonlinear Multi-Agent Influence Networks**

*Mihit Nanda¹ · Hannah Nagpall²*

¹ IILM University Greater Noida — mihit.nanda.cs27@iilm.edu
² Texas A&M University–Kingsville — hannah.nagpall@students.tamuk.edu

---

## Overview

This repository contains the full source code for the paper. The central result is that **classical spectral stability theory accurately predicts the onset of small-perturbation growth in linear diffusion models, but when nonlinear saturation is introduced the spectral threshold no longer fully characterises long-run dynamics.** Transient instability and steady-state persistence become governed by structurally distinct mechanisms — the **dual fragility modes** — whose separation is quantified by the **degree variance theorem** (Theorem 4.4).

---

## Repository Structure

```
NonLinear_Multi-Agent_influencenetworks/
│
├── src/                        # Core Python package
│   ├── __init__.py             # Public API
│   ├── dynamics.py             # Nonlinear update rule, simulation engine
│   ├── networks.py             # Network generators (real-calibrated + canonical)
│   ├── theory.py               # Analytical quantities (thresholds, Lyapunov c, bounds)
│   ├── experiments.py          # Monte Carlo sweep framework
│   ├── agent_experiment.py     # AI agent misinformation experiment
│   └── plotting.py             # Figure generation (publication quality)
│
├── tests/                      # pytest unit tests
│   ├── __init__.py
│   └── test_dynamics.py        # Tests for dynamics, theory, networks
│
├── notebooks/
│   └── exploration.ipynb       # Interactive Jupyter walkthrough
│
├── paper/                      # LaTeX source
│   ├── paper_v8.tex            # Final paper source
│   ├── refs.bib                # Bibliography
│   └── figures/                # Generated figure JPGs
│
├── figures/                    # Output directory for generated figures
│
├── generate_figures.py         # Reproduce all paper figures: python generate_figures.py
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/mihiit/NonLinear_Multi-Agent_influencenetworks.git
cd NonLinear_Multi-Agent_influencenetworks

# 2. Install dependencies
pip install -r requirements.txt

# 3. Reproduce all figures
python generate_figures.py

# 4. Run tests
pytest tests/ -v

# 5. Interactive exploration
jupyter notebook notebooks/exploration.ipynb
```

---

## Key Modules

### `src/dynamics.py`
Implements the nonlinear influence–recovery update rule:

```
x_i(t+1) = (1 - x_i(t)) * [1 - exp(-alpha * p * sum_j A_ij * x_j(t))]
           + x_i(t) * (1 - r)
```

Functions: `step`, `simulate`, `steady_state`, `transient_growth_rate`

### `src/networks.py`
Generates all network types used in the paper:
- **Real-calibrated**: `twitter_graph`, `reddit_graph`, `citation_graph`, `collaboration_graph`
- **Canonical topologies**: `star`, `erdos_renyi`, `complete`, `barabasi_albert`, `watts_strogatz`, `modular`

### `src/theory.py`
All analytical quantities from the paper:

| Function | Paper reference |
|---|---|
| `spectral_threshold` | Eq. (3) |
| `mean_field_threshold` | Theorem 4.3 |
| `decoupling_gap` | Corollary 4.1 |
| `decoupling_gap_lower_bound` | Theorem 4.4 |
| `lyapunov_constant` | Theorem 4.1, Eq. (c_formula) |
| `persistence_lower_bound` | Theorem 4.2(v) |

### `src/experiments.py`
Monte Carlo sweep framework with `alpha_sweep`, `run_topology_sweep`, `run_real_network_sweep`, `phase_diagram`.

### `src/agent_experiment.py`
AI agent misinformation experiment (Section 10): 30 agents, three architectures (hub BA, dense K_n, modular), persistent source injection.

---

## Theoretical Results Summary

| Theorem | Statement |
|---|---|
| **Thm 4.1** (Lyapunov) | Global geometric convergence to zero under strengthened regime: `alpha*p*lambda_max < r` and `r <= 1/sqrt(2)`. Explicit rate `c = 1 - 2(alpha*p*lambda_max)^2 - 2(1-r)^2`. |
| **Thm 4.2** (Persistence) | Five properties of endemic equilibrium above threshold, including lower bound `x* >= 1 - r/(alpha*p*d_bar)`. |
| **Thm 4.3** (Bifurcation) | Transcritical bifurcation at `alpha*_MF = r/(p*d_bar)`. |
| **Thm 4.4** (Spectral gap) | `lambda_max >= d_bar + sigma^2_d / d_bar`; decoupling gap scales as `O(sigma^2_d / d_bar^2)`. |
| **Cor 4.1** (Decoupling) | `|Delta_alpha| >= r*sigma^2_d / (p*d_bar*(d_bar + sigma^2_d/d_bar))`. Zero on regular graphs; grows with degree heterogeneity. |

---

## Citation

```bibtex
@article{nanda2025nonlinear,
  title   = {Decoupling Transient Instability and Steady-State Persistence
             in Nonlinear Multi-Agent Influence Networks},
  author  = {Nanda, Mihit and Nagpall, Hannah},
  year    = {2025},
  url     = {https://github.com/mihiit/NonLinear_Multi-Agent_influencenetworks}
}
```

---

## License

MIT License. See `LICENSE` for details.
