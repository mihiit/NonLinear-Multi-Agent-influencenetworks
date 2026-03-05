"""
plotting.py
-----------
Publication-quality figure generation for all paper figures.

Colour palette (matching paper):
  NAV   = #1A3A6B  (navy blue)
  STEEL = #4682B4
  RED   = #C0392B
  GRN   = #27AE60
  ORG   = #E67E22
  PRP   = #8E44AD
  TEAL  = #16A085
  GOLD  = #B8860B

Authors: Mihit Nanda, Hannah Nagpall
Repo:    https://github.com/mihiit/NonLinear_Multi-Agent_influencenetworks
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Optional

from .experiments import SweepResult


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
PALETTE = {
    "nav":   "#1A3A6B",
    "steel": "#4682B4",
    "red":   "#C0392B",
    "grn":   "#27AE60",
    "org":   "#E67E22",
    "prp":   "#8E44AD",
    "teal":  "#16A085",
    "gold":  "#B8860B",
}

TOPO_COLORS = {
    "star":    PALETTE["red"],
    "er":      PALETTE["grn"],
    "dense":   PALETTE["steel"],
    "ba":      PALETTE["org"],
    "ws":      PALETTE["teal"],
    "modular": PALETTE["prp"],
    "twitter":       PALETTE["red"],
    "reddit":        PALETTE["org"],
    "citation":      PALETTE["steel"],
    "collaboration": PALETTE["grn"],
}

AGENT_COLORS = {
    "hub":     PALETTE["red"],
    "dense":   PALETTE["steel"],
    "modular": PALETTE["prp"],
}

plt.rcParams.update({
    "font.family":     "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi":      160,
})


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: str):
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 1 – topology sweep (x* and g vs alpha)
# ---------------------------------------------------------------------------

def plot_topology_sweep(
    results: Dict[str, SweepResult],
    save_path: str = "figures/fig_topology_sweep.jpg",
    show_spectral: bool = True,
    p: float = 0.5,
    r: float = 0.02,
):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    for name, res in results.items():
        col = TOPO_COLORS.get(name, "grey")
        lam = res.stats.get("lambda_max", 1.0)
        alpha_spec = r / (p * lam) if lam > 0 else 0

        axes[0].plot(res.alpha_values, res.x_star_mean, color=col, lw=1.8, label=name)
        axes[0].fill_between(
            res.alpha_values,
            res.x_star_mean - res.x_star_ci,
            res.x_star_mean + res.x_star_ci,
            color=col, alpha=0.15,
        )
        axes[1].plot(res.alpha_values, res.g_mean, color=col, lw=1.8, label=name)

    axes[0].set_xlabel(r"$\alpha$"); axes[0].set_ylabel(r"Steady-state $x^*$")
    axes[1].set_xlabel(r"$\alpha$"); axes[1].set_ylabel(r"Transient growth rate $g$")
    axes[0].set_title("Steady-State Persistence", fontsize=9)
    axes[1].set_title("Transient Instability",    fontsize=9)

    handles = [mpatches.Patch(color=TOPO_COLORS.get(n, "grey"), label=n)
               for n in results.keys()]
    axes[1].legend(handles=handles, loc="upper left", frameon=False)
    fig.tight_layout()
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 2 – real network experiments
# ---------------------------------------------------------------------------

def plot_real_network(
    results: Dict[str, SweepResult],
    save_path: str = "figures/fig_real_networks.jpg",
    p: float = 0.5,
    r: float = 0.02,
):
    n_nets = len(results)
    fig, axes = plt.subplots(1, n_nets, figsize=(3.5 * n_nets, 3.2))
    if n_nets == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        col = TOPO_COLORS.get(name, PALETTE["nav"])
        lam = res.stats.get("lambda_max", 1.0)
        alpha_spec = r / (p * lam)

        ax2 = ax.twinx()
        ax.plot(res.alpha_values, res.x_star_mean, color=col, lw=1.8, label=r"$x^*$")
        ax2.plot(res.alpha_values, res.g_mean, color=col, lw=1.8,
                 linestyle="--", alpha=0.7, label=r"$g$")
        ax.axvline(alpha_spec, color=PALETTE["gold"], lw=1.2,
                   linestyle=":", label=r"$\alpha^*_{\rm spec}$")

        ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(r"$x^*$", color=col)
        ax2.set_ylabel(r"$g$", color=col, alpha=0.7)
        ax.set_title(name.capitalize(), fontsize=9)
        d_bar = res.stats.get("mean_degree", 0)
        lam_v = res.stats.get("lambda_max", 0)
        ax.text(0.98, 0.05, fr"$\bar{{d}}$={d_bar:.1f}, $\lambda$={lam_v:.1f}",
                transform=ax.transAxes, ha="right", fontsize=7)

    fig.tight_layout()
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 3 – AI agent misinformation
# ---------------------------------------------------------------------------

def plot_agent_experiment(
    results: dict,
    early_t: int = 20,
    late_t: int = 120,
    save_path: str = "figures/fig_agent_experiment.jpg",
):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    # Time series
    for name, res in results.items():
        col = AGENT_COLORS.get(name, "grey")
        axes[0].plot(res.mean_belief, color=col, lw=1.8, label=name)

    axes[0].axvline(early_t, color="grey", lw=0.8, linestyle=":")
    axes[0].axvline(late_t,  color="grey", lw=0.8, linestyle=":")
    axes[0].set_xlabel("Communication round $t$")
    axes[0].set_ylabel("Mean misinformation belief")
    axes[0].legend(frameon=False)

    # Early vs late bar chart
    names  = list(results.keys())
    early  = [results[n].mean_belief[min(early_t, len(results[n].mean_belief)-1)] for n in names]
    late   = [results[n].mean_belief[min(late_t,  len(results[n].mean_belief)-1)] for n in names]
    x      = np.arange(len(names))
    w      = 0.35
    bars_e = axes[1].bar(x - w/2, early, w, label=f"$t={early_t}$ (early)",
                         color=[AGENT_COLORS.get(n, "grey") for n in names], alpha=0.6)
    bars_l = axes[1].bar(x + w/2, late,  w, label=f"$t={late_t}$ (late)",
                         color=[AGENT_COLORS.get(n, "grey") for n in names], alpha=1.0)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names)
    axes[1].set_ylabel("Mean belief level")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 4 – Lyapunov convergence
# ---------------------------------------------------------------------------

def plot_lyapunov(
    X_stable: np.ndarray,
    X_unstable: np.ndarray,
    save_path: str = "figures/fig_lyapunov.jpg",
):
    fig, ax = plt.subplots(figsize=(4.5, 3))
    V_stable   = (X_stable   ** 2).sum(axis=1)
    V_unstable = (X_unstable ** 2).sum(axis=1)
    ax.semilogy(V_stable,   color=PALETTE["grn"],  lw=1.8, label=r"$\alpha<\alpha^*$ (stable)")
    ax.semilogy(V_unstable, color=PALETTE["red"],   lw=1.8, label=r"$\alpha>\alpha^*$ (unstable)")
    ax.set_xlabel("Time step $t$")
    ax.set_ylabel(r"$V(\mathbf{x}(t)) = \|\mathbf{x}(t)\|^2$")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, save_path)
