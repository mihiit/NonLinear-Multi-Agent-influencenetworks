"""
dynamics.py
-----------
Nonlinear influence-recovery dynamics for multi-agent networks.

Implements the update rule:
    x_i(t+1) = (1 - x_i(t)) * [1 - exp(-alpha * p * sum_j A_ij * x_j(t))]
               + x_i(t) * (1 - r)

Authors: Mihit Nanda, Hannah Nagpall
Repo:    https://github.com/mihiit/NonLinear_Multi-Agent_influencenetworks
"""

import numpy as np
from typing import Optional


def step(x: np.ndarray, A: np.ndarray, alpha: float, p: float, r: float) -> np.ndarray:
    """
    One discrete time step of the nonlinear influence-recovery dynamics.

    Parameters
    ----------
    x     : (n,) array of current agent states in [0, 1]
    A     : (n, n) adjacency matrix (symmetric, binary or weighted)
    alpha : influence strength (alpha > 0)
    p     : susceptibility scaling (p in (0, 1])
    r     : per-step recovery rate (r in (0, 1))

    Returns
    -------
    x_new : (n,) array of updated states in [0, 1]
    """
    exposure = A @ x                              # sum_j A_ij x_j  shape (n,)
    activation = (1.0 - x) * (1.0 - np.exp(-alpha * p * exposure))
    recovery   = x * (1.0 - r)
    return np.clip(activation + recovery, 0.0, 1.0)


def simulate(
    A: np.ndarray,
    alpha: float,
    p: float,
    r: float,
    T: int = 200,
    x0: Optional[np.ndarray] = None,
    seed: int = 0,
) -> np.ndarray:
    """
    Run T steps of the dynamics from initial state x0.

    Parameters
    ----------
    A     : (n, n) adjacency matrix
    alpha : influence strength
    p     : susceptibility scaling
    r     : recovery rate
    T     : number of time steps
    x0    : (n,) initial state; if None, sampled Uniform(0, 0.01)
    seed  : random seed for reproducibility

    Returns
    -------
    X : (T+1, n) trajectory matrix; X[0] = x0
    """
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    if x0 is None:
        x0 = rng.uniform(0, 0.01, size=n)
    X = np.empty((T + 1, n))
    X[0] = x0
    for t in range(T):
        X[t + 1] = step(X[t], A, alpha, p, r)
    return X


def steady_state(X: np.ndarray, last: int = 50) -> float:
    """
    Estimate steady-state mean activation from the last `last` steps.

    Parameters
    ----------
    X    : (T+1, n) trajectory from simulate()
    last : number of tail steps to average over

    Returns
    -------
    x_star : scalar steady-state estimate
    """
    return float(X[-last:].mean())


def transient_growth_rate(X: np.ndarray, early: int = 20) -> float:
    """
    Estimate transient growth rate g via OLS on log(mean_x(t)) for t = 1..early.

    Parameters
    ----------
    X     : (T+1, n) trajectory
    early : number of early steps to use

    Returns
    -------
    g : scalar OLS slope (growth rate)
    """
    mean_x = X[1:early + 1].mean(axis=1)           # (early,)
    mean_x = np.maximum(mean_x, 1e-12)             # guard log(0)
    log_mean = np.log(mean_x)
    t = np.arange(1, early + 1, dtype=float)
    # OLS: g = cov(t, log_mean) / var(t)
    g = np.polyfit(t, log_mean, 1)[0]
    return float(g)
