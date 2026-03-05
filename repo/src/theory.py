"""
theory.py
---------
Analytical quantities from the paper's theoretical results.

Implements:
  - Spectral instability threshold  (linearised, Eq. 3)
  - Mean-field threshold            (Theorem 4.3)
  - Decoupling gap                  (Corollary 4.1 / Theorem 4.4)
  - Lyapunov contraction constant   (Theorem 4.1, Eq. c_formula)
  - Persistence lower bound         (Theorem 4.2 part v)
  - Spectral gap lower bound        (Theorem 4.4)

Authors: Mihit Nanda, Hannah Nagpall
Repo:    https://github.com/mihiit/NonLinear_Multi-Agent_influencenetworks
"""

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Spectral quantities
# ---------------------------------------------------------------------------

def lambda_max(A: np.ndarray) -> float:
    """Largest eigenvalue of adjacency matrix A."""
    return float(np.linalg.eigvalsh(A).max())


def spectral_threshold(A: np.ndarray, p: float, r: float) -> float:
    """
    Classical spectral instability threshold (linearised SIS / Eq. 3):
        alpha*_spectral = r / (p * lambda_max(A))
    """
    return r / (p * lambda_max(A))


def mean_field_threshold(A: np.ndarray, p: float, r: float) -> float:
    """
    Mean-field instability threshold (Theorem 4.3):
        alpha*_MF = r / (p * mean_degree)
    """
    mean_d = float(A.sum(axis=1).mean())
    return r / (p * mean_d)


# ---------------------------------------------------------------------------
# Decoupling gap (Theorem 4.4 / Corollary 4.1)
# ---------------------------------------------------------------------------

def decoupling_gap(A: np.ndarray, p: float, r: float) -> float:
    """
    Decoupling gap Delta_alpha = alpha*_spectral - alpha*_MF.
    Negative value: alpha*_spectral < alpha*_MF (spectral threshold is lower,
    so spectral theory overestimates persistence onset).

    From Corollary 4.1:
        |Delta_alpha| >= r * sigma^2_d / (p * d_bar * (d_bar + sigma^2_d/d_bar))
    """
    return spectral_threshold(A, p, r) - mean_field_threshold(A, p, r)


def decoupling_gap_lower_bound(A: np.ndarray, p: float, r: float) -> float:
    """
    Analytical lower bound on |Delta_alpha| from Theorem 4.4:
        |Delta_alpha| >= r * sigma^2_d / (p * d_bar * (d_bar + sigma^2_d / d_bar))
    """
    degrees = A.sum(axis=1)
    d_bar   = float(degrees.mean())
    sig2    = float(degrees.var())
    if d_bar == 0:
        return 0.0
    return r * sig2 / (p * d_bar * (d_bar + sig2 / d_bar))


def spectral_gap_lower_bound(A: np.ndarray) -> float:
    """
    Lower bound on lambda_max(A) - mean_degree from Theorem 4.4
    via the Collatz-Wielandt formula applied to the degree vector:
        lambda_max >= d_bar + sigma^2_d / d_bar
    """
    degrees = A.sum(axis=1)
    d_bar = float(degrees.mean())
    sig2  = float(degrees.var())
    return sig2 / d_bar if d_bar > 0 else 0.0


# ---------------------------------------------------------------------------
# Lyapunov contraction constant (Theorem 4.1)
# ---------------------------------------------------------------------------

def lyapunov_constant(A: np.ndarray, alpha: float, p: float, r: float) -> float:
    """
    Explicit contraction constant c from Theorem 4.1 (Eq. c_formula):
        c = 1 - 2*(alpha*p*lambda_max)^2 - 2*(1-r)^2

    Returns c if the strengthened regime holds (c > 0), else returns 0.0
    (indicating the theorem's sufficient conditions are not satisfied).

    Strengthened regime: alpha*p*lambda_max(A) < r  AND  r <= 1/sqrt(2).
    """
    lam = lambda_max(A)
    c = 1.0 - 2.0 * (alpha * p * lam) ** 2 - 2.0 * (1.0 - r) ** 2
    return max(0.0, float(c))


def lyapunov_regime_satisfied(A: np.ndarray, alpha: float, p: float, r: float) -> bool:
    """
    Check whether the strengthened Lyapunov regime (Theorem 4.1) holds:
        alpha * p * lambda_max(A) < r   AND   r <= 1/sqrt(2)
    """
    lam = lambda_max(A)
    return (alpha * p * lam < r) and (r <= 1.0 / np.sqrt(2.0))


# ---------------------------------------------------------------------------
# Persistence lower bound (Theorem 4.2 part v)
# ---------------------------------------------------------------------------

def persistence_lower_bound(A: np.ndarray, alpha: float, p: float, r: float) -> float:
    """
    Lower bound on steady-state x* from Theorem 4.2(v):
        x* >= 1 - r / (alpha * p * d_bar)

    Valid when alpha * p * d_bar > r (above mean-field threshold).
    Returns 0 if below threshold.
    """
    d_bar = float(A.sum(axis=1).mean())
    val   = 1.0 - r / (alpha * p * d_bar) if alpha * p * d_bar > r else 0.0
    return max(0.0, float(val))


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def theory_summary(A: np.ndarray, alpha: float, p: float, r: float) -> dict:
    """
    Return a dict of all key theoretical quantities for a given (A, alpha, p, r).
    """
    degrees = A.sum(axis=1)
    return {
        "n":                         A.shape[0],
        "mean_degree":               float(degrees.mean()),
        "degree_var":                float(degrees.var()),
        "lambda_max":                lambda_max(A),
        "alpha_spectral":            spectral_threshold(A, p, r),
        "alpha_mf":                  mean_field_threshold(A, p, r),
        "decoupling_gap":            decoupling_gap(A, p, r),
        "decoupling_gap_lb":         decoupling_gap_lower_bound(A, p, r),
        "spectral_gap_lb":           spectral_gap_lower_bound(A),
        "lyapunov_c":                lyapunov_constant(A, alpha, p, r),
        "lyapunov_regime_ok":        lyapunov_regime_satisfied(A, alpha, p, r),
        "persistence_lb":            persistence_lower_bound(A, alpha, p, r),
    }
