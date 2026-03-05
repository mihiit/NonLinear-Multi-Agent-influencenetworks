"""
tests/test_dynamics.py
----------------------
Unit tests for the dynamics, theory, and network modules.

Run with:  pytest tests/ -v
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dynamics  import step, simulate, steady_state, transient_growth_rate
from src.networks  import star, erdos_renyi, complete, twitter_graph, spectral_stats
from src.theory    import (spectral_threshold, mean_field_threshold, decoupling_gap,
                            lyapunov_constant, lyapunov_regime_satisfied,
                            persistence_lower_bound, theory_summary)


# ─────────────────────────────────────────────
# Dynamics
# ─────────────────────────────────────────────

class TestStep:
    def test_output_in_unit_interval(self):
        A = erdos_renyi(20, p=0.2, seed=0)
        x = np.random.default_rng(0).uniform(0, 1, 20)
        x_new = step(x, A, alpha=0.3, p=0.5, r=0.05)
        assert x_new.min() >= 0.0 - 1e-9
        assert x_new.max() <= 1.0 + 1e-9

    def test_zero_state_is_fixed_point(self):
        A = complete(10)
        x = np.zeros(10)
        x_new = step(x, A, alpha=1.0, p=1.0, r=0.1)
        np.testing.assert_allclose(x_new, np.zeros(10), atol=1e-12)

    def test_shape_preserved(self):
        A = star(15)
        x = np.ones(15) * 0.5
        assert step(x, A, 0.2, 0.5, 0.03).shape == (15,)


class TestSimulate:
    def test_trajectory_shape(self):
        A = erdos_renyi(30, p=0.1, seed=1)
        X = simulate(A, alpha=0.1, p=0.5, r=0.02, T=50)
        assert X.shape == (51, 30)

    def test_trajectory_stays_in_unit_cube(self):
        A = complete(20)
        X = simulate(A, alpha=0.5, p=0.5, r=0.02, T=100)
        assert X.min() >= -1e-9
        assert X.max() <= 1.0 + 1e-9

    def test_reproducibility(self):
        A = twitter_graph(n=50, seed=7)
        X1 = simulate(A, 0.2, 0.5, 0.02, T=30, seed=42)
        X2 = simulate(A, 0.2, 0.5, 0.02, T=30, seed=42)
        np.testing.assert_array_equal(X1, X2)

    def test_below_threshold_decays_to_zero(self):
        """Below spectral threshold, system should decay to near-zero."""
        A  = erdos_renyi(40, p=0.1, seed=3)
        st = spectral_stats(A)
        # Use alpha well below threshold
        alpha_spec = 0.02 / (0.5 * st["lambda_max"])
        alpha      = alpha_spec * 0.3
        X = simulate(A, alpha, p=0.5, r=0.02, T=300, seed=0)
        assert steady_state(X) < 0.05


class TestMetrics:
    def test_steady_state_range(self):
        A = complete(20)
        X = simulate(A, 0.5, 0.5, 0.02, T=200)
        xs = steady_state(X)
        assert 0.0 <= xs <= 1.0

    def test_transient_growth_above_threshold_positive(self):
        A = complete(20)
        # Dense network, high alpha → should have positive growth
        g = transient_growth_rate(simulate(A, 0.8, 0.5, 0.02, T=50))
        assert g > 0.0


# ─────────────────────────────────────────────
# Networks
# ─────────────────────────────────────────────

class TestNetworks:
    def test_star_shape(self):
        A = star(10)
        assert A.shape == (10, 10)

    def test_star_symmetry(self):
        A = star(10)
        np.testing.assert_array_equal(A, A.T)

    def test_complete_all_ones_off_diag(self):
        A = complete(5)
        off = A - np.diag(np.diag(A))
        np.testing.assert_allclose(off, np.ones((5, 5)) - np.eye(5))

    def test_spectral_stats_keys(self):
        A = twitter_graph(n=50, seed=0)
        s = spectral_stats(A)
        for key in ("lambda_max", "mean_degree", "degree_var", "n"):
            assert key in s

    def test_spectral_stats_lambda_max_positive(self):
        A = twitter_graph(n=50, seed=0)
        assert spectral_stats(A)["lambda_max"] > 0


# ─────────────────────────────────────────────
# Theory
# ─────────────────────────────────────────────

class TestTheory:
    def setup_method(self):
        self.A = erdos_renyi(40, p=0.12, seed=5)
        self.p = 0.5
        self.r = 0.02

    def test_spectral_threshold_positive(self):
        assert spectral_threshold(self.A, self.p, self.r) > 0

    def test_mf_threshold_geq_spectral(self):
        """mean-field threshold >= spectral threshold (lambda_max >= mean_degree)."""
        assert mean_field_threshold(self.A, self.p, self.r) >= \
               spectral_threshold(self.A, self.p, self.r) - 1e-9

    def test_decoupling_gap_nonpositive(self):
        """Delta_alpha = alpha*_spectral - alpha*_mf <= 0 (spectral is smaller)."""
        assert decoupling_gap(self.A, self.p, self.r) <= 1e-9

    def test_lyapunov_c_positive_in_regime(self):
        alpha = spectral_threshold(self.A, self.p, self.r) * 0.5
        c = lyapunov_constant(self.A, alpha, self.p, self.r)
        if lyapunov_regime_satisfied(self.A, alpha, self.p, self.r):
            assert c > 0

    def test_persistence_lb_zero_below_threshold(self):
        alpha = spectral_threshold(self.A, self.p, self.r) * 0.5
        assert persistence_lower_bound(self.A, alpha, self.p, self.r) == 0.0

    def test_theory_summary_keys(self):
        s = theory_summary(self.A, alpha=0.1, p=self.p, r=self.r)
        expected = ["n", "mean_degree", "degree_var", "lambda_max",
                    "alpha_spectral", "alpha_mf", "decoupling_gap",
                    "lyapunov_c", "persistence_lb"]
        for k in expected:
            assert k in s, f"Missing key: {k}"
