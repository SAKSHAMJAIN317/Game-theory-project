"""Unit tests for baseline algorithms and theoretical expectation validations."""

import math
import numpy as np
import pytest

from src.config import SystemConfig
from src.channel import generate_topology
from src.baselines import (
    run_random_allocation,
    run_round_robin,
    run_one_pass_greedy,
    run_exhaustive_optimum,
)


def test_random_baseline_pair_count_matches_theory():
    """
    Verify that random baseline co-channel pair count matches analytical expectation:
    E[Pairs] = N * (N - 1) / (2 * K).
    For N=20, K=4: E[Pairs] = 190 / 4 = 47.5.
    """
    N = 20
    K = 4
    expected_mean = N * (N - 1) / (2.0 * K)  # 47.5
    cfg = SystemConfig(N=N, K=K)
    topo = generate_topology(config=cfg, seed=42)

    # Run 500 random draws
    _, draws = run_random_allocation(topo, num_draws=500, rng=np.random.default_rng(12345))
    empirical_pairs = [d.co_channel_pairs for d in draws]
    mean_pairs = float(np.mean(empirical_pairs))

    # Standard error of the mean
    sem = float(np.std(empirical_pairs, ddof=1) / np.sqrt(len(empirical_pairs)))
    # Mean should be within 3 standard errors of analytical expectation
    assert abs(mean_pairs - expected_mean) <= 3.0 * sem, (
        f"Empirical mean {mean_pairs:.2f} differs from analytical expectation {expected_mean} (sem={sem:.2f})"
    )


def test_round_robin_properties():
    """Verify Round-Robin deterministic distribution and co-channel pair count."""
    N = 20
    K = 4
    cfg = SystemConfig(N=N, K=K)
    topo = generate_topology(config=cfg, seed=42)
    res = run_round_robin(topo)

    # Exactly 5 users per channel: 4 * binom(5, 2) = 4 * 10 = 40 pairs
    assert res.co_channel_pairs == 40
    counts = np.bincount(res.allocation, minlength=K)
    assert np.all(counts == 5)


def test_exhaustive_search_finds_global_optimum():
    """Verify that exhaustive search correctly finds the global maximum of Phi."""
    cfg = SystemConfig(N=6, K=2)
    topo = generate_topology(config=cfg, seed=42)

    res_phi, res_sr = run_exhaustive_optimum(topo)

    # Check that no other allocation has higher Phi
    import itertools
    for alloc_tuple in itertools.product(range(2), repeat=6):
        a = np.array(alloc_tuple)
        p = topo.compute_potential(a)
        assert p <= res_phi.phi + 1.0e-12
