"""Unit tests for simulation reproducibility, bit-for-bit identical outputs, and metrics."""

import numpy as np
import pytest

from src.config import SystemConfig
from src.channel import generate_topology
from src.game import run_sequential_best_response_g1
from src.metrics import calculate_jain_index, compute_metrics


def test_same_seed_gives_identical_results():
    """
    Verify strict simulation reproducibility:
    Running the simulation twice with identical seed produces bit-for-bit identical metrics,
    allocations, potential, and sum-rate.
    """
    seed = 42
    cfg = SystemConfig(N=20, K=4)

    # Run 1
    topo1 = generate_topology(config=cfg, seed=seed)
    res1 = run_sequential_best_response_g1(topo1, tol=1.0e-9)
    m1 = compute_metrics(topo1, res1.allocation, "Proposed NE (G1)")

    # Run 2
    topo2 = generate_topology(config=cfg, seed=seed)
    res2 = run_sequential_best_response_g1(topo2, tol=1.0e-9)
    m2 = compute_metrics(topo2, res2.allocation, "Proposed NE (G1)")

    # Bit-for-bit equality checks
    assert np.array_equal(topo1.tx_pos, topo2.tx_pos)
    assert np.array_equal(topo1.rx_pos, topo2.rx_pos)
    assert np.array_equal(topo1.G, topo2.G)
    assert np.array_equal(topo1.W, topo2.W)
    assert np.array_equal(res1.allocation, res2.allocation)
    assert res1.rounds == res2.rounds
    assert res1.switches == res2.switches
    assert np.isclose(res1.final_phi, res2.final_phi, atol=0.0)
    assert np.isclose(m1.sum_rate_mbps, m2.sum_rate_mbps, atol=0.0)
    assert np.isclose(m1.jain_index, m2.jain_index, atol=0.0)


def test_jain_index_properties():
    """Verify mathematical properties of Jain's Fairness Index."""
    # 1. Perfectly equal rates -> Jain index = 1.0
    equal_rates = np.array([10.0, 10.0, 10.0, 10.0])
    assert math.isclose(calculate_jain_index(equal_rates), 1.0, rel_tol=1e-9)

    # 2. Complete monopoly (1 link gets rate, rest get 0) -> Jain index = 1/N
    monopoly_rates = np.array([100.0, 0.0, 0.0, 0.0])
    assert math.isclose(calculate_jain_index(monopoly_rates), 0.25, rel_tol=1e-9)

    # 3. Always bounded in [1/N, 1.0]
    rng = np.random.default_rng(42)
    for _ in range(20):
        r = rng.uniform(0.1, 50.0, size=10)
        j = calculate_jain_index(r)
        assert 0.1 <= j <= 1.0


import math
