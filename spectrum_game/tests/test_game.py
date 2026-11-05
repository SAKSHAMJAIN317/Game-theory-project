"""Unit tests for Best Response dynamics, Nash Equilibrium verification, and invariance."""

import numpy as np
import pytest

from src.config import SystemConfig
from src.channel import generate_topology
from src.game import run_sequential_best_response_g1, verify_ne_g1


def test_k1_no_moves():
    """Verify that when K=1 (single channel available), no player can switch."""
    cfg = SystemConfig(N=15, K=1)
    topo = generate_topology(config=cfg, seed=42)

    res = run_sequential_best_response_g1(topo, tol=1.0e-9)

    assert res.converged is True
    assert res.switches == 0
    assert res.rounds == 1
    assert res.is_nash is True
    # All players assigned to channel 0
    assert np.all(res.allocation == 0)


def test_verify_ne_passes_on_100_percent_of_g1_runs():
    """Verify that the independent Nash Equilibrium test passes on 100% of G1 runs."""
    cfg = SystemConfig(N=20, K=4)
    for seed in range(30):
        topo = generate_topology(config=cfg, seed=seed)
        res = run_sequential_best_response_g1(topo, tol=1.0e-9)

        assert res.converged is True, f"Failed to converge on seed {seed}"
        assert res.is_nash is True, f"Failed independent NE verification on seed {seed}"

        # Double check with direct call to verify_ne_g1
        is_ne, violation = verify_ne_g1(res.allocation, topo, tol=1.0e-9)
        assert is_ne is True, f"NE violation details: {violation}"


def test_channel_relabel_invariance():
    """
    Verify channel-relabel invariance: permuting channel indices
    leaves exact potential and network sum-rate strictly invariant.
    """
    cfg = SystemConfig(N=20, K=4)
    topo = generate_topology(config=cfg, seed=42)
    res = run_sequential_best_response_g1(topo, tol=1.0e-9)

    original_a = res.allocation
    original_phi = topo.compute_potential(original_a)
    original_sum_rate = float(np.sum(topo.compute_rates_mbps(original_a)))

    # Try several permutations of {0, 1, 2, 3}
    rng = np.random.default_rng(777)
    for _ in range(10):
        perm = rng.permutation(cfg.K)
        # Apply permutation mapping: permuted_a[i] = perm[original_a[i]]
        permuted_a = np.array([perm[c] for c in original_a])

        perm_phi = topo.compute_potential(permuted_a)
        perm_sum_rate = float(np.sum(topo.compute_rates_mbps(permuted_a)))

        assert np.isclose(perm_phi, original_phi, atol=1.0e-12), (
            f"Potential changed under permutation! {original_phi} vs {perm_phi}"
        )
        assert np.isclose(perm_sum_rate, original_sum_rate, atol=1.0e-10), (
            f"Sum rate changed under permutation! {original_sum_rate} vs {perm_sum_rate}"
        )
