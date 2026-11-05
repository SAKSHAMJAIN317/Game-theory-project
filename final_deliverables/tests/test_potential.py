"""Unit tests verifying Exact Potential Game properties: potential identity and monotonicity."""

import numpy as np
import pytest

from src.config import SystemConfig
from src.channel import generate_topology
from src.game import run_sequential_best_response_g1


def test_potential_identity():
    """
    Verify the fundamental Exact Potential Game property:
    Delta u_i == Delta Phi for any player i switching from channel a_i to a'_i.
    Tested across 20 independent random topologies and multiple random deviations.
    """
    tol = 1.0e-9
    rng = np.random.default_rng(2025)

    for seed in range(20):
        N = rng.integers(5, 25)
        K = rng.integers(2, 6)
        cfg = SystemConfig(N=N, K=K)
        topo = generate_topology(config=cfg, seed=seed)

        # Generate random profile a
        a = rng.integers(0, K, size=N)

        # Test multiple random deviations
        for _ in range(15):
            i = int(rng.integers(0, N))
            curr_ch = a[i]
            # Pick a different channel
            possible_channels = [k for k in range(K) if k != curr_ch]
            new_ch = int(rng.choice(possible_channels))

            # Current utility and potential
            u_before = topo.compute_utility_g1(i, curr_ch, a)
            phi_before = topo.compute_potential(a)

            # Deviate
            a_prime = a.copy()
            a_prime[i] = new_ch

            u_after = topo.compute_utility_g1(i, new_ch, a_prime)
            phi_after = topo.compute_potential(a_prime)

            delta_u = u_after - u_before
            delta_phi = phi_after - phi_before

            # Verify Exact Potential property: delta_u == delta_phi
            diff = abs(delta_u - delta_phi)
            assert diff < tol, f"Potential identity failed! diff={diff:.3e}, seed={seed}, player={i}"


def test_phi_monotone_non_decreasing():
    """
    Verify that sequential best response dynamics never decrease the potential Phi(a)
    at any switch step. Tested on multiple topologies.
    """
    for seed in [10, 42, 99, 123]:
        cfg = SystemConfig(N=20, K=4)
        topo = generate_topology(config=cfg, seed=seed)
        res = run_sequential_best_response_g1(topo, tol=1.0e-9)

        phis = [entry.phi for entry in res.history]
        assert len(phis) >= 1

        for step_idx in range(len(phis) - 1):
            phi_curr = phis[step_idx]
            phi_next = phis[step_idx + 1]
            # Each switch step must strictly improve Phi by at least tol
            assert phi_next >= phi_curr - 1.0e-12, (
                f"Potential decreased at step {step_idx}: {phi_curr} -> {phi_next}"
            )
            # Since a switch occurs only when improvement > tol, delta should be > 0
            assert phi_next - phi_curr > 0.0, (
                f"Non-positive potential improvement at step {step_idx}"
            )


def test_brute_force_local_optimum_on_small_instances():
    """
    Verify that the converged state of G1 Best Response is a local maximum of Phi:
    no single user unilateral deviation can strictly increase Phi.
    """
    for seed in [1, 7, 42]:
        cfg = SystemConfig(N=8, K=3)
        topo = generate_topology(config=cfg, seed=seed)
        res = run_sequential_best_response_g1(topo, tol=1.0e-9)

        a_ne = res.allocation
        phi_ne = topo.compute_potential(a_ne)

        # Check every unilateral deviation for all players i and all channels k
        for i in range(topo.N):
            curr_ch = a_ne[i]
            for k in range(topo.K):
                if k == curr_ch:
                    continue
                a_dev = a_ne.copy()
                a_dev[i] = k
                phi_dev = topo.compute_potential(a_dev)
                # No unilateral deviation can improve Phi by more than tol
                assert phi_dev - phi_ne <= 1.0e-9, (
                    f"Profitable unilateral deviation found for player {i} to ch {k}! "
                    f"Gain: {phi_dev - phi_ne:.3e}"
                )
