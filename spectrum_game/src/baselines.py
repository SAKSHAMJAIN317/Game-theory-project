"""Baseline spectrum allocation strategies: Random, Round-Robin, One-Pass Greedy, and Exhaustive Search."""

import itertools
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from .channel import NetworkTopology


@dataclass
class BaselineResult:
    name: str
    allocation: np.ndarray
    sum_rate_mbps: float
    rates_mbps: np.ndarray
    phi: float
    co_channel_pairs: int


def run_round_robin(topology: NetworkTopology) -> BaselineResult:
    """
    Round-Robin Channel Allocation:
    a_i = i mod K for i = 0, ..., N-1.
    """
    N = topology.N
    K = topology.K
    a = np.array([i % K for i in range(N)], dtype=int)
    rates = topology.compute_rates_mbps(a)
    phi = topology.compute_potential(a)

    # Co-channel pair count: sum_{i < j} [a_i == a_j]
    same = (a[:, np.newaxis] == a[np.newaxis, :])
    pairs = int(np.sum(same[np.triu_indices(N, k=1)]))

    return BaselineResult(
        name="Round-Robin",
        allocation=a,
        sum_rate_mbps=float(np.sum(rates)),
        rates_mbps=rates,
        phi=phi,
        co_channel_pairs=pairs,
    )


def run_one_pass_greedy(topology: NetworkTopology) -> BaselineResult:
    """
    One-Pass Greedy Allocation:
    Links 0 to N-1 sequentially choose the channel that minimizes interference
    with already-assigned links: argmin_k sum_{j < i, a_j == k} W[i, j].
    """
    N = topology.N
    K = topology.K
    a = np.zeros(N, dtype=int)

    for i in range(N):
        if i == 0:
            a[i] = 0
            continue
        # Evaluate interference cost on each channel from already assigned users j < i
        channel_costs = np.zeros(K)
        for k in range(K):
            # Players j < i that picked channel k
            assigned_k = np.where((a[:i] == k))[0]
            channel_costs[k] = np.sum(topology.W[i, assigned_k])
        # Pick channel with minimal cost; ties pick lowest index
        a[i] = int(np.argmin(channel_costs))

    rates = topology.compute_rates_mbps(a)
    phi = topology.compute_potential(a)
    same = (a[:, np.newaxis] == a[np.newaxis, :])
    pairs = int(np.sum(same[np.triu_indices(N, k=1)]))

    return BaselineResult(
        name="One-Pass Greedy",
        allocation=a,
        sum_rate_mbps=float(np.sum(rates)),
        rates_mbps=rates,
        phi=phi,
        co_channel_pairs=pairs,
    )


def run_random_allocation(
    topology: NetworkTopology,
    num_draws: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[BaselineResult, List[BaselineResult]]:
    """
    Random Allocation Baseline:
    Draws num_draws independent random allocations, computes metrics for each,
    and returns (representative/mean_result, all_results).
    """
    N = topology.N
    K = topology.K
    if rng is None:
        rng = np.random.default_rng(topology.seed + 1000)

    all_results = []
    sum_rates = []
    phis = []
    pairs_list = []
    all_rates = []

    for _ in range(num_draws):
        a = rng.integers(0, K, size=N)
        rates = topology.compute_rates_mbps(a)
        phi = topology.compute_potential(a)
        same = (a[:, np.newaxis] == a[np.newaxis, :])
        pairs = int(np.sum(same[np.triu_indices(N, k=1)]))

        res = BaselineResult(
            name="Random",
            allocation=a,
            sum_rate_mbps=float(np.sum(rates)),
            rates_mbps=rates,
            phi=phi,
            co_channel_pairs=pairs,
        )
        all_results.append(res)
        sum_rates.append(res.sum_rate_mbps)
        phis.append(phi)
        pairs_list.append(pairs)
        all_rates.append(rates)

    # Average metrics over the 50 draws
    mean_sum_rate = float(np.mean(sum_rates))
    mean_phi = float(np.mean(phis))
    mean_pairs = int(round(np.mean(pairs_list)))
    mean_rates = np.mean(all_rates, axis=0)

    mean_result = BaselineResult(
        name="Random (Averaged 50 draws)",
        allocation=all_results[0].allocation,
        sum_rate_mbps=mean_sum_rate,
        rates_mbps=mean_rates,
        phi=mean_phi,
        co_channel_pairs=mean_pairs,
    )
    return mean_result, all_results


def run_exhaustive_optimum(
    topology: NetworkTopology,
) -> Tuple[BaselineResult, BaselineResult]:
    """
    Exhaustive Optimum for small instances (e.g. N=8, K=3).
    Finds:
    1. Allocation maximizing exact potential Phi(a) (best Phi / minimal interference cost).
    2. Allocation maximizing total sum-rate sum_i R_i(a) (best sum-rate).
    
    Returns:
        (best_phi_result, best_sum_rate_result)
    """
    N = topology.N
    K = topology.K
    assert N <= 10, f"Exhaustive search only supported for N <= 10 (got N={N})"

    # Generate all K^N combinations
    all_allocations = list(itertools.product(range(K), repeat=N))

    best_phi = -float("inf")
    best_phi_alloc = None

    best_sum_rate = -float("inf")
    best_sum_rate_alloc = None

    # Evaluate each allocation
    for alloc_tuple in all_allocations:
        a = np.array(alloc_tuple, dtype=int)
        phi = topology.compute_potential(a)
        if phi > best_phi:
            best_phi = phi
            best_phi_alloc = a.copy()

        rates = topology.compute_rates_mbps(a)
        sum_rate = float(np.sum(rates))
        if sum_rate > best_sum_rate:
            best_sum_rate = sum_rate
            best_sum_rate_alloc = a.copy()

    # Create baseline results
    rates_phi = topology.compute_rates_mbps(best_phi_alloc)
    same_phi = (best_phi_alloc[:, np.newaxis] == best_phi_alloc[np.newaxis, :])
    pairs_phi = int(np.sum(same_phi[np.triu_indices(N, k=1)]))

    res_phi = BaselineResult(
        name="Exhaustive Optimum (Best Phi)",
        allocation=best_phi_alloc,
        sum_rate_mbps=float(np.sum(rates_phi)),
        rates_mbps=rates_phi,
        phi=best_phi,
        co_channel_pairs=pairs_phi,
    )

    rates_sr = topology.compute_rates_mbps(best_sum_rate_alloc)
    same_sr = (best_sum_rate_alloc[:, np.newaxis] == best_sum_rate_alloc[np.newaxis, :])
    pairs_sr = int(np.sum(same_sr[np.triu_indices(N, k=1)]))

    res_sr = BaselineResult(
        name="Exhaustive Optimum (Best Sum-Rate)",
        allocation=best_sum_rate_alloc,
        sum_rate_mbps=best_sum_rate,
        rates_mbps=rates_sr,
        phi=topology.compute_potential(best_sum_rate_alloc),
        co_channel_pairs=pairs_sr,
    )

    return res_phi, res_sr
