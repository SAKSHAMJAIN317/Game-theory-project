"""Game-theoretic spectrum sharing models: G1 (Potential Game) and G2 (Rate-Maximization)."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .channel import NetworkTopology


@dataclass
class IterationLogEntry:
    step: int
    round_idx: int
    player: int
    channel_from: int
    channel_to: int
    phi: float
    sum_rate_mbps: float


@dataclass
class GameResult:
    game_type: str
    allocation: np.ndarray
    initial_allocation: np.ndarray
    converged: bool
    rounds: int
    switches: int
    is_nash: bool
    history: List[IterationLogEntry] = field(default_factory=list)
    final_phi: float = 0.0
    initial_phi: float = 0.0
    final_sum_rate_mbps: float = 0.0
    initial_sum_rate_mbps: float = 0.0
    runtime_sec: float = 0.0


def verify_ne_g1(
    allocation: np.ndarray,
    topology: NetworkTopology,
    tol: float = 1.0e-9,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Independently verify whether an allocation is a pure-strategy Nash Equilibrium for G1.
    A state a is a pure NE iff for every player i:
        u_i(a_i, a_{-i}) >= u_i(k, a_{-i}) - tol  for all k in {0, ..., K-1}.
    Equivalently, no player can unilaterally decrease interference cost by > tol.
    
    Returns:
        (is_ne, violation_dict)
    """
    a = np.asarray(allocation).copy()
    N = topology.N
    K = topology.K

    for i in range(N):
        curr_ch = a[i]
        curr_u = topology.compute_utility_g1(i, curr_ch, a)
        for k in range(K):
            if k == curr_ch:
                continue
            alt_u = topology.compute_utility_g1(i, k, a)
            if alt_u - curr_u > tol:
                return False, {
                    "player": i,
                    "current_channel": curr_ch,
                    "deviating_channel": k,
                    "improvement": alt_u - curr_u,
                }
    return True, None


def verify_ne_g2(
    allocation: np.ndarray,
    topology: NetworkTopology,
    tol: float = 1.0e-9,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Independently verify whether an allocation is a pure-strategy Nash Equilibrium for G2.
    A state a is a pure NE iff for every player i:
        R_i(a_i, a_{-i}) >= R_i(k, a_{-i}) - tol  for all k in {0, ..., K-1}.
    """
    a = np.asarray(allocation).copy()
    N = topology.N
    K = topology.K

    for i in range(N):
        curr_ch = a[i]
        curr_u = topology.compute_utility_g2(i, curr_ch, a)
        for k in range(K):
            if k == curr_ch:
                continue
            alt_u = topology.compute_utility_g2(i, k, a)
            if alt_u - curr_u > tol:
                return False, {
                    "player": i,
                    "current_channel": curr_ch,
                    "deviating_channel": k,
                    "improvement": alt_u - curr_u,
                }
    return True, None


def run_sequential_best_response_g1(
    topology: NetworkTopology,
    init_allocation: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    tol: float = 1.0e-9,
    max_rounds: int = 100,
) -> GameResult:
    """
    Sequential Best Response Dynamics for Game G1:
    - Finite exact potential game with potential Phi(a) = - sum_{i < j} W[i, j] [a_i == a_j].
    - Random player order each round.
    - Switch only if cost improvement > tol (i.e. utility improvement > tol). Ties keep current.
    - Log Phi and sum-rate after every switch.
    - Record rounds, switches, converged flag.
    - Post-termination independent verify_ne_g1.
    """
    import time
    start_time = time.perf_counter()

    N = topology.N
    K = topology.K

    if rng is None:
        rng = np.random.default_rng(topology.seed)

    if init_allocation is None:
        a = rng.integers(0, K, size=N)
    else:
        a = np.asarray(init_allocation).copy()

    initial_a = a.copy()
    initial_phi = topology.compute_potential(initial_a)
    initial_sum_rate = float(np.sum(topology.compute_rates_mbps(initial_a)))

    history: List[IterationLogEntry] = [
        IterationLogEntry(
            step=0,
            round_idx=0,
            player=-1,
            channel_from=-1,
            channel_to=-1,
            phi=initial_phi,
            sum_rate_mbps=initial_sum_rate,
        )
    ]

    total_switches = 0
    converged = False
    completed_rounds = 0

    for r in range(1, max_rounds + 1):
        completed_rounds = r
        round_switches = 0
        order = rng.permutation(N)

        for i in order:
            curr_ch = a[i]
            # Calculate utility for all channels k in 0..K-1
            # u_i(k, a_{-i}) = - sum_{j != i, a_j == k} W[i, j]
            # Vectorized computation across channels:
            # Mask of shape (K, N)
            utilities = np.zeros(K)
            for k in range(K):
                utilities[k] = topology.compute_utility_g1(i, k, a)

            curr_utility = utilities[curr_ch]
            best_utility = np.max(utilities)

            # Ties keep current channel: if current channel achieves max utility within tol, keep it!
            if best_utility - curr_utility > tol:
                # Find candidate best channels
                best_channels = np.where(utilities >= best_utility - 1.0e-12)[0]
                # If current channel is among best, keep it
                if curr_ch in best_channels:
                    chosen_ch = curr_ch
                else:
                    chosen_ch = int(best_channels[0])

                if chosen_ch != curr_ch:
                    a[i] = chosen_ch
                    round_switches += 1
                    total_switches += 1

                    cur_phi = topology.compute_potential(a)
                    cur_sum_rate = float(np.sum(topology.compute_rates_mbps(a)))
                    history.append(
                        IterationLogEntry(
                            step=total_switches,
                            round_idx=r,
                            player=int(i),
                            channel_from=int(curr_ch),
                            channel_to=int(chosen_ch),
                            phi=cur_phi,
                            sum_rate_mbps=cur_sum_rate,
                        )
                    )

        if round_switches == 0:
            converged = True
            break

    runtime_sec = time.perf_counter() - start_time
    is_nash, _ = verify_ne_g1(a, topology, tol=tol)

    final_phi = topology.compute_potential(a)
    final_sum_rate = float(np.sum(topology.compute_rates_mbps(a)))

    return GameResult(
        game_type="G1_Potential",
        allocation=a,
        initial_allocation=initial_a,
        converged=converged,
        rounds=completed_rounds,
        switches=total_switches,
        is_nash=is_nash,
        history=history,
        final_phi=final_phi,
        initial_phi=initial_phi,
        final_sum_rate_mbps=final_sum_rate,
        initial_sum_rate_mbps=initial_sum_rate,
        runtime_sec=runtime_sec,
    )


def run_sequential_best_response_g2(
    topology: NetworkTopology,
    init_allocation: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    tol: float = 1.0e-9,
    max_rounds: int = 100,
) -> GameResult:
    """
    Sequential Best Response Dynamics for Game G2 (Rate-Maximizing Game):
    - u_i(a) = R_i(a)
    - Convergence measured empirically.
    """
    import time
    start_time = time.perf_counter()

    N = topology.N
    K = topology.K

    if rng is None:
        rng = np.random.default_rng(topology.seed)

    if init_allocation is None:
        a = rng.integers(0, K, size=N)
    else:
        a = np.asarray(init_allocation).copy()

    initial_a = a.copy()
    initial_phi = topology.compute_potential(initial_a)
    initial_sum_rate = float(np.sum(topology.compute_rates_mbps(initial_a)))

    history: List[IterationLogEntry] = [
        IterationLogEntry(
            step=0,
            round_idx=0,
            player=-1,
            channel_from=-1,
            channel_to=-1,
            phi=initial_phi,
            sum_rate_mbps=initial_sum_rate,
        )
    ]

    total_switches = 0
    converged = False
    completed_rounds = 0

    for r in range(1, max_rounds + 1):
        completed_rounds = r
        round_switches = 0
        order = rng.permutation(N)

        for i in order:
            curr_ch = a[i]
            rates = np.zeros(K)
            for k in range(K):
                rates[k] = topology.compute_utility_g2(i, k, a)

            curr_rate = rates[curr_ch]
            best_rate = np.max(rates)

            if best_rate - curr_rate > tol:
                best_channels = np.where(rates >= best_rate - 1.0e-12)[0]
                if curr_ch in best_channels:
                    chosen_ch = curr_ch
                else:
                    chosen_ch = int(best_channels[0])

                if chosen_ch != curr_ch:
                    a[i] = chosen_ch
                    round_switches += 1
                    total_switches += 1

                    cur_phi = topology.compute_potential(a)
                    cur_sum_rate = float(np.sum(topology.compute_rates_mbps(a)))
                    history.append(
                        IterationLogEntry(
                            step=total_switches,
                            round_idx=r,
                            player=int(i),
                            channel_from=int(curr_ch),
                            channel_to=int(chosen_ch),
                            phi=cur_phi,
                            sum_rate_mbps=cur_sum_rate,
                        )
                    )

        if round_switches == 0:
            converged = True
            break

    runtime_sec = time.perf_counter() - start_time
    is_nash, _ = verify_ne_g2(a, topology, tol=tol)

    final_phi = topology.compute_potential(a)
    final_sum_rate = float(np.sum(topology.compute_rates_mbps(a)))

    return GameResult(
        game_type="G2_Rate",
        allocation=a,
        initial_allocation=initial_a,
        converged=converged,
        rounds=completed_rounds,
        switches=total_switches,
        is_nash=is_nash,
        history=history,
        final_phi=final_phi,
        initial_phi=initial_phi,
        final_sum_rate_mbps=final_sum_rate,
        initial_sum_rate_mbps=initial_sum_rate,
        runtime_sec=runtime_sec,
    )
