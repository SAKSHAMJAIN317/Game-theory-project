#!/usr/bin/env python3
"""
CLI Demo: Run Game-Theoretic Spectrum Sharing Simulation on a single topology.
Usage: python scripts/run_demo.py [--config config.yaml] [--seed 42]
"""

import os
import sys
import argparse

# Ensure src is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import load_config
from src.simulator import run_experiment_1_demo, get_environment_metadata, save_metadata


def main():
    parser = argparse.ArgumentParser(description="Run Game-Theoretic Spectrum Sharing Demo")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for topology")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config.experiments.demo_seed = args.seed

    print("=" * 78)
    print(" 5G/6G GAME-THEORETIC SPECTRUM SHARING SIMULATOR: DEMO (EXPERIMENT E1)")
    print("=" * 78)
    print(f"Network Parameters: N={config.system.N} links, K={config.system.K} channels, L={config.system.L}m")
    print(f"Radio Parameters:   Bandwidth={config.system.B/1e6:.1f} MHz, Carrier={config.system.f_c/1e9:.2f} GHz, P={config.system.P_dBm:.1f} dBm")
    print(f"Propagation:        Pathloss exponent n={config.system.n_pathloss}, Shadowing std={config.system.sigma_shadowing} dB")
    print(f"Topology Seed:      {config.experiments.demo_seed}")
    print("-" * 78)

    # Save environment metadata
    meta = get_environment_metadata()
    save_metadata(meta, os.path.join(config.paths.logs, "demo_metadata.json"))

    print("\nRunning sequential best-response dynamics and baseline allocations...")
    res = run_experiment_1_demo(config=config, save_outputs=True)

    g1 = res["g1_result"]
    metrics_df = res["metrics_df"]
    history_df = res["history_df"]

    print("\n" + "=" * 78)
    print(" BEST-RESPONSE DYNAMICS SUMMARY (GAME G1 - EXACT POTENTIAL)")
    print("=" * 78)
    print(f"Convergence Status: {'CONVERGED (Pure NE)' if g1.converged else 'MAX ROUNDS REACHED'}")
    print(f"Independent NE Test: {'PASSED (Zero profitable deviations)' if g1.is_nash else 'FAILED'}")
    print(f"Total Rounds:       {g1.rounds}")
    print(f"Total Switches:     {g1.switches}")
    print(f"Execution Time:     {g1.runtime_sec * 1000.0:.2f} ms")
    print(f"Initial Potential:  {g1.initial_phi:.4e}")
    print(f"Final Potential:    {g1.final_phi:.4e}  (Change ΔΦ = +{g1.final_phi - g1.initial_phi:.4e})")
    print(f"Initial Sum-Rate:   {g1.initial_sum_rate_mbps:.2f} Mbps")
    print(f"Final Sum-Rate:     {g1.final_sum_rate_mbps:.2f} Mbps  (Gain = +{g1.final_sum_rate_mbps - g1.initial_sum_rate_mbps:.2f} Mbps)")

    print("\n" + "=" * 78)
    print(" METHOD COMPARISON (Paired on Topology Seed 42)")
    print("=" * 78)
    cols = ["method", "sum_rate_mbps", "mean_rate_mbps", "p5_rate_mbps", "jain_index", "outage_fraction", "co_channel_pairs"]
    disp_df = metrics_df[cols].copy()
    disp_df.columns = ["Method", "Sum-Rate (Mbps)", "Mean (Mbps)", "5th-Pct (Mbps)", "Jain Index", "Outage Frac", "Co-Ch Pairs"]
    print(disp_df.to_string(index=False))

    print("\n" + "=" * 78)
    print(" GENERATED ARTIFACTS")
    print("=" * 78)
    print("Raw CSVs:")
    print(f"  - {os.path.join(config.paths.raw_csv, 'e1_demo_metrics.csv')}")
    print(f"  - {os.path.join(config.paths.raw_csv, 'e1_demo_history.csv')}")
    print("Figures (200 DPI PNG):")
    print(f"  - {os.path.join(config.paths.figures, 'fig1_topology_before_after.png')}")
    print(f"  - {os.path.join(config.paths.figures, 'fig2_phi_vs_iteration.png')}")
    print(f"  - {os.path.join(config.paths.figures, 'fig3_sum_rate_vs_iteration.png')}")
    print(f"  - {os.path.join(config.paths.figures, 'fig5_rate_cdf.png')}")
    print("=" * 78)


if __name__ == "__main__":
    main()
