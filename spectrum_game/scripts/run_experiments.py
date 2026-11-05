#!/usr/bin/env python3
"""
CLI Experiments Suite: Run full experiment suite E1-E6 and generate publication figures and tables.
Usage: python scripts/run_experiments.py [--config config.yaml] [--experiments E1 E2 E3 E4 E5 E6]
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
from src.simulator import (
    run_experiment_1_demo,
    run_experiment_2_density,
    run_experiment_3_channels,
    run_experiment_4_optimality,
    run_experiment_5_sensitivity,
    run_experiment_6_g2_comparison,
    generate_all_tables_and_figures,
    get_environment_metadata,
    save_metadata,
)


def main():
    parser = argparse.ArgumentParser(description="Run Game-Theoretic Spectrum Sharing Experiments Suite")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["E1", "E2", "E3", "E4", "E5", "E6"],
        help="List of experiments to run (e.g. --experiments E1 E2 E3 E4 E5 E6)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    selected_exps = [e.upper() for e in args.experiments]

    print("=" * 80)
    print(" 5G/6G GAME-THEORETIC SPECTRUM SHARING: EXPERIMENT SUITE")
    print("=" * 80)
    print(f"Loaded configuration from: {args.config}")
    print(f"Topologies per experiment: {config.experiments.num_topologies} (seeds 0-99)")
    print(f"Selected experiments:      {', '.join(selected_exps)}")
    print("-" * 80)

    # Save run metadata
    meta = get_environment_metadata()
    meta["selected_experiments"] = selected_exps
    meta["config_file"] = args.config
    save_metadata(meta, os.path.join(config.paths.logs, "experiments_metadata.json"))

    # Execute selected experiments
    if "E1" in selected_exps:
        print("\n[1/6] Running Experiment E1: Single-Topology Demo (Seed 42)...")
        run_experiment_1_demo(config=config, save_outputs=True)

    if "E2" in selected_exps:
        print("\n[2/6] Running Experiment E2: Density Scaling (N in {10,20,30,40,50})...")
        run_experiment_2_density(config=config)

    if "E3" in selected_exps:
        print("\n[3/6] Running Experiment E3: Channel Scaling (K in {2,3,4,5,6,8})...")
        run_experiment_3_channels(config=config)

    if "E4" in selected_exps:
        print("\n[4/6] Running Experiment E4: Optimality Gap vs Exhaustive Search (N=8, K=3)...")
        run_experiment_4_optimality(config=config)

    if "E5" in selected_exps:
        print("\n[5/6] Running Experiment E5: Propagation Sensitivity Analysis (n, sigma_sh)...")
        run_experiment_5_sensitivity(config=config)

    if "E6" in selected_exps:
        print("\n[6/6] Running Experiment E6: Potential Game G1 vs Direct Rate Game G2...")
        run_experiment_6_g2_comparison(config=config)

    # Post-process and generate summary tables and figures
    print("\n" + "=" * 80)
    print(" GENERATING SUMMARY TABLES & PUBLICATION FIGURES (200 DPI)")
    print("=" * 80)
    generate_all_tables_and_figures(config=config)

    print("\n" + "=" * 80)
    print(" EXPERIMENT SUITE COMPLETE")
    print("=" * 80)
    print(f"Raw CSV outputs:  {config.paths.raw_csv}/")
    print(f"Summary Tables:   {config.paths.tables}/ (Markdown & CSV)")
    print(f"Figures (200 DPI):{config.paths.figures}/ (PNGs)")
    print(f"Logs & Metadata:  {config.paths.logs}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
