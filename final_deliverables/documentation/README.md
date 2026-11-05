# Game-Theoretic Spectrum Sharing Simulator: Deliverables Guide

## Overview
This package contains the collected, verified deliverables from the 5G/6G Game-Theoretic Spectrum Sharing Simulation project.

## Mathematical Formulation
- **Game G1 (Exact Potential Game)**:
  - Links: $N$ distributed Tx-Rx pairs, Channels: $K$ orthogonal subbands ($B = 20\text{ MHz}$).
  - Symmetrized coupling: $W[i,j] = \frac{P (g[i,j] + g[j,i])}{\sigma^2}$ with $W[i,i] = 0$.
  - Individual Utility: $u_i(a) = -\sum_{j \neq i, a_j = a_i} W[i,j]$.
  - Exact Potential: $\Phi(a) = -\sum_{i < j, a_i = a_j} W[i,j]$.
  - Exact Potential Property: $\Delta u_i(a_i \to a'_i, a_{-i}) \equiv \Delta \Phi(a_i \to a'_i, a_{-i})$.
  - Guaranteed finite convergence to pure-strategy Nash Equilibrium under sequential best-response dynamics.

## Quick Execution Commands

From the workspace root:

```bash
# 1. Activate Virtual Environment
source .venv/bin/activate

# 2. Run All Unit Tests (15 Passed)
pytest

# 3. Run Single-Topology Demo (Experiment E1, Seed 42)
python scripts/run_demo.py --config config.yaml --seed 42

# 4. Run Full Experiment Suite (Experiments E1 to E6, 100 Topologies each)
python scripts/run_experiments.py --config config.yaml
```

## Deliverables Organization
- `results/`: Raw CSV files for Experiments E1 through E6 (seeds 0–99).
- `figures/`: 11 publication-quality PNG figures (200 DPI).
- `tables/`: Summary tables in both CSV and Markdown formats with paired Wilcoxon signed-rank test results.
- `tests/`: Test files and verified test execution output log.
- `documentation/`: System configuration (`config.yaml`), metadata, and instructions.
