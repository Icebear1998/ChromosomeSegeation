# GEMINI.md - Chromosome Segregation Modeling Project

This file provides a comprehensive overview of the Chromosome Segregation Modeling project for instructional context.

## Project Overview

This is a scientific computing and data analysis project written in Python. Its primary goal is to model the timing of chromosome segregation during cell division. The project fits parameters of different mathematical models to experimental data from various mutant yeast strains to understand which underlying biological mechanism best explains the observed segregation timing.

There are two main complementary approaches implemented:

1.  **Method of Moments (MoM):** A fast, analytical approach using a normal distribution approximation. It's suitable for quick exploration and initial parameter fitting. The code for this pipeline is primarily located in the `SecondVersion/` directory.
2.  **Stochastic Simulation:** A more accurate but computationally expensive approach using the Gillespie algorithm for exact stochastic simulations. The likelihood is calculated using Kernel Density Estimation (KDE) from the simulation results. This is used for final parameter fitting and generating publication-quality results. The code is primarily in the root directory.

The project compares 8 different biological mechanisms, from a simple baseline model to complex combined models involving time-varying degradation rates, degradation bursts, and feedback loops. Model comparison is performed using AIC and BIC on the negative log-likelihoods derived from the fits.

**Technologies:** Python, NumPy, SciPy, Pandas, Matplotlib, scikit-learn (for KDE).

## Directory Structure & Key Files

-   `Data/All_strains_SCStimes.xlsx`: The primary source of experimental data, containing timing differences for chromosome separation across 5 mutant strains.
-   `Docs/`: Contains detailed project documentation, including mathematical formulations (`Project_Documentation.md`) and code references (`CODE_REFERENCE.md`).
-   `SecondVersion/`: Contains the code for the **Method of Moments (MoM) pipeline**.
    -   `MoMCalculations.py`: Core logic for calculating mean and variance using MoM.
    -   `MoMOptimization_join.py`: The main script for running joint parameter optimization using the MoM approach.
    -   `TestDataPlot.py`: Script to visualize the results from the MoM optimization.
-   **Root Directory Scripts**: Contains the code for the **Stochastic Simulation pipeline**.
    -   `MultiMechanismSimulationTimevary.py`: The core simulation engine implementing the Gillespie algorithm for all mechanisms, including time-varying ones.
    -   `simulation_utils.py`: Utility functions for the simulation pipeline (data loading, likelihood calculation via KDE, parameter bounds).
    -   `SimulationOptimization_join.py`: The main script for running joint parameter optimization using simulations.
    -   `TestDataPlotSimulation.py`: Script to visualize the results from the simulation-based optimization.
-   `model_comparison_aic_bic.py`: A crucial script that runs optimizations for all 8 mechanisms, calculates AIC/BIC, and provides a summary for model selection. This is often the final step of the analysis.
-   `requirements.txt`: A list of all Python dependencies for the project.
-   `submit_*.sh` / `ARC.sh`: Shell scripts for submitting jobs to an HPC cluster using the SLURM scheduler.

## Building and Running

### 1. Installation

First, install the required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Running Optimizations

The project does not have a "build" step. Analyses are run by executing Python scripts directly.

**A. Quick Exploration (Method of Moments - Fast)**

This is useful for initial fits and checking if the models behave as expected.

```bash
# Navigate to the MoM pipeline directory
cd SecondVersion

# Run the joint optimization for a specific mechanism (defined inside the script)
python MoMOptimization_join.py

# Visualize the results
python TestDataPlot.py
```

**B. Full Optimization (Stochastic Simulation - Accurate)**

This is the main workflow for generating final results. It is computationally intensive.

```bash
# Run the joint optimization from the root directory.
# Mechanism and other settings are configured inside the script.
python SimulationOptimization_join.py

# Visualize the results
python TestDataPlotSimulation.py
```

### 3. Model Comparison

This is the primary analysis to compare all 8 mechanisms. This script runs `SimulationOptimization_join.py` for all mechanisms and is very time-consuming.

**Local Run (for testing or small runs):**

```bash
python model_comparison_aic_bic.py
```

**HPC Run (for full analysis):**

The analysis is designed to be run on a High-Performance Computing (HPC) cluster.

```bash
sbatch submit_model_comparison.sh
```

## Development Conventions

-   **Two Pipelines:** The code is strictly separated between the MoM (`SecondVersion/`) and Simulation (`/`) pipelines.
-   **Parameterization:**
    -   Wild-type parameters are defined relative to chromosome 2 using ratios (`r21`, `r23`, `R21`, `R23`).
    -   For time-varying mechanisms, the preferred parameterization is `(k_max, tau)`, where `tau` is the time to reach the maximum degradation rate.
-   **Optimization Strategy:** A two-step optimization is used:
    1.  **Global Search:** Differential Evolution (`scipy.optimize.differential_evolution`).
    2.  **Local Refinement:** L-BFGS-B to fine-tune the best results from the global search.
-   **Likelihood Calculation:**
    -   MoM: Assumes a normal distribution.
    -   Simulation: Uses Kernel Density Estimation (KDE) with `sklearn.neighbors.KernelDensity` on the distribution of simulated timings to calculate the likelihood of the experimental data.
-   **Configuration:** Most scripts are configured by changing variables directly within the file (e.g., `mechanism`, `num_simulations`, `selected_strains`). There are no external configuration files.
-   **Output:** Optimized parameters and results are saved to `.txt` files (e.g., `simulation_optimized_parameters_{mechanism}.txt`). Plots are saved as image files.
