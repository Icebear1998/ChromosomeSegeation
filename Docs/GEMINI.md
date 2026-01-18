# GEMINI.md - Chromosome Segregation Modeling Project

This file provides a comprehensive overview of the Chromosome Segregation Modeling project for instructional context.

## Project Overview

This is a scientific computing and data analysis project written in Python. Its primary goal is to model the timing of chromosome segregation during cell division. The project fits parameters of different mathematical models to experimental data from various mutant yeast strains to understand which underlying biological mechanism best explains the observed segregation timing.

There are three main complementary approaches implemented:

1.  **Method of Moments (MoM):** A fast, analytical approach using a normal distribution approximation. Suitable for quick exploration. Code in `SecondVersion/`.
2.  **Stochastic Simulation (Gillespie):** The traditional exact stochastic simulation algorithm. Accurate but computationally expensive.
3.  **Fast Simulation (New):** optimized simulation methods that achieve O(1) complexity relative to molecule count, providing profound speedups (100x-1000x) over Gillespie while maintaining statistical exactness for specific model types.
    *   **Beta Sampling Method:** For constant and time-varying rate models without feedback.
    *   **Sum of Waiting Times Method:** For feedback-based models (assuming independent chromosomes).

The project compares 16 different mechanism variations, including simple, fixed burst, time-varying rates, and feedback loops.

## Directory Structure & Key Files

-   `Data/All_strains_SCStimes.xlsx`: Experimental data source.
-   `Docs/`: Detailed documentation (`Project_Documentation.md`, `CODE_REFERENCE.md`).
-   `SecondVersion/`: **Method of Moments (MoM) pipeline**.
    -   `MoMCalculations.py`: Core MoM logic.
    -   `MoMOptimization_join.py`: Main MoM optimization script.
    -   `TestDataPlot.py`: MoM visualization.
-   **Root Directory Scripts**: **Stochastic Simulation pipeline**.
    -   **Core Simulation Engines:**
        -   `MultiMechanismSimulationTimevary.py`: Traditional Gillespie logic.
        -   `FastBetaSimulation.py`: **[NEW]** Vectorized Beta sampling for O(1) simulation of linear decay.
        -   `FastFeedbackSimulation.py`: **[NEW]** Vectorized Sum of Waiting Times for O(1) simulation of feedback.
    -   **Analysis & Optimization:**
        -   `simulation_utils.py`: Central hub for data loading and **automatic dispatch** to the fastest available simulation method.
        -   `SimulationOptimization_join.py`: Main differential evolution optimization script.
        -   `AnalyzeParameterStabilitySimulation.py`: **[NEW]** Script to analyze parameter stability across multiple optimization runs.
        -   `model_comparison_aic_bic.py`: Compare all mechanics using AIC/BIC. Now powered by Fast Simulation.
    -   **Validation:**
        -   `TestBetaVsGillespie.py`: Validates Beta method against Gillespie.
        -   `TestFeedbackVsGillespie.py`: Validates Feedback method against Gillespie.
        -   `TestDataPlotSimulation.py`: Plotting simulation results against experimental data.
-   `submit_*.sh`: SLURM submission scripts (e.g., `submit_model_comparison.sh`).

## Building and Running

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Running Optimizations

**A. Quick Exploration (MoM)**

```bash
cd SecondVersion
python MoMOptimization_join.py
```

**B. Full Optimization (Fast Simulation)**

Optimization now defaults to Fast methods where available.

```bash
# Run joint optimization (configured in script)
python SimulationOptimization_join.py

# Visualize results
python TestDataPlotSimulation.py
```

**C. Parameter Stability Analysis**

```bash
# Analyze stability of fitted parameters for a specific mechanism
python AnalyzeParameterStabilitySimulation.py
```

### 3. Model Comparison

Comparisons are now feasible on standard timeframe due to Fast Simulation.

```bash
# Runs optimization for all mechanisms
sbatch submit_model_comparison.sh
# OR locally:
python model_comparison_aic_bic.py
```

## Development Conventions

-   **Pipeline Separation:** MoM (`SecondVersion/`) logic is kept separate from Simulation logic, though `model_comparison_aic_bic.py` bridges them.
-   **Simulation Dispatch:** `simulation_utils.py` contains the logic to choose `FastBetaSimulation`, `FastFeedbackSimulation`, or fallback to `MultiMechanismSimulationTimevary` (Gillespie) based on the mechanism name.
-   **Likelihood:** Uses Kernel Density Estimation (KDE) on simulation outputs.
-   **Mechanism Testing:** When adding new mechanisms, ensure they are added to `simulation_utils.py` dispatch and tested via `Test*VsGillespie.py` scripts.
