# Chromosome Segregation Project - Code Reference

**Last Updated**: December 2024  
**Version**: 2.2

---

## Table of Contents

1. [Project Framework](#project-framework)
2. [Directory Structure](#directory-structure)
3. [Core Mathematical Framework](#core-mathematical-framework)
4. [Main Code Files](#main-code-files)
5. [Workflow Summary](#workflow-summary)
6. [Quick Start Guide](#quick-start-guide)

---

## Project Framework

### Overview

This project implements a comprehensive mathematical and computational framework for modeling chromosome segregation timing during cell division. The core approach involves:

1. **Mathematical Modeling**: Using Method of Moments (MoM) approximations and stochastic simulations
2. **Parameter Optimization**: Fitting model parameters to experimental data using Earth Mover's Distance (EMD) minimization
3. **Cross-Validation**: 5-Fold Cross-Validation to ensure model generalization
4. **Validation**: Parameter recovery studies and diagnostic analyses

### Biological Context

The project models how cohesin proteins are degraded during cell division, allowing sister chromatids to separate. Key aspects:

- **Cohesin degradation**: Proteins holding chromosomes together are removed over time
- **Threshold mechanism**: Separation occurs when cohesin count drops below a threshold
- **Timing differences**: We measure time differences between chromosome separations (T1-T2, T3-T2)
- **Mutant strains**: Different genetic backgrounds affect degradation rates and thresholds

### Two Complementary Approaches

#### 1. Method of Moments (MoM) - Analytical Approach

- **Fast**: ~1ms per evaluation
- **Approximation**: Assumes normal distribution
- **Use case**: Quick parameter exploration, initial fits

#### 2. Stochastic Simulation - Computational Approach

- **Accurate**: Captures full distribution via Gillespie algorithm
- **Flexible**: Uses Earth Mover's Distance (EMD) as the primary robust metric, with fallbacks to Kernel Density Estimation (KDE)
- **Robust**: EMD avoids binning/bandwidth issues and provides a stable gradient for optimization

---

## Directory Structure

```
ChromosomeSegeation/
│
├── Data/                              # Experimental data
│   ├── All_strains_SCStimes.xlsx     # Main experimental timing data (1,423 data points)
│   └── Chromosome_diff.xlsx          # Additional chromosome information
│
├── Docs/                              # Documentation
│   ├── CODE_REFERENCE.md             # This file - comprehensive code reference
│   ├── Project_Documentation.md      # Scientific/mathematical documentation
│   ├── ANALYSIS_WORKFLOW.md          # Analysis workflow guide
│   └── ARC_DEPLOYMENT_GUIDE.md       # HPC cluster deployment guide
│
├── SecondVersion/                     # MoM-based optimization pipeline
│   ├── MoMCalculations.py            # Method of Moments calculations
│   ├── MoMOptimization_join.py       # Joint MoM optimization
│   ├── MoMOptimization_independent.py# Independent MoM optimization
│   ├── MultiMechanismSimulation.py   # Gillespie simulation (non-time-varying)
│   ├── TestDataPlot.py               # Visualization for MoM results
│   └── Chromosomes_Theory.py         # Core theoretical calculations (MoM)
│
├── Root Directory/                    # Simulation-based optimization pipeline
│   ├── SimulationOptimization_join.py         # Joint simulation optimization (EMD/NLL)
│   ├── CrossValidation_EMD.py                # 5-Fold Cross-Validation script
│   ├── SimulationOptimization_independent.py  # Independent simulation optimization
│   ├── MultiMechanismSimulationTimevary.py   # Gillespie simulation (time-varying)
│   ├── simulation_utils.py                    # Shared simulation utilities (EMD, KDE)
│   ├── TestDataPlotSimulation.py             # Visualization for simulation results
│   ├── AnalyzeEmdSampleEfficiency.py         # Analysis of EMD sample efficiency
│   ├── Chromosomes_Theory.py                 # Core theoretical calculations
│   │
│   ├── model_comparison_aic_bic.py           # Model comparison framework
│   ├── demonstrate_kde.py                     # KDE demonstration script
│   │
│   └── submit_*.sh                           # HPC job submission scripts
│
├── Results/                           # Output plots and figures
├── diagnostic_plots/                  # Diagnostic analysis outputs
└── parameter_distribution_plots/      # Parameter distribution visualizations
```

---

## Core Mathematical Framework

### The Fundamental Model

We model the time difference `X = T₁ - T₂` where:

- `Tᵢ` = Time for chromosome i to reach separation threshold
- Cohesin count decreases: `N → N-1 → N-2 → ... → n` (separation occurs)
- Rate of decrease: `k(t)` (can be constant or time-varying)

### Eight Mechanisms Implemented

| #   | Mechanism                       | Parameters | Description                                    |
| --- | ------------------------------- | ---------- | ---------------------------------------------- |
| 1   | `simple`                        | 11         | Baseline: constant degradation rate            |
| 2   | `fixed_burst`                   | 12         | Degradation in fixed-size bursts               |
| 3   | `feedback_onion`                | 12         | Rate modified by "onion-like" feedback         |
| 4   | `fixed_burst_feedback_onion`    | 13         | Burst + onion feedback combined                |
| 5   | `time_varying_k`                | 12         | Rate increases linearly with time: k(t) = k₁·t |
| 6   | `time_varying_k_fixed_burst`    | 13         | Time-varying + burst                           |
| 7   | `time_varying_k_feedback_onion` | 13         | Time-varying + onion feedback                  |
| 8   | `time_varying_k_combined`       | 14         | All three mechanisms combined                  |

### Parameter Types

#### Base Parameters (Wildtype)

- `n₂, N₂`: Threshold and initial count for chromosome 2 (reference)
- `r₂₁, r₂₃`: Ratios for chromosome 1 and 3 thresholds (n₁/n₂, n₃/n₂)
- `R₂₁, R₂₃`: Ratios for chromosome 1 and 3 initial counts (N₁/N₂, N₃/N₂)
- `k_max`: Maximum degradation rate
- `tau`: Time to reach maximum rate (τ = k_max/k₁)

#### Mechanism-Specific Parameters

- `burst_size`: Size of degradation bursts (1-50 cohesins)
- `n_inner`: Inner threshold for onion feedback (1-100 cohesins)

#### Mutant Parameters

- `alpha`: Threshold reduction (threshold mutants) - reduces n values
- `beta_k`: Rate reduction (separase mutants) - reduces k_max
- `beta_tau`: Time scale increase (APC mutants) - increases tau (2-10x)
- `beta_tau2`: Time scale increase (Velcade mutants) - increases tau (2-40x)

### Experimental Data Structure

Five mutant strains with measurements of `T1-T2` and `T3-T2`:

| Strain               | T1-T2 Points | T3-T2 Points | Biological Effect            |
| -------------------- | ------------ | ------------ | ---------------------------- |
| Wildtype             | 126          | 145          | Normal baseline              |
| Threshold            | 67           | 62           | Reduced cohesin threshold    |
| Separase (degRate)   | 161          | 259          | Slower degradation           |
| APC (degRateAPC)     | 158          | 161          | Delayed degradation onset    |
| Velcade (degRateVel) | 131          | 153          | Strong proteasome inhibition |
| **Total**            | **643**      | **780**      | **1,423 data points**        |

---

## Main Code Files

### 1. Core Theory & Calculations

#### `Chromosomes_Theory.py` (Root)

**Purpose**: Implements core theoretical calculations for all mechanisms

**Key Functions**:

```python
def get_harmonic_moments(N, n, k, W=1.0, burst_size=1)
    """Calculate mean and variance for harmonic sum degradation"""

def compute_moments_time_varying(N, n, k_1, k_max, W=1.0, burst_size=1)
    """Calculate moments for time-varying mechanisms"""

def calculate_probability_pdf_mom(n1, n2, n3, N1, N2, N3, k, ...)
    """Calculate PDF for time differences using MoM approximation"""
```

**What it does**:

- Harmonic sum calculations for mean/variance of separation times
- Time-varying rate calculations (k(t) = k₁·t)
- Burst mechanism modifications
- Feedback (onion) modifications
- Normal approximation for PDFs

**Usage**: Called by both MoM and simulation optimization pipelines

---

#### `SecondVersion/Chromosomes_Theory.py`

**Purpose**: Same as root version but specialized for MoM pipeline

**Differences**:

- May have slight variations in implementation details
- Used exclusively by `SecondVersion/` scripts
- Generally synchronized with root version

---

### 2. Method of Moments (MoM) Pipeline

#### `SecondVersion/MoMCalculations.py`

**Purpose**: High-level MoM calculations for all mechanisms

**Key Functions**:

```python
def compute_moments_mom(mechanism, params)
    """Compute mean and variance for any mechanism"""

def compute_pdf_mom(mechanism, params, x_values)
    """Generate PDF using normal approximation"""

def compute_pdf_for_mechanism(mechanism, params, x_grid, pair)
    """Route to correct mechanism calculation"""
```

**What it does**:

- Abstracts mechanism-specific calculations
- Provides unified interface for all 8 mechanisms
- Implements normal approximation: `f(x) ~ N(μ_X, σ²_X)`

**Called by**: `MoMOptimization_*.py` files

---

#### `SecondVersion/MoMOptimization_join.py`

**Purpose**: Joint optimization using MoM across all strains

**Key Features**:

- Optimizes all parameters simultaneously
- Uses differential evolution (global) + L-BFGS-B (local refinement)
- Fast: 5-30 minutes per mechanism
- Refactored with helper functions to reduce code duplication
- Simplified logging and output
- Dictionary-based mechanism configuration for maintainability

**Workflow**:

```
1. Define objective function (negative log-likelihood)
2. Set parameter bounds
3. Run differential evolution (population-based search)
4. Refine with L-BFGS-B
5. Save results to optimized_parameters_*.txt
```

**Key Functions**:

```python
def _get_mech_params(mechanism, param_dict)
    """Extract mechanism-specific parameters"""

def _add_strain_nll(total_nll, mechanism, data_12, data_32, ...)
    """Add NLL for a strain pair (12 and 32)"""
```

**Output Files**: `SecondVersion/optimized_parameters_{mechanism}_join.txt`

---

#### `SecondVersion/MoMOptimization_independent.py`

**Purpose**: Independent optimization strategy (wildtype first, then mutants)

**Key Features**:

- Two-stage optimization
- Stage 1: Fit wildtype parameters only
- Stage 2: Fix wildtype, optimize mutant parameters
- More robust convergence for difficult cases

---

#### `SecondVersion/TestDataPlot.py`

**Purpose**: Visualize MoM optimization results

**Key Features**:

- Loads optimized parameters
- Generates comparison plots (experimental vs MoM PDF)
- Shows both T1-T2 and T3-T2 for all strains
- Creates 2×5 subplot grids

**Usage**:

```python
python SecondVersion/TestDataPlot.py
```

---

### 3. Stochastic Simulation

#### `MultiMechanismSimulationTimevary.py`

**Purpose**: Gillespie algorithm simulation for time-varying mechanisms

**Key Class**:

```python
class MultiMechanismSimulationTimevary:
    def __init__(self, mechanism, initial_state_list, rate_params, n0_list, max_time)
    def simulate(self) -> (times, states, separation_times)
```

**What it does**:

- Implements exact stochastic simulation (Gillespie algorithm)
- Handles all 8 mechanisms
- Time-varying rates: updates `k(t) = min(k₁·t, k_max)` at each step
- Tracks individual chromosome separations
- Returns precise separation times for T1, T2, T3

**Key Mechanisms**:

- **Simple**: Constant rate `k`
- **Burst**: Remove `burst_size` cohesins per event
- **Onion**: Modify rate by `W(N, n_inner)`
- **Time-varying**: `k(t)` increases linearly until reaching `k_max`

**Called by**: All simulation optimization scripts

---

#### `SecondVersion/MultiMechanismSimulation.py`

**Purpose**: Gillespie simulation for non-time-varying mechanisms

**Differences from time-varying version**:

- Constant rates only
- Used for mechanisms 1-4 (simple, fixed_burst, feedback_onion, fixed_burst_feedback_onion)
- Legacy support for MoM validation

---

### 4. Simulation-Based Optimization

#### `simulation_utils.py`

**Purpose**: Shared utilities for simulation-based optimization

**Key Functions**:

```python
def load_experimental_data()
    """Load experimental data from Excel file"""

def apply_mutant_params(base_params, mutant_type, alpha, beta_k, beta_tau, beta_tau2)
    """Apply mutant-specific modifications to parameters"""

def run_simulation_for_dataset(mechanism, params, n0_list, num_simulations=500)
    """Run multiple simulations and return time differences"""

def calculate_likelihood(experimental_data, simulated_data)
    """Calculate NLL using Kernel Density Estimation (KDE)"""

def get_parameter_bounds(mechanism)
    """Get optimization bounds for each mechanism (handles both simple and time-varying)"""
```


**Key Functions**:

```python
def load_experimental_data()
    """Load experimental data from Excel file"""

def apply_mutant_params(base_params, mutant_type, ...)
    """Apply mutant-specific modifications to parameters"""

def run_simulation_for_dataset(mechanism, params, n0_list, num_simulations=500)
    """Run multiple simulations (Fast/Gillespie) and return time differences"""

def calculate_emd(exp_data, sim_data)
    """Calculate Earth Mover's Distance (Wasserstein) between datasets"""

def calculate_likelihood(experimental_data, simulated_data)
    """Calculate NLL using Kernel Density Estimation (KDE)"""

def get_parameter_bounds(mechanism)
    """Get optimization bounds for each mechanism"""
```

**Key Features**:
- **Automatic Dispatch**: Routes to `FastBetaSimulation` or `FastFeedbackSimulation` when possible for O(1) performance
- **Dual Metrics**: Supports both EMD (robust, default) and NLL/KDE (legacy)
- **Unified Parameter Handling**: Consistent interface for mutation operators across all mechanisms

**Called by**: All `SimulationOptimization_*.py` scripts

---

#### `SimulationOptimization_join.py`

**Purpose**: Joint simulation-based optimization across all strains

**Key Features**:

**Key Features**:

- **Default Metric**: Uses Earth Mover's Distance (EMD) for robust optimization
- **Fast Simulation**: Leverages vectorized sampling for 100x speedup
- **Flexible**: Can switch back to NLL/KDE via `objective_metric='nll'` argument
- **Strain Selection**: Can optimize on specific subsets of strains
- **Unified**: Handles all mechanism types (simple, burst, feedback, time-varying) seamlessly
- **Clean Output**: Simplified logging and unified parameter unpacking

**Workflow**:

```
1. Load experimental data
2. Define objective function:
   a. Run simulations with test parameters
   b. Fit KDE to simulation results
   c. Evaluate experimental data under KDE
   d. Return negative log-likelihood
3. Run differential evolution
4. Save results
```

**Key Function**:

```python
def joint_objective(params_vector, mechanism, datasets, num_simulations=500)
    """Objective function for optimization"""
    # Returns total NLL across all strains
```

**Output Files**: `simulation_optimized_parameters_R1_{mechanism}.txt`

**Usage**:

```python
python SimulationOptimization_join.py
```

---


---

#### `CrossValidation_EMD.py`

**Purpose**: Rigorous model validation using 5-Fold Cross-Validation

**Workflow**:
1. Splits experimental data into 5 stratified folds
2. For each fold:
   - **Train**: Optimize parameters on 80% of data (minimizing EMD)
   - **Validate**: Calculate EMD on held-out 20%
3. Reports average Train EMD and Validation EMD
4. Saves fold-specific parameters and scores

**Usage**:
```bash
python CrossValidation_EMD.py
```

---

#### `SimulationOptimization_independent.py`

**Purpose**: Independent simulation-based optimization strategy

**Key Features**:

- Two-stage approach (wildtype first, then mutants)
- Generally more robust but slower
- Good for difficult convergence cases

---

#### `TestDataPlotSimulation.py`

**Purpose**: Visualize simulation-based optimization results

**Key Features**:

- Loads optimized parameters from simulation files
- Runs new simulations for visualization
- Compares simulation histograms with experimental data
- Creates 2×5 subplot grids (same layout as MoM version)
- Supports both single dataset and multi-dataset plotting
- **New**: 3.5-second bin width for histograms

**Usage**:

```python
# Configuration options
run_all_mechanisms = False  # Set True to test all mechanisms
mechanism = 'time_varying_k_combined'
dataset = 'wildtype'  # Or 'threshold', 'degrade', 'degradeAPC', 'velcade'

python TestDataPlotSimulation.py
```

---

### 5. Model Comparison

#### `model_comparison_aic_bic.py`

**Purpose**: Comprehensive model comparison framework

**What it does**:

- Runs optimization for all 8 mechanisms
- Multiple runs per mechanism (typically 5-10)
- Calculates AIC and BIC for model selection
- Generates summary statistics

**Metrics**:

```
AIC = 2k + 2·NLL
BIC = k·ln(n) + 2·NLL

where:
  k = number of parameters
  n = number of data points (1,423)
  NLL = negative log-likelihood
```

**Output**:

- Summary CSV with mean/std AIC/BIC
- Convergence statistics
- Individual run results

**Usage**:

```bash
# Local run
python model_comparison_aic_bic.py

# HPC cluster
sbatch submit_model_comparison.sh
```

---

#### `model_comparison_wildtype_apc.py`

**Purpose**: Model comparison for specific strain subsets

**Use case**: Compare mechanisms using only wildtype and APC mutant data

**Why useful**:

- Faster optimization with fewer strains
- Test mechanism performance on specific biological conditions
- Isolate effects of particular mutants

---

### 6. Visualization & Analysis

#### `demonstrate_kde.py`

**Purpose**: Educational tool demonstrating KDE vs normal distribution fitting

**What it does**:

1. Loads optimized parameters
2. Runs simulations
3. Fits both KDE (scipy) and normal distribution to simulation data
4. Loads experimental data
5. Calculates NLL for both methods against experimental data
6. Creates side-by-side comparison plots

**Key Visualizations**:

- Histogram of simulation data
- KDE fit (scipy) with NLL in legend
- Normal distribution fit with NLL in legend
- Mean and statistics

**Output**: Compares which method (KDE vs Normal) better fits experimental data

**Usage**:

```python
python demonstrate_kde.py
```

**Configurable**:

- `mechanism`: Which mechanism to test
- `mutant_type`: Which strain ('wildtype', 'threshold', etc.)
- `num_simulations`: How many simulations to run

---

#### `analyze_model_comparison_results.py`

**Purpose**: Analyze model comparison outputs

**Features**:

- Parses optimization output files
- Generates diagnostic plots
- Parameter distribution analysis
- Convergence diagnostics

---

### 7. Parameter Recovery & Validation

#### `SecondVersion/parameter_recovery_mom.py`

**Purpose**: Validate that optimization can recover known parameters

**Workflow**:

```
1. Choose "true" parameters
2. Generate synthetic data from these parameters
3. Run optimization on synthetic data
4. Compare recovered parameters to true parameters
5. Assess parameter identifiability
```

**Output**:

- Recovery error statistics
- Correlation matrices
- Parameter distribution plots

---

#### `analyze_parameter_recovery.py`

**Purpose**: Analyze parameter recovery results

**Features**:

- Recovery accuracy metrics
- Identifiability analysis
- Correlation structure visualization

---

### 8. HPC Deployment

#### `submit_model_comparison.sh`

**Purpose**: SLURM job script for HPC clusters

**Features**:

```bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --mem=64GB

# Runs model_comparison_aic_bic.py with parallel processing
```

**Usage**:

```bash
sbatch submit_model_comparison.sh
```

---

#### `ARC.sh`

**Purpose**: Generic job submission template for ARC cluster

---

## Workflow Summary

### Standard Analysis Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DATA PREPARATION                                         │
│    - Experimental data in Data/All_strains_SCStimes.xlsx   │
│    - 5 strains × 2 chromosome pairs = 10 datasets         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. QUICK EXPLORATION (Optional)                            │
│    - Use MoM pipeline for fast initial fits               │
│    - Run: python SecondVersion/MoMOptimization_join.py    │
│    - Visualize: python SecondVersion/TestDataPlot.py       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. FULL OPTIMIZATION                                        │
│    - Use simulation pipeline for accurate fits             │
│    - Run: python SimulationOptimization_join.py           │
│    - Or submit to cluster: sbatch submit_job.sh           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. VISUALIZATION                                            │
│    - Inspect fits: python TestDataPlotSimulation.py       │
│    - Check convergence and quality                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. VALIDATION & COMPARISON                                  │
│    - Run Cross-Validation: python CrossValidation_EMD.py    │
│    - Compare mechanisms: python model_comparison_cv_emd.py  │
│    - Assess generalization and sample efficiency           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. VALIDATION (Optional)                                    │
│    - Parameter recovery studies                            │
│    - Diagnostic analyses                                   │
│    - Sensitivity analyses                                  │
└─────────────────────────────────────────────────────────────┘
```

### File Dependencies

```
Chromosomes_Theory.py
    ↓
    ├─→ MoMCalculations.py
    │       ↓
    │       ├─→ MoMOptimization_join.py
    │       │       ↓
    │       │       └─→ TestDataPlot.py
    │       │
    │       └─→ MoMOptimization_independent.py
    │
    └─→ MultiMechanismSimulationTimevary.py
            ↓
            ├─→ simulation_utils.py
            │       ↓
            │       ├─→ SimulationOptimization_join.py
            │       │       ↓
            │       │       └─→ TestDataPlotSimulation.py
            │       │
            │       └─→ SimulationOptimization_independent.py
            │
            └─→ demonstrate_kde.py
```

---

## Quick Start Guide

### For New Users

1. **Install Dependencies**:

```bash
pip install -r requirements.txt
```

2. **Run a Quick Test** (MoM - Fast):

```bash
cd SecondVersion
python MoMOptimization_join.py
python TestDataPlot.py
```

3. **Run Full Optimization** (Simulation - Accurate):

```bash
python SimulationOptimization_join.py
python TestDataPlotSimulation.py
```

4. **Compare All Mechanisms**:

```bash
python model_comparison_aic_bic.py
```

### For HPC Users

1. **Single Mechanism**:

```bash
sbatch submit_job.sh
```

2. **Full Model Comparison**:

```bash
sbatch submit_model_comparison.sh
```

---

## Key Design Decisions

### 1. Two Separate Pipelines

**Why**: MoM provides fast exploration; simulation provides accurate final results

### 2. Ratio-Based Parameterization

**Why**: Reduces parameter space and enforces biological relationships between chromosomes

### 3. KDE vs Normal Approximation

**Why**: KDE captures non-normal distributions (skewness, heavy tails) that biological data often exhibits

### 4. Tau Parameterization (k_max, tau) vs (k_1, k_max)

**Why**: Tau has direct biological interpretation (time to full activation) and better numerical properties

### 5. Raw NLL (No Normalization)

### 5. EMD vs NLL (Wasserstein Distance)

**Why**: 
- **Robustness**: EMD doesn't require bandwidth tuning like KDE
- **Stability**: Provides meaningful gradients even when distributions don't overlap
- **Interpretability**: Units are in minutes (average shift needed), unlike abstract NLL values
- **Efficiency**: Converges faster with fewer simulations for many mechanisms

---

## Common Tasks

### Adding a New Mechanism

1. Add calculations to `Chromosomes_Theory.py`
2. Update `MoMCalculations.py` (for MoM pipeline)
3. Update `MultiMechanismSimulationTimevary.py` (for simulation pipeline)
4. Add parameter bounds to `simulation_utils.py`
5. Update model comparison scripts

### Changing Parameter Bounds

Edit `simulation_utils.py`:

```python
def get_parameter_bounds(mechanism):
    bounds = [
        (1.0, 50.0),    # n2 - modify as needed
        # ... etc
    ]
```

### Running on Subset of Strains

In `SimulationOptimization_join.py`:

```python
results = run_optimization(
    mechanism,
    datasets,
    selected_strains=['wildtype', 'degradeAPC']  # Specify strains
)
```

---

## Troubleshooting

### Optimization Not Converging

1. Check parameter bounds (may be too restrictive)
2. Try independent optimization strategy
3. Increase max_iterations
4. Use initial guess from MoM results

### High NLL Values

1. Check constraint violations (n < N required)
2. Verify experimental data loaded correctly
3. Increase num_simulations (may need >500 for complex mechanisms)

### Memory Issues

1. Reduce num_simulations
2. Process strains sequentially instead of in parallel
3. Use HPC cluster with more memory

---

## Performance Benchmarks

| Task                             | Method     | Time        | Hardware         |
| -------------------------------- | ---------- | ----------- | ---------------- |
| Single mechanism (MoM)           | Joint      | 5-30 min    | Laptop (4 cores) |
| Single mechanism (Sim)           | Joint      | 2-8 hours   | Laptop (4 cores) |
| Full model comparison (8×5 runs) | Simulation | 24-48 hours | HPC (32 cores)   |
| Parameter recovery (100 runs)    | MoM        | 1-2 hours   | HPC (32 cores)   |

---

## Version History

- **v2.2** (Dec 2024): Code cleanup and refactoring - simplified logging, unified parameter handling, improved maintainability
- **v2.1** (Nov 2024): Added KDE demonstration, 3.5s histogram bins
- **v2.0** (Oct 2024): Model comparison framework, raw NLL implementation
- **v1.5** (Sep 2024): Simulation-based optimization pipeline
- **v1.0** (Aug 2024): Initial MoM implementation

---

## References

For detailed mathematical derivations and biological context, see:

- `Docs/Project_Documentation.md` - Full scientific documentation
- `Docs/ANALYSIS_WORKFLOW.md` - Step-by-step analysis guide
- `Docs/ARC_DEPLOYMENT_GUIDE.md` - HPC deployment instructions

---

**Note**: This document focuses on code structure and usage. For mathematical details, biological interpretation, and theoretical background, refer to `Project_Documentation.md`.
