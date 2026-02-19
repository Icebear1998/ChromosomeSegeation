# Chromosome Segregation Timing — Stochastic Models & Parameter Estimation

A computational framework for modeling the stochastic cleavage of cohesin complexes during chromosome segregation.

## Overview

Sister chromatids are held together by cohesin complexes and separate once enough cohesins have been cleaved by separase. This project models that process as an **inhomogeneous Poisson process** with a time-dependent cleavage rate that ramps linearly to a maximum value *k*<sub>max</sub> over a timescale *τ*:

$$
k(t) = \begin{cases} k_{\max}\,\frac{t}{\tau}, & 0 \le t \le \tau \\[4pt] k_{\max}, & t > \tau \end{cases}
$$

Each of the three fission-yeast chromosomes starts with *N<sub>i</sub>* cohesins and separates when its count falls to a threshold *n<sub>i</sub>*. The model outputs two pairwise separation-time differences, **ΔT₁₂ = T₁ − T₂** and **ΔT₃₂ = T₃ − T₂**, which are compared to experimental distributions.

### Mechanistic Variants

| Model | Extra Feature | Extra Parameters |
|---|---|---|
| **Basic** | — | — |
| **Processive separase action** | Fixed burst size *b* cohesins removed per event | *b* |
| **Steric hindrance** | Effective rate scales with surface-to-volume ratio of remaining cohesin sphere | *n*<sub>inner</sub> |

Each variant is tested with two separase activity ramp regimes:
- **Slow ramp**: 2 < τ < 240 s
- **Fast ramp** (separase auto-activation): 0.5 < τ < 5 s

This yields **six model variants** in total. Perturbation conditions (MBC treatment, separase mutant, APC/C mutant, velcade treatment) are modeled with multiplicative factors on the relevant parameters.

## Project Structure

```
ChromosomeSegregation/
├── Data/
│   └── All_strains_SCStimes.xlsx          # Experimental separation-time data
├── Parameter files/                       # Optimized parameter sets per mechanism
│   ├── simulation_optimized_parameters_time_varying_k.txt
│   ├── simulation_optimized_parameters_time_varying_k_fixed_burst.txt
│   ├── simulation_optimized_parameters_time_varying_k_steric_hindrance.txt
│   ├── simulation_optimized_parameters_time_varying_k_combined.txt
│   └── ..._wfeedback.txt variants
│
│── Simulation Engine ──────────────────────────────────────────────────
├── MultiMechanismSimulationTimevary.py    # Gillespie simulation 
├── FastBetaSimulation.py                  # O(1) fast simulation via order statistics
├── FastFeedbackSimulation.py              # Vectorized simulation for steric hindrance
├── simulation_utils.py                    # Shared utilities (data loading, EMD, parameter handling)
│
│── Optimization & Validation ──────────────────────────────────────────
├── SimulationOptimization.py              # Differential evolution + L-BFGS-B optimization
├── CrossValidation.py                     # 5-fold cross-validation with EMD
├── ModelComparison.py                     # Cross-validated model comparison across mechanisms
├── AggregateResults.py                    # Aggregate CV results from HPC job arrays
│
│── Analysis Scripts ───────────────────────────────────────────────────
├── SensitivityAnalysis_ParameterSweep.py  # Parameter sensitivity analysis
├── AnalyzeSampleEfficiency.py             # EMD convergence vs. simulation sample size
├── AnalyzeTolEfficiency.py                # Optimizer tolerance efficiency analysis
├── AnalyzePopulationEfficiency.py         # Optimizer population size efficiency analysis
│
│── Visualization ──────────────────────────────────────────────────────
├── PlotSimulationVsData.py                # Overlay simulation histograms on experimental data
├── PlotFeedbackImpact.py                  # Compare feedback vs. no-feedback model variants
│
│── HPC Job Scripts ────────────────────────────────────────────────────
├── submit_job.sh                          # SLURM script for single-mechanism optimization
├── submit_model_comparison_cv.sh          # SLURM job array for parallel CV across mechanisms
├── submit_tol_analysis.sh                 # SLURM script for tolerance analysis
├── submit_pop_analysis.sh                 # SLURM script for population size analysis
│
└── requirements.txt                       # Python dependencies
```

## Simulation Algorithms

### 1. Gillespie Simulation (`MultiMechanismSimulationTimevary.py`)

A modified Gillespie algorithm that handles the time-dependent cleavage rate via inverse-CDF sampling. Used as the **reference implementation** and for generating sample trajectories.

### 2. Fast Simulation — Order Statistics (`FastBetaSimulation.py`)

For models **without** steric hindrance, individual cohesin cleavage events are independent. The separation time equals the (*n*+1)-th order statistic of *N* i.i.d. waiting times, which follows a **Beta distribution**. Each chromosome's separation time is sampled in **O(1)** by:

1. Drawing one sample from Beta(*N* − *n*, *n* + 1)
2. Transforming through the inverse CDF of the cleavage-time distribution

This provides a **100×–1000× speedup** over the Gillespie algorithm. The processive separase model is handled by transforming to effective counts: *N'* = ⌈*N*/*b*⌉.

### 3. Fast Simulation — Vectorized (`FastFeedbackSimulation.py`)

For the **steric hindrance** model, cleavage events are state-dependent and not independent. The simulation is accelerated by **vectorizing across *M* independent trajectories**, computing the next-event time for all simulations simultaneously at each step.

## Parameter Estimation

### Objective Metric — Earth Mover's Distance (EMD)

Model fit is assessed using the **Wasserstein-1 distance** (EMD) between experimental and simulated distributions of ΔT₁₂ and ΔT₃₂. The aggregate EMD sums over five experimental conditions (wildtype, MBC, separase mutant, APC/C mutant, velcade). Each evaluation generates *M* = 10,000 simulated samples.

### Optimizer — Differential Evolution + Local Refinement

Parameters are estimated using a **two-stage protocol**:

1. **Global search**: `scipy.optimize.differential_evolution` with strategy `best1bin`, population size 10×, mutation (0.5, 1.0), recombination 0.7, tolerance 0.01
2. **Local refinement**: L-BFGS-B on the top candidates

### Cross-Validation

**5-fold cross-validation** is performed to reduce overfitting. Data are split into stratified folds; optimization runs on 80% and validation EMD is computed on the held-out 20%. Model comparison uses the mean ± SE of cross-validated aggregate EMD.

## Key Scripts & Workflows

### Run a Single Optimization

```bash
python SimulationOptimization.py <mechanism>
# e.g., python SimulationOptimization.py time_varying_k
```

Available mechanisms:
- `time_varying_k` — Basic model
- `time_varying_k_fixed_burst` — Processive separase action
- `time_varying_k_steric_hindrance` — Steric hindrance
- `time_varying_k_combined` — Processive + steric hindrance
- Append `_wfeedback` for fast-ramp (separase auto-activation) variants

### Run Cross-Validated Model Comparison

```bash
# Single mechanism
python CrossValidation.py

# All mechanisms in parallel (HPC)
sbatch submit_model_comparison_cv.sh

# Aggregate results after all jobs complete
python AggregateResults.py <SLURM_JOB_ID>
```

### Sensitivity Analysis

```bash
python SensitivityAnalysis_ParameterSweep.py
```

Performs OAT sensitivity analysis by sweeping each parameter across its range while holding others at optimized values. Outputs CSV data and plots for each parameter.

### Visualize Results

```bash
# Overlay simulation vs. experimental histograms
python PlotSimulationVsData.py

# Compare feedback vs. no-feedback variants
python PlotFeedbackImpact.py
```

### Hyperparameter Benchmarks

```bash
# EMD convergence vs. sample size
python AnalyzeSampleEfficiency.py

# Optimizer tolerance efficiency
python AnalyzeTolEfficiency.py

# Population size efficiency
python AnalyzePopulationEfficiency.py
```

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| NumPy | Array operations, random sampling |
| SciPy | Optimization (`differential_evolution`, `L-BFGS-B`), Wasserstein distance |
| Pandas | Data I/O and tabulation |
| openpyxl | Reading experimental Excel data |
| Matplotlib | Plotting |
| Seaborn | Enhanced statistical plots |
| PyTorch | (Optional) GPU-accelerated operations |

## HPC Usage

Job scripts are configured for **SLURM** on the Virginia Tech ARC cluster. Adjust `#SBATCH` directives and the `module load` / `conda activate` lines for your environment.

```bash
# Single optimization job
sbatch submit_job.sh time_varying_k

# Parallel model comparison (job array)
sbatch submit_model_comparison_cv.sh
```
