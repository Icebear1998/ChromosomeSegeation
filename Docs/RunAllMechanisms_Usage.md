# RunAllMechanisms.py Usage Guide

## Overview

This script runs a comprehensive analysis of all chromosome segregation mechanisms with both joint and independent optimization strategies, organizes results in structured folders, and performs AIC/BIC model comparison.

## Prerequisites

### Required Python Packages

The script requires the following packages to be installed:

```bash
pip install numpy pandas matplotlib scipy openpyxl
```

### Required Files

Make sure the following files are present in your working directory:

- `MoMOptimization_join.py`
- `MoMOptimization_independent.py`
- `TestDataPlot.py`
- `MoMCalculations.py`
- `MultiMechanismSimulation.py`
- `Data/All_strains_SCStimes.xlsx`

## Running the Analysis

### Basic Usage

```bash
python RunAllMechanisms.py
```

This will:

1. Run all 5 mechanisms with both optimization strategies (10 total runs)
2. Create organized results in `ResultsAllRun/` folder
3. Generate comparison plots and parameter files
4. Perform AIC/BIC model selection
5. Create comprehensive analysis reports

## Mechanisms Tested

- `simple`: Basic harmonic sum degradation
- `fixed_burst`: Fixed-size burst degradation
- `time_varying_k`: Time-dependent degradation rate
- `feedback_onion`: Onion-like feedback mechanism
- `fixed_burst_feedback_onion`: Combined burst + onion feedback

## Optimization Strategies

- `join`: Joint optimization across all mutant strains
- `independent`: Independent optimization (wild-type first, then mutants)

## Output Structure

### Directory Organization

```
ResultsAllRun/
├── simple_join/
│   ├── parameters_simple_join.txt
│   ├── plot_simple_join.png
│   └── metrics_simple_join.json
├── simple_independent/
│   ├── parameters_simple_independent.txt
│   ├── plot_simple_independent.png
│   └── metrics_simple_independent.json
├── [... other mechanism_strategy folders ...]
├── model_comparison_report.csv
└── analysis_summary.txt
```

### Plot Layout

Each plot shows a 2x2 layout with datasets in this order:

- **Top Left**: Wildtype
- **Top Right**: Initial proteins mutant
- **Bottom Left**: Threshold mutant
- **Bottom Right**: Degradation rate mutant

Each subplot shows:

- Experimental data (histograms)
- Stochastic simulation data (histograms)
- Method of Moments PDF (lines)
- Statistical summaries

### Parameter Files

Each parameter file contains:

- Wild-type parameters (n1, n2, n3, N1, N2, N3, k)
- Mechanism-specific parameters
- Mutant parameters (alpha, beta_k, beta2_k, gamma)
- Individual and total negative log-likelihoods

### Metrics Files

JSON files containing:

- Negative log-likelihood (NLL)
- Number of parameters
- Number of data points
- Akaike Information Criterion (AIC)
- Bayesian Information Criterion (BIC)

## Model Comparison

### AIC/BIC Analysis

The script automatically performs model selection using:

- **AIC = 2k + 2(-ln(L))**: Penalizes model complexity
- **BIC = ln(n)k + 2(-ln(L))**: More stringent penalty for complexity

Where:

- k = number of parameters
- n = number of data points
- L = likelihood

### Comparison Reports

- `model_comparison_report.csv`: Detailed comparison table
- `analysis_summary.txt`: Human-readable summary with rankings

## Expected Runtime

- **Per mechanism**: 5-30 minutes depending on complexity
- **Total analysis**: 1-5 hours for all mechanisms
- **Plotting**: Additional 10-20 minutes

## Troubleshooting

### Common Issues

1. **Missing packages**: Install required packages with pip
2. **Memory issues**: Reduce `num_sim` parameter in plotting functions
3. **Optimization failures**: Check parameter bounds and data quality
4. **File permissions**: Ensure write access to working directory

### Error Handling

The script includes comprehensive error handling:

- Failed optimizations are logged but don't stop the analysis
- Partial results are saved even if some mechanisms fail
- Detailed error messages help diagnose issues

## Customization

### Modifying Mechanisms

To test different mechanisms, edit the `mechanisms` list in the `MechanismRunner` class:

```python
self.mechanisms = ['simple', 'fixed_burst', 'your_new_mechanism']
```

### Changing Plot Parameters

Modify the `num_sim` parameter in `create_custom_plot()` to adjust simulation size:

```python
# Faster (lower quality): num_sim=500
# Standard: num_sim=1000
# High quality (slower): num_sim=1500
```

### Custom Results Directory

```python
runner = MechanismRunner(base_results_dir="MyCustomResults")
```

## Output Interpretation

### Best Model Selection

- **AIC-based**: Best for prediction and model comparison
- **BIC-based**: Best for identifying the "true" model (more conservative)
- **Lower values are better** for both AIC and BIC

### Model Weights

- AIC/BIC weights show relative support for each model
- Sum to 1.0 across all models
- Higher weights indicate stronger support

### Parameter Interpretation

- **Runtime**: Optimization time (seconds)
- **NLL**: Negative log-likelihood (lower is better fit)
- **Num_Params**: Model complexity
- **Data_Points**: Total experimental observations used

## Example Output

```
BEST MODEL (AIC): feedback_onion (independent)
BEST MODEL (BIC): simple (join)

Top 3 models by AIC:
  1. feedback_onion (independent) - AIC: 2847.3
  2. fixed_burst_feedback_onion (join) - AIC: 2851.7
  3. simple (independent) - AIC: 2856.2
```

This indicates that the onion feedback mechanism with independent optimization provides the best fit to the data, while the simple mechanism with joint optimization is preferred by the more conservative BIC criterion.
