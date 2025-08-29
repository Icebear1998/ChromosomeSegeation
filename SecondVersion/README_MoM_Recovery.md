# MoM Parameter Recovery Study - SecondVersion

This directory contains a local parameter recovery study system using Method of Moments (MoM) approach. This complements the simulation-based recovery study running on ARC.

## 📁 Files Overview

### Core Scripts

- **`parameter_recovery_mom.py`** - Main MoM parameter recovery script
- **`analyze_parameter_recovery_mom.py`** - Analysis and visualization script
- **`test_mom_recovery.py`** - Test script to verify system works

### Existing MoM System

- **`MoMOptimization_join.py`** - Joint MoM optimization (generates ground truth parameters)
- **`MoMCalculations.py`** - Method of Moments calculations
- **`Chromosomes_Theory.py`** - Core theory functions
- Other MoM-related files...

## 🎯 Purpose

This MoM-based parameter recovery study serves several purposes:

1. **Local Comparison**: Run parameter recovery locally while ARC simulation study is running
2. **Method Comparison**: Compare MoM vs simulation-based parameter identifiability
3. **Approximation Assessment**: Evaluate how well MoM approximations capture parameter constraints
4. **Quick Iteration**: Test parameter recovery concepts without waiting for HPC resources

## 🚀 Quick Start

### Step 1: Verify System Works

```bash
cd SecondVersion/
python test_mom_recovery.py
```

### Step 2: Generate Ground Truth (if needed)

```bash
# Run MoM optimization to get ground truth parameters
python MoMOptimization_join.py
```

### Step 3: Run MoM Parameter Recovery

```bash
# Run the MoM parameter recovery study
python parameter_recovery_mom.py
```

### Step 4: Analyze Results

```bash
# Analyze and visualize the results
python analyze_parameter_recovery_mom.py
```

## ⚙️ Configuration

Edit `parameter_recovery_mom.py` main function to configure:

```python
# Configuration
mechanism = 'fixed_burst_feedback_onion'  # Mechanism to study
gamma_mode = 'separate'                   # 'unified' or 'separate'
n_recovery_runs = 25                      # Number of recovery attempts
synthetic_data_size = 300                 # Size of synthetic dataset
max_iterations = 100                      # Max iterations per recovery
```

## 📊 How It Works

### 1. Ground Truth Establishment

- Uses optimized parameters from `MoMOptimization_join.py` as ground truth
- Loads parameters from `optimized_parameters_*.txt` files

### 2. Synthetic Data Generation

- Uses MoM theoretical PDFs to generate synthetic data
- Creates realistic time-series data for all strains (wildtype, threshold, degrate, degrateAPC)
- Samples from theoretical distributions instead of running simulations

### 3. Parameter Recovery

- Runs multiple optimization attempts from random starting points
- Uses the same `joint_objective` function as the original MoM optimization
- Tests if optimization can recover the known ground truth parameters

### 4. Analysis and Visualization

- Calculates parameter recovery accuracy and sloppiness metrics
- Generates distribution plots, correlation matrices, and error summaries
- Provides comprehensive text report with recommendations

## 📈 Expected Output

### Files Generated:

1. **`mom_parameter_recovery_MECHANISM_TIMESTAMP.csv`**
   - All recovered parameter sets with ground truth comparisons
2. **`mom_synthetic_data_MECHANISM_TIMESTAMP.csv`**

   - Synthetic datasets used as recovery targets

3. **Analysis Plots:**

   - `mom_parameter_recovery_distributions_*.png`
   - `mom_parameter_correlations_*.png`
   - `mom_nll_convergence_analysis_*.png`
   - `mom_recovery_error_summary_*.png`

4. **`mom_recovery_analysis_report_*.txt`**
   - Comprehensive analysis report

## 🔍 Interpreting Results

### Parameter Quality (same as simulation-based):

- 🟢 **Excellent**: ≤5% recovery error
- 🟡 **Good**: ≤10% recovery error
- 🟠 **Fair**: ≤25% recovery error
- 🔴 **Poor**: >25% recovery error

### MoM-Specific Insights:

- **High accuracy**: MoM approximation captures parameter well
- **Low accuracy**: MoM may miss important parameter constraints
- **High sloppiness**: Parameter poorly constrained by moments
- **Low sloppiness**: Parameter well-constrained by moments

## 🆚 MoM vs Simulation Comparison

### MoM Advantages:

- ✅ **Fast**: No time-consuming simulations
- ✅ **Analytical**: Uses theoretical PDFs directly
- ✅ **Local**: Runs on local machine immediately
- ✅ **Deterministic**: Same synthetic data each time

### MoM Limitations:

- ❌ **Approximation**: May miss simulation-based constraints
- ❌ **Less Realistic**: Synthetic data from theory, not simulation
- ❌ **PDF Dependent**: Limited by quality of MoM approximations

### Simulation Advantages:

- ✅ **Realistic**: Uses actual stochastic simulations
- ✅ **Complete**: Captures all model dynamics
- ✅ **No Approximation**: Direct from model equations

### Simulation Limitations:

- ❌ **Slow**: Requires many simulations per evaluation
- ❌ **HPC Dependent**: Needs high-performance computing
- ❌ **Stochastic**: Results vary between runs

## 🎯 Use Cases

### 1. Quick Parameter Assessment

Run MoM recovery to quickly assess which parameters might be identifiable before committing to expensive simulation studies.

### 2. MoM Validation

Compare MoM vs simulation recovery to validate that MoM approximations are capturing the right parameter constraints.

### 3. Method Development

Test new parameter recovery approaches locally before implementing on HPC systems.

### 4. Educational Understanding

Understand parameter identifiability concepts without needing access to HPC resources.

## 🛠️ Troubleshooting

### Common Issues:

**No parameter files found:**

```bash
# Run MoM optimization first
python MoMOptimization_join.py
```

**Import errors:**

```bash
# Ensure you're in the SecondVersion directory
cd SecondVersion/
python test_mom_recovery.py
```

**Low success rate:**

- Check that ground truth parameters are reasonable
- Verify MoM PDF calculations are working
- Consider adjusting parameter bounds

**Poor parameter recovery:**

- May indicate MoM approximation limitations
- Compare with simulation-based results
- Consider hybrid MoM+simulation approaches

## 📋 Workflow Integration

### Typical Workflow:

1. **Local MoM Study**: Run this system immediately
2. **ARC Simulation Study**: Submit to HPC for comprehensive analysis
3. **Comparison**: Compare MoM vs simulation results
4. **Method Selection**: Choose best approach for each parameter
5. **Publication**: Report both approaches for completeness

### Timeline:

- **MoM Study**: 30 minutes - 2 hours (local)
- **Simulation Study**: 12-24 hours (ARC)
- **Comparison Analysis**: 1-2 hours (local)

This allows you to get initial insights immediately while waiting for the comprehensive simulation-based results from ARC.

## 📞 Getting Help

- Check `test_mom_recovery.py` output for system diagnostics
- Review MoM optimization files for ground truth parameter issues
- Compare with main directory simulation-based recovery for validation
- Ensure all SecondVersion MoM files are present and up-to-date
