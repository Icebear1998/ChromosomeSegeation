# Parameter Recovery Analysis Workflow

This document describes the complete workflow for running parameter recovery studies and analyzing results.

## 🎯 Overview

The parameter recovery system is split into two parts:

- **ARC (High-Performance Computing)**: Data generation and recovery runs
- **Local Machine**: Analysis and visualization

## 📋 Complete Workflow

### Step 1: Deploy to ARC

1. **Prepare the ARC package**:

   ```bash
   # The ARC_ParameterRecovery/ folder contains everything needed for ARC
   ls ARC_ParameterRecovery/
   ```

2. **Copy to ARC**:

   ```bash
   scp -r ARC_ParameterRecovery your_username@arc.university.edu:~/
   ```

3. **Submit the job on ARC**:
   ```bash
   # On ARC
   cd ARC_ParameterRecovery
   sbatch submit_recovery.slurm
   ```

### Step 2: Monitor Progress

```bash
# Check job status
squeue -u your_username

# Monitor progress
tail -f recovery_study_JOBID.out
```

### Step 3: Download Results (After Completion)

```bash
# From your local machine
scp your_username@arc.university.edu:~/ARC_ParameterRecovery/recovery_results_*.csv ./
scp your_username@arc.university.edu:~/ARC_ParameterRecovery/recovery_summary_*.txt ./
scp your_username@arc.university.edu:~/ARC_ParameterRecovery/synthetic_data_*.csv ./
```

### Step 4: Analyze Results Locally

```bash
# Basic analysis (auto-detects most recent results)
python analyze_parameter_recovery.py

# Specify a particular results file
python analyze_parameter_recovery.py recovery_results_time_varying_k_combined_20241201_143022.csv

# Generate only the report (no plots)
python analyze_parameter_recovery.py --no-plots

# Don't save files (display only)
python analyze_parameter_recovery.py --no-save
```

## 📊 Expected Analysis Output

### Plots Generated:

1. **Parameter Recovery Distributions** (`parameter_recovery_distributions_*.png`)

   - Histograms showing recovered parameter values vs. ground truth
   - Color-coded by recovery accuracy

2. **Parameter Correlations** (`parameter_correlations_*.png`)

   - Heatmap showing correlations between recovered parameters
   - Identifies parameter dependencies and trade-offs

3. **NLL and Convergence Analysis** (`nll_convergence_analysis_*.png`)

   - Distribution of final negative log-likelihood values
   - Success rate of optimization convergence

4. **Recovery Error Summary** (`recovery_error_summary_*.png`)
   - Bar charts showing relative error and coefficient of variation
   - Color-coded by parameter quality (good/fair/poor)

### Text Report:

- **Comprehensive Analysis Report** (`recovery_analysis_report_*.txt`)
  - Detailed statistics for each parameter
  - Model identifiability assessment
  - Recommendations for model improvement

## 🔍 Interpreting Results

### Parameter Quality Classification:

**Recovery Error (Relative to Ground Truth)**:

- 🟢 **Excellent**: ≤5% error
- 🟡 **Good**: ≤10% error
- 🟠 **Fair**: ≤25% error
- 🔴 **Poor**: >25% error

**Parameter Sloppiness (Coefficient of Variation)**:

- 🟢 **Low sloppiness**: CV ≤ 0.1 (well-constrained)
- 🟡 **Moderate sloppiness**: CV ≤ 0.5 (moderately constrained)
- 🔴 **High sloppiness**: CV > 0.5 (poorly constrained)

### Model Identifiability:

- **Good**: >80% parameters with good recovery, >60% with low sloppiness
- **Moderate**: >60% parameters with good recovery
- **Poor**: <60% parameters with good recovery

## 🛠️ Troubleshooting

### Common Issues:

**Low Success Rate (<50% convergence)**:

- Increase `max_iterations` in ARC job script
- Check parameter bounds are reasonable
- Verify synthetic data generation succeeded

**Poor Parameter Recovery (>25% error)**:

- Parameters may be practically unidentifiable
- Consider model reparameterization
- Add parameter constraints or priors

**High Parameter Sloppiness (CV > 0.5)**:

- Parameters are poorly constrained by data
- May indicate model overparameterization
- Consider reducing model complexity

**No Analysis Output**:

- Ensure results CSV file is in correct format
- Check that recovery results contain successful runs
- Verify `analyze_parameter_recovery.py` is in local directory

## 📁 File Organization

```
Your Local Project/
├── analyze_parameter_recovery.py          # Analysis script (LOCAL ONLY)
├── recovery_results_*.csv                # Downloaded from ARC
├── recovery_summary_*.txt                # Downloaded from ARC
├── synthetic_data_*.csv                  # Downloaded from ARC
├── parameter_recovery_distributions_*.png # Generated locally
├── parameter_correlations_*.png           # Generated locally
├── nll_convergence_analysis_*.png        # Generated locally
├── recovery_error_summary_*.png          # Generated locally
├── recovery_analysis_report_*.txt        # Generated locally
└── ARC_ParameterRecovery/                # For deployment to ARC
    ├── run_recovery_study.py
    ├── submit_recovery.slurm
    └── [other ARC files]
```

## 🎯 Study Goals Reminder

The parameter recovery study helps you:

1. **Sanity Check**: Verify optimization can recover known parameters
2. **Parameter Identifiability**: Determine which parameters are uniquely determinable
3. **Model Sloppiness**: Identify poorly constrained parameters
4. **Optimization Reliability**: Assess consistency of optimization approach

Good results show high recovery success rate, low parameter errors, and clear identification of well-constrained vs. sloppy parameters.

## 📞 Getting Help

- Check SLURM logs on ARC: `recovery_study_JOBID.out` and `recovery_study_JOBID.err`
- Look for error files: `recovery_error_*.txt`
- Review the analysis report for specific recommendations
- Contact ARC support for system-specific issues
