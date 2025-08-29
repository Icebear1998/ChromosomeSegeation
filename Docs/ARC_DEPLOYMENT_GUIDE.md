# Parameter Recovery Study - ARC Deployment Guide

This guide explains how to deploy and run the parameter recovery study on your university's ARC (Advanced Research Computing) system.

## 📁 Files Created

### Ground Truth Parameters

- `ground_truth_parameters_rounded.txt` - Rounded ground truth parameters from your optimization results

### Scripts

- `run_recovery_local_test.sh` - Local test script (run this first)
- `run_recovery_arc.sh` - Production ARC job script
- `parameter_recovery_study.py` - Main recovery study script
- `analyze_parameter_recovery.py` - Analysis script (run locally after ARC job)

## 🚀 Step-by-Step Deployment

### Step 1: Local Testing (REQUIRED)

Before deploying to ARC, test locally to ensure everything works:

```bash
# Run the local test (takes ~3-5 minutes)
./run_recovery_local_test.sh
```

This will:

- Check all required files and dependencies
- Run a minimal recovery study (3 runs, small datasets)
- Verify the system works correctly
- Create test results for verification

**Only proceed to ARC if the local test passes!**

### Step 2: Prepare Files for ARC

Copy these files to your ARC working directory:

**Required Python files:**

```
parameter_recovery_study.py
analyze_parameter_recovery.py
simulation_utils.py
Chromosomes_Theory.py
MultiMechanismSimulationTimevary.py
ground_truth_parameters_rounded.txt
run_recovery_arc.sh
```

**Required data files:**

```
Data/All_strains_SCStimes.xlsx
```

### Step 3: Modify ARC Script (if needed)

Edit `run_recovery_arc.sh` to match your ARC system:

```bash
# Modify SLURM parameters for your system:
#SBATCH --partition=compute          # Change to your partition name
#SBATCH --time=24:00:00             # Adjust time limit
#SBATCH --cpus-per-task=32          # Adjust CPU count
#SBATCH --mem=64G                   # Adjust memory

# Modify module loading:
# module load python/3.9             # Your Python module
# module load scipy-stack            # Your SciPy module
```

### Step 4: Submit ARC Job

```bash
# Submit the job
sbatch run_recovery_arc.sh

# Check job status
squeue -u your_username

# Monitor progress
tail -f recovery_JOBID.out
```

### Step 5: Configuration Options

The ARC script uses these production settings:

```python
CONFIG = {
    'mechanism': 'time_varying_k_combined',
    'n_recovery_runs': 100,          # 100 recovery attempts
    'synthetic_data_size': 1000,     # 1000 synthetic data points
    'recovery_simulations': 200,     # 200 simulations per recovery
    'max_iterations': 150,           # 150 optimization iterations
    'n_processes': 32,               # Use all available CPUs
}
```

**Estimated runtime:** 8-16 hours (depending on CPU count and settings)

## 📊 Expected Output Files

After successful completion, you'll get:

1. **`parameter_recovery_time_varying_k_combined_TIMESTAMP.csv`**

   - Main results with all recovered parameters
   - Ground truth values for comparison
   - Convergence status and NLL values

2. **`synthetic_data_time_varying_k_combined_TIMESTAMP.csv`**

   - Synthetic dataset used as target
   - Reference for what the recovery was trying to match

3. **`recovery_summary_time_varying_k_combined_TIMESTAMP.txt`**

   - Summary statistics and job information
   - Success rates and performance metrics

4. **`recovery_JOBID.out`** and **`recovery_JOBID.err`**
   - SLURM job output and error logs

## 🔍 Analysis After ARC Job

Download the results and run analysis locally:

```bash
# Download the CSV file from ARC
scp your_arc_system:path/to/parameter_recovery_*.csv .

# Run analysis locally
python analyze_parameter_recovery.py
```

This will generate:

- Visualization plots (PNG files)
- Comprehensive analysis report (TXT file)
- Parameter recovery statistics

## ⚙️ Troubleshooting

### Common Issues:

1. **Local test fails:**

   - Check Python environment and package versions
   - Ensure all required files are present
   - Verify ground truth parameter file is readable

2. **ARC job fails to start:**

   - Check SLURM parameters match your system
   - Verify partition name and resource limits
   - Ensure modules are loaded correctly

3. **ARC job runs but fails:**

   - Check `recovery_JOBID.err` for error messages
   - Verify all Python files were copied correctly
   - Check memory and time limits

4. **Low success rate in results:**
   - Consider increasing `max_iterations`
   - Check if parameter bounds are appropriate
   - Verify synthetic data generation worked correctly

### Resource Recommendations:

- **CPUs:** 16-32 cores (more = faster)
- **Memory:** 32-64 GB (depends on dataset size)
- **Time:** 12-24 hours (depends on configuration)
- **Storage:** ~1-2 GB for output files

## 📈 Interpreting Results

The analysis will tell you:

- **Parameter Recovery Accuracy:** How well each parameter was recovered
- **Model Sloppiness:** Which parameters are poorly constrained
- **Identifiability:** Whether parameters are uniquely determinable
- **Correlations:** Which parameters are interdependent

Good recovery typically shows:

- Mean relative error < 10% for well-identified parameters
- Low coefficient of variation (< 0.5) for constrained parameters
- High success rate (> 70%) for convergence

## 🎯 Ground Truth Parameters Used

The study uses these rounded ground truth values:

```
n2 = 5.12          N2 = 168.84        k_max = 0.01314    tau = 47.26
r21 = 2.42         r23 = 1.46         R21 = 1.29         R23 = 2.47
burst_size = 14.28 n_inner = 20.50    alpha = 0.654
beta_k = 0.720     beta_tau = 2.90
```

These were derived from your optimized parameters in `simulation_optimized_parameters_time_varying_k_combined_finetune.txt`.
