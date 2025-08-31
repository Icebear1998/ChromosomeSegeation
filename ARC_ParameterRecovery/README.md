# Parameter Recovery Study for ARC

This directory contains all files needed to run a comprehensive parameter recovery study on your university's ARC (Advanced Research Computing) system.

## 📁 Directory Structure

```
ARC_ParameterRecovery/
├── Data/
│   └── All_strains_SCStimes.xlsx      # Experimental data
├── run_recovery_study.py              # Main orchestration script
├── submit_recovery.slurm              # SLURM job submission script
├── SimulationOptimization_join.py     # Optimization functions
├── simulation_utils.py                # Simulation utilities
├── MultiMechanismSimulationTimevary.py # Time-varying simulations
├── Chromosomes_Theory.py              # Core theory functions
├── ground_truth_parameters_rounded.txt # Ground truth parameters
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## ⭐ Key Features

1. **Ground Truth Establishment**: Uses your best real-data fit as the "true" parameters
2. **High-Quality Synthetic Data**: Generates noise-free target data for recovery
3. **Parallel Processing**: Runs multiple recovery optimizations simultaneously
4. **Incremental Saving**: ✨ **NEW!** Saves results immediately after each run completes
5. **Checkpoint Backups**: ✨ **NEW!** Creates backup files every 10 completed runs
6. **Progress Tracking**: ✨ **NEW!** Shows real-time progress as runs complete
7. **Comprehensive Results**: Saves all parameter vectors and NLL scores for analysis
8. **HPC Optimized**: Designed for long-running jobs on cluster systems
9. **Fault Tolerant**: If the job gets interrupted, you won't lose completed results

## 🚀 Quick Start Guide

### Step 1: Transfer Files to ARC

Copy this entire directory to your ARC system:

```bash
# From your local machine
scp -r ARC_ParameterRecovery your_username@arc.your_university.edu:~/
```

### Step 2: Set Up Environment on ARC

Log in to ARC and set up the Python environment:

```bash
# Log in to ARC
ssh your_username@arc.your_university.edu

# Navigate to project directory
cd ARC_ParameterRecovery

# Load required modules (modify for your system)
module load anaconda/2022.10

# Create conda environment
conda create --name param_recovery python=3.9 -y
conda activate param_recovery

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configure Job Script

Edit `submit_recovery.slurm` to match your ARC system:

```bash
# Edit the SLURM parameters
nano submit_recovery.slurm

# Key lines to modify:
#SBATCH --partition=normal_q           # Change to your partition
#SBATCH --account=your_account_name    # Change to your account
module load anaconda/2022.10          # Change to your module
conda activate param_recovery          # Change to your environment name
```

### Step 4: Submit Job

Submit the parameter recovery study:

```bash
# Submit the job
sbatch submit_recovery.slurm

# Check job status
squeue -u your_username

# Monitor progress (optional)
tail -f recovery_study_JOBID.out
```

### Step 5: Retrieve Results and Analyze Locally

After completion, download results to your local machine for analysis:

```bash
# From your local machine - download the results
scp your_username@arc.your_university.edu:~/ARC_ParameterRecovery/recovery_results_*.csv ./
scp your_username@arc.your_university.edu:~/ARC_ParameterRecovery/recovery_summary_*.txt ./
scp your_username@arc.your_university.edu:~/ARC_ParameterRecovery/synthetic_data_*.csv ./

# Run analysis locally (NOT on ARC)
python analyze_parameter_recovery.py recovery_results_*.csv
```

**Note**: Analysis and visualization should be done on your local machine, not on ARC. The `analyze_parameter_recovery.py` script is in your main project directory.

## ⚙️ Configuration Options

The study can be configured by editing parameters in `submit_recovery.slurm`:

```bash
# Main parameters
MECHANISM="time_varying_k_combined"     # Mechanism to study
NUM_RUNS=100                            # Number of recovery attempts
GT_ITERATIONS=150                       # Ground truth optimization iterations
SYNTHETIC_SIZE=1000                     # Synthetic data points per strain
MAX_ITERATIONS=120                      # Recovery optimization iterations
```

### Performance Settings

- **CPUs**: 32 cores (modify `--cpus-per-task` in SLURM script)
- **Memory**: 64 GB (modify `--mem` in SLURM script)
- **Time**: 24 hours (modify `--time` in SLURM script)
- **Expected runtime**: 12-24 hours depending on configuration

## 📊 What the Study Does

### Step 1: Ground Truth Establishment

- Optimizes parameters against your real experimental data
- Uses the best-fit parameters as "ground truth" for the recovery study

### Step 2: Synthetic Data Generation

- Generates high-quality synthetic datasets using ground truth parameters
- Creates data for all strains (wildtype, threshold, degrate, degrateAPC)

### Step 3: Parameter Recovery

- Runs 100 independent optimization attempts (configurable)
- Each attempt starts from a different random initial guess
- All attempts try to recover the ground truth from synthetic data
- Runs in parallel using all available CPU cores

### Step 4: Results Collection

- Saves all recovered parameter sets to CSV
- Includes convergence status and likelihood values
- Generates summary statistics

## 📈 Expected Output Files

After successful completion:

1. **`recovery_results_MECHANISM_TIMESTAMP.csv`**

   - All recovered parameter sets
   - Ground truth values for comparison
   - Convergence status and NLL values

2. **`synthetic_data_MECHANISM_TIMESTAMP.csv`**

   - Synthetic datasets used as targets
   - Reference for recovery attempts

3. **`recovery_summary_MECHANISM_TIMESTAMP.txt`**

   - Summary statistics and job information
   - Success rates and performance metrics

4. **`recovery_study_JOBID.out`** and **`recovery_study_JOBID.err`**
   - SLURM job logs

## 🔍 Local Analysis Workflow

**Important**: All analysis and visualization should be done on your local machine, NOT on ARC.

### Download Results from ARC

```bash
# Download all result files to your local project directory
scp your_username@arc.your_university.edu:~/ARC_ParameterRecovery/recovery_results_*.csv ./
scp your_username@arc.your_university.edu:~/ARC_ParameterRecovery/recovery_summary_*.txt ./
scp your_username@arc.your_university.edu:~/ARC_ParameterRecovery/synthetic_data_*.csv ./
```

### Run Analysis Locally

```bash
# Analyze the results (will auto-detect most recent file)
python analyze_parameter_recovery.py

# Or specify a specific results file
python analyze_parameter_recovery.py recovery_results_time_varying_k_combined_20241201_143022.csv

# Generate report only (no plots)
python analyze_parameter_recovery.py --no-plots

# View results without saving files
python analyze_parameter_recovery.py --no-save
```

### Analysis Output

The local analysis will generate:

- **Parameter recovery distribution plots**: Show how well each parameter was recovered
- **Correlation matrices**: Reveal parameter relationships and dependencies
- **NLL and convergence analysis**: Assess optimization performance
- **Recovery error summary**: Identify well-constrained vs. sloppy parameters
- **Comprehensive text report**: Detailed analysis with recommendations

## 💾 Incremental Saving & Progress Tracking

**NEW FEATURE**: The recovery study now saves results incrementally as each run completes, making it much more robust for long-running jobs:

### How It Works

- **Real-time Saving**: Each recovery run saves its result immediately to the CSV file
- **File Locking**: Uses proper file locking to prevent corruption during parallel writes
- **Progress Updates**: Shows "Progress: X runs completed" after each save
- **Checkpoint Backups**: Creates `recovery_results_checkpoint_10.csv`, `recovery_results_checkpoint_20.csv`, etc. every 10 runs
- **Fault Tolerance**: If the job gets interrupted, you keep all completed results

### Benefits

- ✅ **No Lost Work**: If SLURM kills your job, you keep all completed runs
- ✅ **Monitor Progress**: Check the CSV file anytime to see current progress
- ✅ **Resume Capability**: Can restart from where you left off (manually)
- ✅ **Early Analysis**: Start analyzing partial results while the job is still running

### Monitoring Your Job

```bash
# Check how many results you have so far
wc -l recovery_results.csv

# View the latest results
tail -5 recovery_results.csv

# Check for checkpoint files
ls -la *checkpoint*.csv
```

## 🛠️ Troubleshooting

### Common Issues:

1. **Job fails to start**:

   - Check SLURM parameters match your system
   - Verify partition and account names
   - Ensure resource requests are within limits

2. **Python environment issues**:

   - Verify module loading commands
   - Check conda environment activation
   - Install missing packages with pip

3. **File not found errors**:

   - Ensure all files were copied correctly
   - Check file permissions
   - Verify working directory is correct

4. **Low success rate**:
   - Increase `MAX_ITERATIONS` in job script
   - Check parameter bounds in optimization code
   - Verify synthetic data generation succeeded

### Getting Help:

- Check SLURM logs: `recovery_study_JOBID.out` and `recovery_study_JOBID.err`
- Look for error files: `recovery_error_TIMESTAMP.txt`
- Contact your ARC support team for system-specific issues

## 📋 Ground Truth Parameters

The study uses these rounded ground truth parameters:

```
n2 = 5.12          N2 = 168.84        k_max = 0.01314    tau = 47.26
r21 = 2.42         r23 = 1.46         R21 = 1.29         R23 = 2.47
burst_size = 14.28 n_inner = 20.50    alpha = 0.654
beta_k = 0.720     beta_tau = 2.90
```

These were derived from your optimized parameters and will be used to generate synthetic data and assess recovery accuracy.

## 🎯 Study Goals

This parameter recovery study will help you:

1. **Sanity Check**: Verify that your optimization pipeline can recover known parameters
2. **Parameter Identifiability**: Determine which parameters are uniquely determinable
3. **Model Sloppiness**: Identify parameters that are poorly constrained by the data
4. **Optimization Reliability**: Assess the consistency of your optimization approach

Good results typically show:

- High recovery success rate (>70%)
- Low parameter recovery error (<10%)
- Consistent parameter values across runs
- Clear identification of well-constrained vs. sloppy parameters
