#!/bin/bash
#SBATCH --account=polya
#SBATCH --partition=normal_q
#SBATCH --qos=owl_normal_base
#SBATCH --job-name=cv_emd_comparison
#SBATCH --array=0-3
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --output=logs/cv_emd_%A_%a.out
#SBATCH --error=logs/cv_emd_%A_%a.err

# SLURM Job Array for Model Comparison using EMD Cross-Validation
# Each array task runs one mechanism independently
# 
# Key SLURM Parameters Explained:
# --array=0-7         : Creates 8 parallel jobs (indices 0 through 7)
# --cpus-per-task=96  : Each job gets 96 CPUs for parallel optimization
# Lines 23-30 below   : Define which mechanism each array index runs

# Create logs directory if it doesn't exist
mkdir -p logs

# Define all mechanisms to test (THIS IS WHERE ALL 8 MECHANISMS ARE SPECIFIED)
# The array index ($SLURM_ARRAY_TASK_ID) selects which mechanism this job runs
mechanisms=(
   "time_varying_k_combined"         # Array index 7
)

# Get mechanism for this specific array task
mechanism=${mechanisms[$SLURM_ARRAY_TASK_ID]}

# Generate run ID from SLURM job ID (to group all mechanisms from this run)
RUN_ID="${SLURM_ARRAY_JOB_ID}"

echo "================================================================"
echo "SLURM Job Array: Model Comparison CV with EMD"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Mechanism: $mechanism"
echo "Run ID: $RUN_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Started at: $(date)"
echo "================================================================"

# Load Python environment (VT ARC specific)
module load Miniforge3
source activate $HOME/.conda/envs/simulationOptimizationEnv

# Set environment variables for optimization
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print environment info
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "================================================================"

# Run cross-validation for single mechanism
python model_comparison_cv_emd.py \
    --mechanism "$mechanism" \
    --run-id "$RUN_ID" \
    --k-folds 5 \
    --n-simulations 2000 \
    --max-iter 1000 \
    --tol 0.01

# Record completion
echo "================================================================"
echo "Completed at: $(date)"
echo "Exit code: $?"
echo "Total runtime: $SECONDS seconds"
echo "================================================================"
