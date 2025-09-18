#!/bin/bash
#SBATCH --job-name=param_recovery
#SBATCH --output=recovery_study_%j.out
#SBATCH --error=recovery_study_%j.err
#SBATCH --partition=PARTITION_NAME         # REPLACE: your partition (e.g., normal, compute, general)
#SBATCH --account=ACCOUNT_NAME             # REPLACE: your account name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32                 # ADJUST: number of CPU cores (16, 32, 64, etc.)
#SBATCH --mem=64G                          # ADJUST: memory (32G, 64G, 128G, etc.)
#SBATCH --time=24:00:00                    # ADJUST: time limit (12:00:00, 24:00:00, 48:00:00)

echo "Starting Parameter Recovery Study..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Load Python environment
module purge
module load PYTHON_MODULE_NAME            # REPLACE: your Python module (e.g., anaconda/2023.03)
source activate CONDA_ENV_NAME            # REPLACE: your conda environment name

echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"

# Run the parameter recovery study
python run_recovery_study.py \
    --mechanism time_varying_k_combined \
    --num_runs 80 \
    --output_file recovery_results_$(date +%Y%m%d_%H%M%S).csv \
    --synthetic_size 1000 \
    --max_iterations 200 \
    --num_simulations 400

echo "Job finished at: $(date)"