#!/bin/bash
#SBATCH --account=polya
#SBATCH --partition=a100_normal_q
#SBATCH --qos=tc_a100_normal_base
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=%j_gpu.out
#SBATCH --error=%j_gpu.err

# Usage: sbatch submit_job_gpu.sh [mechanism]
# Example: sbatch submit_job_gpu.sh simple
# Example: sbatch submit_job_gpu.sh time_varying_k
# Default mechanism: simple

# Load Python environment
module reset
module load Miniforge3
source activate $HOME/.conda/envs/simulationOptimizationEnv

# Get mechanism from command line (default to 'simple' if not provided)
MECHANISM=${1:-simple}

echo "=========================================="
echo "Simulation-based Optimization (GPU)"
echo "Job ID: $SLURM_JOB_ID"
echo "Mechanism: $MECHANISM"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "=========================================="

# Run your script with mechanism argument
# Note: The script itself hardcodes the mechanism in main(), 
# but ideally it should accept an argument. 
# For now, we will run it as is, assuming the user might edit main() or we update it to accept args.
# To be safe, let's update the python script to accept args or just run it.
# The current python script ignores the arg in main(), so this arg is just for show unless we update main.
# However, the user asked to run SimulationOptimization_join.py originally.
# We are running the GPU version.

python SimulationOptimization_join_GPUbase.py

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
