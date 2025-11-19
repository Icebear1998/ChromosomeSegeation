#!/bin/bash
#SBATCH --account=polya
#SBATCH --partition=normal_q
#SBATCH --qos=owl_normal_base
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# Usage: sbatch submit_job.sh [mechanism]
# Example: sbatch submit_job.sh simple
# Example: sbatch submit_job.sh time_varying_k
# Default mechanism: simple

# Load Python environment
module load Miniforge3
source activate $HOME/.conda/envs/simulationOptimizationEnv

# Get mechanism from command line (default to 'simple' if not provided)
MECHANISM=${1:-simple}

echo "=========================================="
echo "Simulation-based Optimization"
echo "Job ID: $SLURM_JOB_ID"
echo "Mechanism: $MECHANISM"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "=========================================="

# Run your script with mechanism argument
python SimulationOptimization_join.py $MECHANISM

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="