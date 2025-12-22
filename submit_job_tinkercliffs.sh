#!/bin/bash
#SBATCH --account=polya
#SBATCH --partition=normal_q
#SBATCH --qos=tc_normal_base
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --constraint=amd
#SBATCH --output=tc_%j.out
#SBATCH --error=tc_%j.err

# Usage: sbatch submit_job_tinkercliffs.sh [mechanism]
# Example: sbatch submit_job_tinkercliffs.sh simple
# Example: sbatch submit_job_tinkercliffs.sh time_varying_k
# Default mechanism: simple

# Load Python environment
module reset
module load Miniforge3
source activate $HOME/.conda/envs/simulationOptimizationEnv


echo "=========================================="
echo "Simulation-based Optimization (TinkerCliffs)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "=========================================="

# Run your script with mechanism argument (force force CPU usage if needed, but script defaults to CPU if no GPU)
# Note: SimulationOptimization_join.py is the CPU version.
python SimulationOptimization_join.py ${1:-simple}

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
