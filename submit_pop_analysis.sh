#!/bin/bash
#SBATCH --account=polya
#SBATCH --partition=normal_q
#SBATCH --qos=owl_normal_base
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --output=%j_pop_analysis.out
#SBATCH --error=%j_pop_analysis.err

# Usage: sbatch submit_pop_analysis.sh

# Load Python environment
module load Miniforge3
source activate $HOME/.conda/envs/simulationOptimizationEnv


echo "=========================================="
echo "Population Efficiency Analysis (MoM)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "=========================================="

# Run the analysis script
python AnalyzePopulationEfficiency_EMD.py

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
