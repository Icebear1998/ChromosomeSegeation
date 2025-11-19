#!/bin/bash
#SBATCH --account=polya
#SBATCH --partition=normal_q
#SBATCH --qos=owl_normal_base
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# Load Python environment
module load Miniforge3
source activate $HOME/.conda/envs/simulationOptimizationEnv

# Run your script
python SimulationOptimization_nontimevarying.py