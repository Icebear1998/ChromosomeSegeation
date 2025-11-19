#!/bin/bash
#SBATCH --account=polya
#SBATCH --partition=normal_q
#SBATCH --qos=owl_normal_base
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --output=test_fixed_burst_%j.out
#SBATCH --error=test_fixed_burst_%j.err
#SBATCH --job-name=test_fixed_burst

# Script to test fixed_burst mechanism with MoM optimization
# Uses Excel data from All_strains_SCStimes.xlsx
# Tests Option 2 (Fractional Burst) implementation

echo "======================================================================"
echo "Testing Fixed_Burst Mechanism with Fractional Burst (Option 2)"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "======================================================================"
echo ""

# Load Python environment
module load Miniforge3
source activate $HOME/.conda/envs/simulationOptimizationEnv

# Check if environment loaded correctly
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment"
    exit 1
fi

echo "Python version:"
python --version
echo ""

echo "Checking required packages..."
python -c "import numpy; import pandas; import scipy; import matplotlib; print('All packages available')"
echo ""

# Change to working directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"
echo ""

# Check if data file exists
if [ ! -f "Data/All_strains_SCStimes.xlsx" ]; then
    echo "ERROR: Data file not found: Data/All_strains_SCStimes.xlsx"
    exit 1
fi

echo "Data file found: Data/All_strains_SCStimes.xlsx"
echo ""

# Run the test script
echo "======================================================================"
echo "Running test_simulation_fixed_burst_excel.py"
echo "======================================================================"
echo ""

python test_simulation_fixed_burst_excel.py 2>&1 | tee test_fixed_burst_${SLURM_JOB_ID}.log

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✅ Test completed successfully!"
    echo "======================================================================"
    echo "End time: $(date)"
    echo ""
    echo "Output files:"
    echo "  - Log: test_fixed_burst_${SLURM_JOB_ID}.log"
    echo "  - Plot: fixed_burst_mom_excel_data.png"
    echo "  - SLURM out: test_fixed_burst_${SLURM_JOB_ID}.out"
    echo "  - SLURM err: test_fixed_burst_${SLURM_JOB_ID}.err"
    echo ""
else
    echo ""
    echo "======================================================================"
    echo "❌ Test failed with error code: $?"
    echo "======================================================================"
    echo "End time: $(date)"
    echo "Check the log files for details."
    echo ""
    exit 1
fi

