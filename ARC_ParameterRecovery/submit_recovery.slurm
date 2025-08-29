#!/bin/bash
#SBATCH --job-name=param_recovery          # Job name
#SBATCH --output=recovery_study_%j.out     # Standard output log (%j expands to jobID)
#SBATCH --error=recovery_study_%j.err      # Standard error log
#SBATCH --partition=normal_q               # Partition (queue) name - MODIFY FOR YOUR SYSTEM
#SBATCH --account=your_account_name        # Account name - MODIFY FOR YOUR SYSTEM
#SBATCH --nodes=1                          # We need all processes on one node for multiprocessing
#SBATCH --ntasks=1                         # We are running one main python script
#SBATCH --cpus-per-task=32                 # Request 32 CPU cores for the multiprocessing pool
#SBATCH --mem=64G                          # Memory requirement (64 GB)
#SBATCH --time=24:00:00                    # Time limit hrs:min:sec (24 hours)

echo "=========================================="
echo "Parameter Recovery Study - ARC Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo ""

# --- 1. Set up the software environment ---
echo "Setting up software environment..."
module purge

# MODIFY THESE MODULE LOADS FOR YOUR ARC SYSTEM:
# Option 1: If using system anaconda/miniconda
module load anaconda/2022.10          # Load anaconda module (check available versions)
source activate your_conda_env_name   # Activate your conda environment

# Option 2: If using system python + pip
# module load python/3.9
# module load scipy-stack

# Option 3: If using your own conda installation
# export PATH="/path/to/your/miniconda3/bin:$PATH"
# source activate your_env_name

echo "✓ Environment setup complete"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "Available memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "Available CPUs: $SLURM_CPUS_PER_TASK"
echo ""

# --- 2. Verify required files ---
echo "Checking required files..."
required_files=(
    "run_recovery_study.py"
    "SimulationOptimization_join.py"
    "simulation_utils.py"
    "MultiMechanismSimulationTimevary.py"
    "Chromosomes_Theory.py"
    "Data/All_strains_SCStimes.xlsx"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: Required file $file not found!"
        echo "Please ensure all files are in the working directory."
        exit 1
    fi
done
echo "✓ All required files found"
echo ""

# --- 3. Define experiment parameters ---
echo "Configuring experiment parameters..."

# Main parameters - MODIFY AS NEEDED
MECHANISM="time_varying_k_combined"     # Mechanism to study
NUM_RUNS=100                            # Number of recovery attempts
OUTPUT_FILE="recovery_results_${MECHANISM}_$(date +%Y%m%d_%H%M%S).csv"

# Ground truth optimization settings
GT_ITERATIONS=150                       # Iterations for finding ground truth
GT_SIMULATIONS=400                      # Simulations for ground truth optimization

# Synthetic data settings
SYNTHETIC_SIZE=1000                     # Synthetic data points per strain

# Recovery optimization settings
MAX_ITERATIONS=120                      # Iterations per recovery run
NUM_SIMULATIONS=250                     # Simulations per recovery evaluation

echo "Experiment Configuration:"
echo "  Mechanism: $MECHANISM"
echo "  Recovery runs: $NUM_RUNS"
echo "  Output file: $OUTPUT_FILE"
echo "  Ground truth iterations: $GT_ITERATIONS"
echo "  Ground truth simulations: $GT_SIMULATIONS"
echo "  Synthetic data size: $SYNTHETIC_SIZE"
echo "  Recovery iterations: $MAX_ITERATIONS"
echo "  Recovery simulations: $NUM_SIMULATIONS"
echo ""

# --- 4. Run the parameter recovery study ---
echo "=========================================="
echo "Starting Parameter Recovery Study"
echo "=========================================="
echo "Expected runtime: 12-24 hours"
echo "Progress will be logged to recovery_study_${SLURM_JOB_ID}.out"
echo ""

# Run the main Python script with all parameters
python run_recovery_study.py \
    --mechanism $MECHANISM \
    --num_runs $NUM_RUNS \
    --output_file $OUTPUT_FILE \
    --gt_iterations $GT_ITERATIONS \
    --gt_simulations $GT_SIMULATIONS \
    --synthetic_size $SYNTHETIC_SIZE \
    --max_iterations $MAX_ITERATIONS \
    --num_simulations $NUM_SIMULATIONS

# Capture the exit status
EXIT_STATUS=$?

echo ""
echo "=========================================="
if [ $EXIT_STATUS -eq 0 ]; then
    echo "✓ PARAMETER RECOVERY STUDY COMPLETED SUCCESSFULLY"
    echo "✓ Results saved to: $OUTPUT_FILE"
    
    # List output files
    echo ""
    echo "Generated files:"
    ls -la recovery_results_*.csv synthetic_data_*.csv recovery_summary_*.txt 2>/dev/null || echo "  No output files found (check for errors)"
    
    echo ""
    echo "Next steps:"
    echo "1. Download the results CSV file to your local machine:"
    echo "   scp your_username@arc_system:$(pwd)/$OUTPUT_FILE /local/path/"
    echo "2. Run analyze_parameter_recovery.py locally for detailed analysis"
    echo "3. Review the recovery summary file for quick insights"
    
else
    echo "✗ PARAMETER RECOVERY STUDY FAILED"
    echo "✗ Exit status: $EXIT_STATUS"
    echo "✗ Check the error log: recovery_study_${SLURM_JOB_ID}.err"
    
    # Try to find error files
    echo ""
    echo "Error files (if any):"
    ls -la recovery_error_*.txt 2>/dev/null || echo "  No error files found"
fi

echo "=========================================="
echo "Job finished at: $(date)"
echo "Total job runtime: $(( $(date +%s) - $(date -d "$SLURM_JOB_START_TIME" +%s) )) seconds"
echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"

# Exit with the same status as the Python script
exit $EXIT_STATUS
