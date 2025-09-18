#!/bin/bash
#
# Deployment script for ARC Parameter Recovery Study
#

echo "=========================================="
echo "ARC Parameter Recovery Study Deployment"
echo "=========================================="
echo ""

# --- Define experiment parameters ---
echo "Configuring experiment parameters..."

# Main parameters
export MECHANISM="time_varying_k_combined"     # Mechanism to study
export NUM_RUNS=60                            # Number of recovery attempts (default in script)
export OUTPUT_FILE="recovery_results_${MECHANISM}_$(date +%Y%m%d_%H%M%S).csv"

# Note: Ground truth parameters are now hard-coded in the get_ground_truth_params function
# No need for GT_ITERATIONS and GT_SIMULATIONS anymore

# Synthetic data settings
export SYNTHETIC_SIZE=1000                    # Synthetic data points per strain (default in script)

# Recovery optimization settings
export MAX_ITERATIONS=200                     # Iterations per recovery run (default in script)
export NUM_SIMULATIONS=300                    # Simulations per recovery evaluation (default in script)

# Check if we're in the right directory
if [ ! -f "run_recovery_study.py" ]; then
    echo "❌ Error: This script must be run from the ARC_ParameterRecovery directory"
    echo "Please cd to the ARC_ParameterRecovery directory and run this script again."
    exit 1
fi

echo "✓ Found ARC_ParameterRecovery directory"

# Pre-configured ARC system information
ARC_HOST="owl1.arc.vt.edu"
ARC_USER="kientp"
SLURM_PARTITION="normal_q"
SLURM_ACCOUNT="polya"
PYTHON_MODULE="Miniforge3"
CONDA_ENV="simulationOptimizationEnv"
CPU_CORES="48"
MEMORY="64"
TIME_HOURS="12"

echo ""
echo "Configuration Summary:"
echo "  ARC Host: $ARC_HOST"
echo "  Username: $ARC_USER"
echo "  Partition: $SLURM_PARTITION"
echo "  Account: $SLURM_ACCOUNT"
echo "  Python Module: $PYTHON_MODULE"
echo "  Conda Environment: $CONDA_ENV"
echo "  CPU Cores: $CPU_CORES"
echo "  Memory: ${MEMORY}G"
echo "  Time Limit: ${TIME_HOURS}:00:00"
echo ""

echo ""
echo "Updating SLURM job script..."

# Create a backup of the original
cp submit_recovery.slurm submit_recovery.slurm.backup

# Update the SLURM script
cat > submit_recovery.slurm << EOF
#!/bin/bash
#SBATCH --job-name=param_recovery          # Job name
#SBATCH --output=recovery_study_%j.out     # Standard output log (%j expands to jobID)
#SBATCH --error=recovery_study_%j.err      # Standard error log
#SBATCH --partition=$SLURM_PARTITION       # Partition (queue) name
#SBATCH --account=$SLURM_ACCOUNT           # Account name
#SBATCH --nodes=1                          # We need all processes on one node for multiprocessing
#SBATCH --ntasks=1                         # We are running one main python script
#SBATCH --cpus-per-task=$CPU_CORES         # Request CPU cores for the multiprocessing pool
#SBATCH --mem=${MEMORY}G                   # Memory requirement
#SBATCH --time=${TIME_HOURS}:00:00         # Time limit hrs:min:sec

echo "=========================================="
echo "Parameter Recovery Study - ARC Job"
echo "=========================================="
echo "Job ID: \$SLURM_JOB_ID"
echo "Job Name: \$SLURM_JOB_NAME"
echo "Node: \$SLURM_NODELIST"
echo "CPUs: \$SLURM_CPUS_PER_TASK"
echo "Memory: \$SLURM_MEM_PER_NODE MB"
echo "Partition: \$SLURM_JOB_PARTITION"
echo "Account: \$SLURM_JOB_ACCOUNT"
echo "Start time: \$(date)"
echo "Working directory: \$(pwd)"
echo ""

# --- 1. Set up the software environment ---
echo "Setting up software environment..."
module purge
module load $PYTHON_MODULE
conda activate $CONDA_ENV

echo "✓ Environment setup complete"
echo "Python version: \$(python --version)"
echo "Python path: \$(which python)"
echo "Available memory: \$(free -h | grep '^Mem:' | awk '{print \$2}')"
echo "Available CPUs: \$SLURM_CPUS_PER_TASK"
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

for file in "\${required_files[@]}"; do
    if [ ! -f "\$file" ]; then
        echo "ERROR: Required file \$file not found!"
        echo "Please ensure all files are in the working directory."
        exit 1
    fi
done
echo "✓ All required files found"
echo ""

# --- 3. Define experiment parameters ---
echo "Configuring experiment parameters..."

# Main parameters
MECHANISM="time_varying_k_combined"     # Mechanism to study
NUM_RUNS=60                            # Number of recovery attempts
OUTPUT_FILE="recovery_results_\${MECHANISM}_\$(date +%Y%m%d_%H%M%S).csv"

# Synthetic data settings
SYNTHETIC_SIZE=1000                    # Synthetic data points per strain (default in script)

# Recovery optimization settings
MAX_ITERATIONS=100                     # Iterations per recovery run (default in script)
NUM_SIMULATIONS=200                    # Simulations per recovery evaluation (default in script)

echo "Experiment Configuration:"
echo "  Mechanism: \$MECHANISM"
echo "  Recovery runs: \$NUM_RUNS"
echo "  Output file: \$OUTPUT_FILE"
# Ground truth parameters are hard-coded in get_ground_truth_params function
echo "  Synthetic data size: \$SYNTHETIC_SIZE"
echo "  Recovery iterations: \$MAX_ITERATIONS"
echo "  Recovery simulations: \$NUM_SIMULATIONS"
echo ""

# --- 4. Run the parameter recovery study ---
echo "=========================================="
echo "Starting Parameter Recovery Study"
echo "=========================================="
echo "Expected runtime: 12-24 hours"
echo "Progress will be logged to recovery_study_\${SLURM_JOB_ID}.out"
echo ""

# Run the main Python script with all parameters
python run_recovery_study.py --mechanism "time_varying_k_combined" --num_runs 80 --output_file \$OUTPUT_FILE --synthetic_size 300 --max_iterations 200 --num_simulations 300

# Capture the exit status
EXIT_STATUS=\$?

echo ""
echo "=========================================="
if [ \$EXIT_STATUS -eq 0 ]; then
    echo "✓ PARAMETER RECOVERY STUDY COMPLETED SUCCESSFULLY"
    echo "✓ Results saved to: \$OUTPUT_FILE"
    
    # List output files
    echo ""
    echo "Generated files:"
    ls -la recovery_results_*.csv synthetic_data_*.csv recovery_summary_*.txt 2>/dev/null || echo "  No output files found (check for errors)"
    
    echo ""
    echo "Next steps:"
    echo "1. Download the results CSV file to your local machine:"
    echo "   scp $ARC_USER@$ARC_HOST:\$(pwd)/\$OUTPUT_FILE /local/path/"
    echo "2. Run analyze_parameter_recovery.py locally for detailed analysis"
    echo "3. Review the recovery summary file for quick insights"
    
else
    echo "✗ PARAMETER RECOVERY STUDY FAILED"
    echo "✗ Exit status: \$EXIT_STATUS"
    echo "✗ Check the error log: recovery_study_\${SLURM_JOB_ID}.err"
    
    # Try to find error files
    echo ""
    echo "Error files (if any):"
    ls -la recovery_error_*.txt 2>/dev/null || echo "  No error files found"
fi

echo "=========================================="
echo "Job finished at: \$(date)"
echo "Total job runtime: \$(( \$(date +%s) - \$(date -d "\$SLURM_JOB_START_TIME" +%s) )) seconds"
echo "Node: \$SLURM_NODELIST"
echo "Job ID: \$SLURM_JOB_ID"

# Exit with the same status as the Python script
exit \$EXIT_STATUS
EOF

echo "✓ SLURM job script updated successfully!"
echo ""

# Create deployment instructions
cat > DEPLOYMENT_INSTRUCTIONS.txt << EOF
ARC Parameter Recovery Study - Deployment Instructions
====================================================

Generated on: $(date)
ARC System: $ARC_HOST
Username: $ARC_USER

STEP 1: Transfer Files to ARC
-----------------------------
From your local machine, run:

scp -r ARC_ParameterRecovery $ARC_USER@$ARC_HOST:~/

STEP 2: Set Up Environment on ARC
---------------------------------
Log in to ARC and set up the environment:

ssh $ARC_USER@$ARC_HOST
cd ARC_ParameterRecovery
module load $PYTHON_MODULE
conda create --name $CONDA_ENV python=3.9 -y
conda activate $CONDA_ENV
pip install -r requirements.txt

STEP 3: Test the Setup (Optional but Recommended)
------------------------------------------------
Run the local test to verify everything works:

python test_local.py

STEP 4: Submit the Job
---------------------
Submit the parameter recovery study:

sbatch submit_recovery.slurm

STEP 5: Monitor the Job
----------------------
Check job status:

squeue -u $ARC_USER

Monitor progress (optional):

tail -f recovery_study_JOBID.out

STEP 6: Retrieve Results
-----------------------
After completion, download results to your local machine:

scp $ARC_USER@$ARC_HOST:~/ARC_ParameterRecovery/recovery_results_*.csv ./
scp $ARC_USER@$ARC_HOST:~/ARC_ParameterRecovery/recovery_summary_*.txt ./

Configuration Used:
------------------
Partition: $SLURM_PARTITION
Account: $SLURM_ACCOUNT  
CPUs: $CPU_CORES
Memory: ${MEMORY}G
Time Limit: ${TIME_HOURS} hours
Python Module: $PYTHON_MODULE
Conda Environment: $CONDA_ENV

Expected Runtime: 12-24 hours
Expected Output: ~100 parameter recovery attempts
EOF

echo "✓ Deployment instructions created: DEPLOYMENT_INSTRUCTIONS.txt"
echo ""
echo "=========================================="
echo "✅ DEPLOYMENT CONFIGURATION COMPLETE!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review the updated submit_recovery.slurm file"
echo "2. Follow the instructions in DEPLOYMENT_INSTRUCTIONS.txt"
echo "3. Test locally first with: python test_local.py"
echo "4. Transfer files to ARC and submit the job"
echo ""
echo "Files ready for deployment:"
ls -la | grep -E '\.(py|slurm|txt|xlsx)$'
echo ""
echo "Good luck with your parameter recovery study! 🚀"
