#!/bin/bash
#
# Deployment helper script for ARC Parameter Recovery Study
# This script helps you deploy the study to your ARC system
#

echo "=========================================="
echo "ARC Parameter Recovery Study Deployment"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "run_recovery_study.py" ]; then
    echo "❌ Error: This script must be run from the ARC_ParameterRecovery directory"
    echo "Please cd to the ARC_ParameterRecovery directory and run this script again."
    exit 1
fi

echo "✓ Found ARC_ParameterRecovery directory"

# Function to prompt for user input
prompt_user() {
    local prompt="$1"
    local default="$2"
    local var_name="$3"
    
    if [ -n "$default" ]; then
        read -p "$prompt [$default]: " user_input
        if [ -z "$user_input" ]; then
            user_input="$default"
        fi
    else
        read -p "$prompt: " user_input
        while [ -z "$user_input" ]; do
            echo "This field is required."
            read -p "$prompt: " user_input
        done
    fi
    
    eval "$var_name='$user_input'"
}

echo ""
echo "This script will help you configure the SLURM job script for your ARC system."
echo "Please provide the following information about your ARC system:"
echo ""

# Get ARC system information
prompt_user "ARC hostname (e.g., arc.university.edu)" "" "ARC_HOST"
prompt_user "Your username on ARC" "$USER" "ARC_USER"
prompt_user "SLURM partition name" "normal_q" "SLURM_PARTITION"
prompt_user "SLURM account name" "" "SLURM_ACCOUNT"
prompt_user "Python module name (e.g., anaconda/2022.10)" "anaconda/2022.10" "PYTHON_MODULE"
prompt_user "Conda environment name" "param_recovery" "CONDA_ENV"
prompt_user "Number of CPU cores" "32" "CPU_CORES"
prompt_user "Memory (GB)" "64" "MEMORY"
prompt_user "Time limit (hours)" "24" "TIME_HOURS"

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

read -p "Is this configuration correct? (y/n): " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "❌ Configuration cancelled. Please run this script again."
    exit 1
fi

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
NUM_RUNS=100                            # Number of recovery attempts
OUTPUT_FILE="recovery_results_\${MECHANISM}_\$(date +%Y%m%d_%H%M%S).csv"

# Ground truth optimization settings
GT_ITERATIONS=150                       # Iterations for finding ground truth
GT_SIMULATIONS=400                      # Simulations for ground truth optimization

# Synthetic data settings
SYNTHETIC_SIZE=1000                     # Synthetic data points per strain

# Recovery optimization settings
MAX_ITERATIONS=120                      # Iterations per recovery run
NUM_SIMULATIONS=250                     # Simulations per recovery evaluation

echo "Experiment Configuration:"
echo "  Mechanism: \$MECHANISM"
echo "  Recovery runs: \$NUM_RUNS"
echo "  Output file: \$OUTPUT_FILE"
echo "  Ground truth iterations: \$GT_ITERATIONS"
echo "  Ground truth simulations: \$GT_SIMULATIONS"
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
python run_recovery_study.py \\
    --mechanism \$MECHANISM \\
    --num_runs \$NUM_RUNS \\
    --output_file \$OUTPUT_FILE \\
    --gt_iterations \$GT_ITERATIONS \\
    --gt_simulations \$GT_SIMULATIONS \\
    --synthetic_size \$SYNTHETIC_SIZE \\
    --max_iterations \$MAX_ITERATIONS \\
    --num_simulations \$NUM_SIMULATIONS

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
