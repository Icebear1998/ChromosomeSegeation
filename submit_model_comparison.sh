#!/bin/bash
#SBATCH --account=polya
#SBATCH --partition=normal_q
#SBATCH --qos=owl_normal_base
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --output=model_comparison_%j.out
#SBATCH --error=model_comparison_%j.err
#SBATCH --job-name=model_comparison_aic_bic

# Job description
echo "=========================================="
echo "AIC/BIC Model Comparison Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "=========================================="

# Load Python environment
module load Miniforge3
source activate $HOME/.conda/envs/simulationOptimizationEnv

# Set environment variables for optimization
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print system info
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "Available memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "CPU info: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"

# Create output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="model_comparison_results_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

# Run the model comparison analysis
echo "Starting model comparison analysis..."
echo "Analyzing 8 mechanisms across 5 datasets (including Velcade)"
echo "Mechanisms: simple, fixed_burst, feedback_onion, fixed_burst_feedback_onion,"
echo "           time_varying_k, time_varying_k_fixed_burst, time_varying_k_feedback_onion, time_varying_k_combined"
echo "Datasets: wildtype, threshold, degrade, degradeAPC, velcade"
echo "Parallel processing: ENABLED (utilizing $SLURM_CPUS_PER_TASK CPUs)"
echo "Configuration: 10 runs per mechanism, 500 simulations per evaluation"

# Run with output redirection to capture all results
python model_comparison_aic_bic.py 2>&1 | tee ${OUTPUT_DIR}/model_comparison_log.txt

# Check if the analysis completed successfully
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Model comparison analysis completed successfully!"
    
    # Move generated files to output directory
    echo "Moving output files to $OUTPUT_DIR..."
    mv model_comparison_results_*.csv $OUTPUT_DIR/ 2>/dev/null || echo "No CSV results to move"
    mv model_comparison_summary_*.txt $OUTPUT_DIR/ 2>/dev/null || echo "No summary files to move"
    mv aic_bic_comparison_*.png $OUTPUT_DIR/ 2>/dev/null || echo "No plots to move"
    
    # Create a summary of what was generated
    echo "Files generated:" > ${OUTPUT_DIR}/file_summary.txt
    ls -la $OUTPUT_DIR/ >> ${OUTPUT_DIR}/file_summary.txt
    
    echo "Results saved in: $OUTPUT_DIR"
    echo "Log file: ${OUTPUT_DIR}/model_comparison_log.txt"
    
else
    echo "=========================================="
    echo "ERROR: Model comparison analysis failed!"
    echo "Check the error log: model_comparison_${SLURM_JOB_ID}.err"
    echo "Check the output log: ${OUTPUT_DIR}/model_comparison_log.txt"
    exit 1
fi

echo "=========================================="
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "=========================================="
