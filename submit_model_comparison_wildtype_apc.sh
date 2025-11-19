#!/bin/bash
#SBATCH --account=polya
#SBATCH --partition=normal_q
#SBATCH --qos=owl_normal_base
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=16G
#SBATCH --output=model_comparison_wildtype_apc_%j.out
#SBATCH --error=model_comparison_wildtype_apc_%j.err
#SBATCH --job-name=model_comparison_wt_apc

# Job description
echo "=========================================="
echo "AIC/BIC Model Comparison Analysis"
echo "WILDTYPE + APC DATA ONLY VERSION"
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
OUTPUT_DIR="model_comparison_wildtype_apc_results_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

# Run the model comparison analysis
echo "Starting wildtype + APC model comparison analysis..."
echo "Analyzing 8 mechanisms across 2 datasets (wildtype + degradeAPC only)"
echo "Mechanisms: simple, fixed_burst, feedback_onion, fixed_burst_feedback_onion,"
echo "           time_varying_k, time_varying_k_fixed_burst, time_varying_k_feedback_onion, time_varying_k_combined"
echo "Datasets: wildtype, degradeAPC (focused analysis for faster comparison)"
echo "Parallel processing: ENABLED (utilizing $SLURM_CPUS_PER_TASK CPUs)"
echo "Configuration: 5 runs per mechanism, 400 simulations per evaluation"
echo "Expected runtime: ~6-12 hours (faster than full analysis)"

# Run with output redirection to capture all results
python model_comparison_wildtype_apc.py 2>&1 | tee ${OUTPUT_DIR}/model_comparison_wildtype_apc_log.txt

# Check if the analysis completed successfully
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Wildtype + APC model comparison analysis completed successfully!"
    
    # Move generated files to output directory
    echo "Moving output files to $OUTPUT_DIR..."
    mv model_comparison_wildtype_apc_*.csv $OUTPUT_DIR/ 2>/dev/null || echo "No CSV results to move"
    mv model_comparison_wildtype_apc_*.png $OUTPUT_DIR/ 2>/dev/null || echo "No plots to move"
    mv *wildtype_apc*.txt $OUTPUT_DIR/ 2>/dev/null || echo "No additional text files to move"
    
    # Create a summary of what was generated
    echo "Files generated:" > ${OUTPUT_DIR}/file_summary.txt
    ls -la $OUTPUT_DIR/ >> ${OUTPUT_DIR}/file_summary.txt
    
    # Calculate data points analyzed
    echo "" >> ${OUTPUT_DIR}/file_summary.txt
    echo "Analysis Summary:" >> ${OUTPUT_DIR}/file_summary.txt
    echo "- Datasets analyzed: wildtype, degradeAPC" >> ${OUTPUT_DIR}/file_summary.txt
    echo "- Mechanisms compared: 8 total" >> ${OUTPUT_DIR}/file_summary.txt
    echo "- MoM-based: simple, fixed_burst, feedback_onion, fixed_burst_feedback_onion" >> ${OUTPUT_DIR}/file_summary.txt
    echo "- Simulation-based: time_varying_k, time_varying_k_fixed_burst, time_varying_k_feedback_onion, time_varying_k_combined" >> ${OUTPUT_DIR}/file_summary.txt
    echo "- Optimization runs per mechanism: 5" >> ${OUTPUT_DIR}/file_summary.txt
    echo "- Simulations per evaluation: 400" >> ${OUTPUT_DIR}/file_summary.txt
    echo "- Total CPUs utilized: $SLURM_CPUS_PER_TASK" >> ${OUTPUT_DIR}/file_summary.txt
    echo "- Analysis focus: Faster comparison using core datasets only" >> ${OUTPUT_DIR}/file_summary.txt
    
    echo "Results saved in: $OUTPUT_DIR"
    echo "Log file: ${OUTPUT_DIR}/model_comparison_wildtype_apc_log.txt"
    
    # Display quick summary if CSV file exists
    if ls ${OUTPUT_DIR}/model_comparison_wildtype_apc_*.csv 1> /dev/null 2>&1; then
        echo ""
        echo "Quick Results Summary:"
        echo "======================"
        CSV_FILE=$(ls ${OUTPUT_DIR}/model_comparison_wildtype_apc_*.csv | head -1)
        echo "Best models by AIC/BIC (from $CSV_FILE):"
        # Show header and first few rows
        head -4 "$CSV_FILE" | column -t -s','
        echo "..."
        echo "(See full results in CSV file)"
    fi
    
else
    echo "=========================================="
    echo "ERROR: Wildtype + APC model comparison analysis failed!"
    echo "Check the error log: model_comparison_wildtype_apc_${SLURM_JOB_ID}.err"
    echo "Check the output log: ${OUTPUT_DIR}/model_comparison_wildtype_apc_log.txt"
    exit 1
fi

echo "=========================================="
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "Average time per mechanism: $((SECONDS / 8)) seconds"
echo "Efficiency: Focused analysis on wildtype + APC data only"
echo "=========================================="
