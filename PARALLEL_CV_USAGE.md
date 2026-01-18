# Model Comparison Parallelization with SLURM

## Quick Start

1. **Submit job array** (runs 8 mechanisms in parallel):
   ```bash
   sbatch submit_model_comparison_cv.sh
   ```
   This returns a job ID (e.g., `123456`)

2. **Monitor progress**:
   ```bash
   squeue -u $USER
   # or
   watch squeue -u $USER
   ```

3. **Aggregate results** (after all jobs finish):
   ```bash
   python aggregate_cv_results.py <JOB_ID>
   ```
   Replace `<JOB_ID>` with the array job ID from step 1

## What Happens

### Submit Script (`submit_model_comparison_cv.sh`)
- Launches 8 independent jobs (one per mechanism)
- Each job runs on 48 CPUs with 32GB RAM
- Mechanisms tested:
  - simple
  - fixed_burst
  - feedback_onion
  - fixed_burst_feedback_onion
  - time_varying_k
  - time_varying_k_fixed_burst
  - time_varying_k_feedback_onion
  - time_varying_k_combined

- Each job calls:
  ```bash
  python model_comparison_cv_emd.py --mechanism <MECH> --run-id <JOB_ID>
  ```

- Results saved as: `cv_results_{mechanism}_{run_id}.csv`

### Aggregation Script (`aggregate_cv_results.py`)
- Finds all `cv_results_*_{run_id}.csv` files
- Loads and combines the results
- Generates:
  - `model_comparison_cv_summary_{timestamp}.csv` - Summary table
  - `model_comparison_cv_emd_{timestamp}.png` - Comparison plots

## Key Features

✅ **Safe Result Collection**: Uses `run_id` to prevent mixing with previous runs
✅ **Parallel Execution**: All 8 mechanisms run simultaneously (as resources allow)
✅ **Independent Jobs**: If one mechanism fails, others continue
✅ **Easy Monitoring**: Check individual job status with `squeue` or log files

## Log Files

- Located in `logs/`
- Format: `cv_emd_{JOB_ID}_{ARRAY_TASK_ID}.out/.err`
- Each mechanism has its own log file

## Customization

Edit `submit_model_comparison_cv.sh` to change:
- `--cpus-per-task`: Number of CPUs per mechanism
- `--mem`: Memory allocation
- `--time`: Wall time limit
- `--n-simulations`: Simulation count (in python call)
- `--k-folds`: Number of CV folds (in python call)

## Standalone Usage

You can also run a single mechanism locally:
```bash
python model_comparison_cv_emd.py --mechanism simple --run-id test01
```

Or run all mechanisms sequentially (old behavior):
```bash
python model_comparison_cv_emd.py
```
