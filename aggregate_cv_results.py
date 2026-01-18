#!/usr/bin/env python3
"""
Aggregate Cross-Validation Results from SLURM Job Array.

This script collects individual CV result files from a model comparison
job array and generates summary tables and comparison plots.

Usage:
    python aggregate_cv_results.py <run_id>
    
Where <run_id> is the SLURM_ARRAY_JOB_ID from submit_model_comparison_cv.sh

The script will:
1. Find all cv_results_*_<run_id>.csv files
2. Load and aggregate the results
3. Generate summary table and comparison plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import sys
import os
import glob

# Optional seaborn import
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Import create functions from main script
from model_comparison_cv_emd import (
    create_comparison_plots, 
    create_summary_table,
    get_parameter_count
)


def find_cv_result_files(run_id=None):
    """
    Find all CV result files for a specific run ID.
    
    Args:
        run_id (str): Run ID to filter files (SLURM_ARRAY_JOB_ID)
    
    Returns:
        list: List of matching CSV file paths
    """
    if run_id:
        # Find files with specific run_id
        pattern = f'cv_results_*_{run_id}.csv'
    else:
        # Find all cv_results files (dangerous! may mix runs)
        pattern = 'cv_results_*.csv'
    
    files = glob.glob(pattern)
    return sorted(files)


def load_cv_results(csv_files):
    """
    Load and aggregate CV results from multiple CSV files.
    
    Args:
        csv_files (list): List of CV result CSV file paths
    
    Returns:
        list: List of result dictionaries
    """
    all_results = []
    
    for csv_file in csv_files:
        try:
            # Extract mechanism name from filename
            # Format: cv_results_{mechanism}_{run_id}.csv or cv_results_{mechanism}.csv
            basename = os.path.basename(csv_file)
            parts = basename.replace('.csv', '').split('_')
            
            # Handle both formats
            if len(parts) >= 3:
                # cv_results_{mechanism}_{run_id}.csv
                mechanism = '_'.join(parts[2:-1]) if len(parts) > 4 else parts[2]
            else:
                print(f"⚠️  Warning: Unexpected filename format: {csv_file}")
                continue
            
            # Load CSV
            df = pd.read_csv(csv_file)
            
            if len(df) == 0:
                print(f"⚠️  Warning: Empty CSV file: {csv_file}")
                continue
            
            val_emds = df['val_emd'].values
            train_emds = df['train_emd'].values
            
            result = {
                'mechanism': mechanism,
                'n_params': get_parameter_count(mechanism),
                'k_folds': len(df),
                'mean_val_emd': np.mean(val_emds),
                'std_val_emd': np.std(val_emds),
                'mean_train_emd': np.mean(train_emds),
                'std_train_emd': np.std(train_emds),
                'val_emds': val_emds.tolist(),
                'train_emds': train_emds.tolist(),
                'success': True,
                'csv_file': csv_file
            }
            
            all_results.append(result)
            print(f"✅ Loaded: {mechanism} (n={len(df)} folds, Val EMD={result['mean_val_emd']:.2f})")
            
        except Exception as e:
            print(f"❌ Error loading {csv_file}: {e}")
            continue
    
    return all_results


def main():
    """
    Main aggregation function.
    """
    print("="*80)
    print("AGGREGATING CROSS-VALIDATION RESULTS")
    print("="*80)
    
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Aggregate CV results from SLURM job array')
    parser.add_argument('run_id', type=str, nargs='?', default=None,
                       help='Run ID (SLURM_ARRAY_JOB_ID) to filter result files')
    parser.add_argument('--pattern', type=str, default=None,
                       help='Custom glob pattern for CSV files (overrides run_id)')
    
    args = parser.parse_args()
    
    # Find CV result files
    if args.pattern:
        print(f"\n📁 Using custom pattern: {args.pattern}")
        csv_files = glob.glob(args.pattern)
    elif args.run_id:
        print(f"\n📁 Finding results for Run ID: {args.run_id}")
        csv_files = find_cv_result_files(args.run_id)
    else:
        print("\n⚠️  WARNING: No run ID specified!")
        print("   This will aggregate ALL cv_results_*.csv files in the directory.")
        print("   Results from different runs may be mixed.")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        csv_files = find_cv_result_files(None)
    
    if not csv_files:
        print(f"\n❌ No CV result files found!")
        if args.run_id:
            print(f"   Searched for: cv_results_*_{args.run_id}.csv")
        print("\n💡 Make sure the SLURM job array has completed successfully.")
        return
    
    print(f"\n📊 Found {len(csv_files)} CV result files:")
    for f in csv_files:
        print(f"   - {f}")
    
    # Load and aggregate results
    print(f"\n🔄 Loading and aggregating results...")
    all_results = load_cv_results(csv_files)
    
    if not all_results:
        print(f"\n❌ No results could be loaded!")
        return
    
    print(f"\n✅ Successfully loaded {len(all_results)} mechanism results")
    
    # Generate summary table and plots
    print(f"\n📊 Generating summary and visualizations...")
    
    summary_df = create_summary_table(all_results, save_table=True)
    create_comparison_plots(all_results, save_plots=True)
    
    print(f"\n🎉 Aggregation complete!")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"📁 Results saved with timestamp: {timestamp}")
    
    if args.run_id:
        print(f"📋 Run ID: {args.run_id}")
    
    print(f"\n💡 Interpretation:")
    print(f"   - Lower Validation EMD indicates better model fit to unseen data")
    print(f"   - Use the summary table to identify the best mechanism(s)")
    print(f"   - Individual fold results are in the original CSV files")


if __name__ == "__main__":
    main()
