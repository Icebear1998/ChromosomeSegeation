#!/usr/bin/env python3
"""
Extract and summarize tau values from cross-validation results.

This script analyzes the optimal tau values from time-varying k mechanisms
across all cross-validation folds.
"""

import numpy as np
import pandas as pd
import json
import glob
import os
from datetime import datetime

# Import parameter names function
from simulation_utils import get_parameter_names


def extract_tau_from_cv_results(run_id=None):
    """
    Extract tau values from CV result files for time-varying k mechanisms.
    
    Args:
        run_id (str): Run ID to filter files (optional)
    
    Returns:
        dict: Dictionary with mechanism names as keys and tau arrays as values
    """
    # Time-varying k mechanisms that have tau parameter
    time_varying_mechanisms = [
        'time_varying_k',
        'time_varying_k_fixed_burst',
        'time_varying_k_feedback_onion',
        'time_varying_k_combined'
    ]
    
    tau_data = {}
    
    for mechanism in time_varying_mechanisms:
        # Find the CSV file for this mechanism
        if run_id:
            pattern = f'cv_results_{mechanism}_{run_id}.csv'
        else:
            pattern = f'cv_results_{mechanism}_*.csv'
        
        files = glob.glob(pattern)
        
        if not files:
            print(f"⚠️  Warning: No CV results found for {mechanism}")
            continue
        
        # Use the most recent file if multiple exist
        csv_file = sorted(files)[-1]
        
        print(f"📂 Reading: {csv_file}")
        
        # Read the CSV
        df = pd.read_csv(csv_file)
        
        # Get parameter names for this mechanism
        param_names = get_parameter_names(mechanism)
        
        # Find tau index
        tau_index = param_names.index('tau')
        
        # Extract tau values from each fold
        tau_values = []
        for _, row in df.iterrows():
            params_str = row['params']
            params_list = json.loads(params_str)
            tau = params_list[tau_index]
            tau_values.append(tau)
        
        tau_data[mechanism] = np.array(tau_values)
        print(f"   ✅ Extracted {len(tau_values)} tau values")
    
    return tau_data


def create_tau_summary_table(tau_data, save_table=True):
    """
    Create and display a summary table of tau values.
    
    Args:
        tau_data (dict): Dictionary with mechanism names as keys and tau arrays as values
        save_table (bool): Whether to save table to CSV
    
    Returns:
        pd.DataFrame: Summary table
    """
    summary_rows = []
    
    for mechanism, tau_values in tau_data.items():
        # Calculate statistics
        mean_tau = np.mean(tau_values)
        std_tau = np.std(tau_values)
        min_tau = np.min(tau_values)
        max_tau = np.max(tau_values)
        
        summary_rows.append({
            'Mechanism': mechanism,
            'Mean Tau (min)': f'{mean_tau:.2f}',
            'Std Tau (min)': f'{std_tau:.2f}',
            'Min Tau (min)': f'{min_tau:.2f}',
            'Max Tau (min)': f'{max_tau:.2f}',
            'Range (min)': f'{max_tau - min_tau:.2f}',
            'CV (%)': f'{(std_tau/mean_tau)*100:.1f}',
            'Tau Values': ', '.join([f'{t:.2f}' for t in tau_values])
        })
    
    df = pd.DataFrame(summary_rows)
    
    print(f"\n{'='*100}")
    print("TAU VALUE SUMMARY - TIME-VARYING K MECHANISMS")
    print(f"{'='*100}")
    print(df.to_string(index=False))
    
    # Calculate grand statistics across all mechanisms
    all_tau_values = np.concatenate(list(tau_data.values()))
    grand_mean = np.mean(all_tau_values)
    grand_std = np.std(all_tau_values)
    grand_min = np.min(all_tau_values)
    grand_max = np.max(all_tau_values)
    
    print(f"\n{'='*100}")
    print("GRAND STATISTICS (ALL TIME-VARYING K MECHANISMS)")
    print(f"{'='*100}")
    print(f"Grand Mean: {grand_mean:.2f} min")
    print(f"Grand Std: {grand_std:.2f} min")
    print(f"Grand Range: {grand_min:.2f} - {grand_max:.2f} min")
    print(f"Total Samples: {len(all_tau_values)}")
    
    if save_table:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'tau_summary_{timestamp}.csv'
        df.to_csv(filename, index=False)
        print(f"\n💾 Summary table saved as: {filename}")
    
    return df


def main():
    """
    Main function to extract and summarize tau values.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract and summarize tau values from CV results')
    parser.add_argument('--run-id', type=str, default=None,
                       help='Run ID to filter result files (optional)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("TAU VALUE EXTRACTION AND SUMMARY")
    print("="*80)
    
    if args.run_id:
        print(f"Run ID: {args.run_id}")
    else:
        print("Run ID: Not specified (will use most recent files)")
    
    # Extract tau values
    print(f"\n📊 Extracting tau values from CV results...")
    tau_data = extract_tau_from_cv_results(args.run_id)
    
    if not tau_data:
        print("\n❌ No tau data found! Make sure CV result files exist.")
        return
    
    # Create summary table
    print(f"\n📊 Creating summary table...")
    summary_df = create_tau_summary_table(tau_data, save_table=True)
    
    print(f"\n🎉 Tau analysis complete!")
    print(f"\n💡 Interpretation:")
    print(f"   - Tau represents the characteristic time for degradation rate change")
    print(f"   - Consistency across mechanisms suggests robust time-varying dynamics")
    print(f"   - CV (%) shows relative variability across folds")


if __name__ == "__main__":
    main()
