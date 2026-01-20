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
from simulation_utils import get_parameter_names, get_parameter_bounds
import json


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
            
            # Parse parameter data if available
            params_list = []
            if 'params' in df.columns:
                for params_str in df['params']:
                    try:
                        params_list.append(json.loads(params_str))
                    except:
                        params_list.append(None)
            
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
                'params': params_list if params_list else None,
                'success': True,
                'csv_file': csv_file
            }
            
            all_results.append(result)
            print(f"✅ Loaded: {mechanism} (n={len(df)} folds, Val EMD={result['mean_val_emd']:.2f})")
            
        except Exception as e:
            print(f"❌ Error loading {csv_file}: {e}")
            continue
    
    return all_results


def plot_parameter_distributions(mechanism_results, run_id=None, save_plots=True):
    """
    Create parameter distribution plots for a mechanism, similar to stability plots.
    Shows box plots with overlaid colored data points.
    
    Args:
        mechanism_results (dict): Results dictionary for a single mechanism
        run_id (str): Run ID for filename (optional)
        save_plots (bool): Whether to save the plot
    """
    mechanism = mechanism_results['mechanism']
    params = mechanism_results['params']
    
    if not params or all(p is None for p in params):
        print(f"⚠️  No parameter data available for {mechanism}")
        return
    
    # Get parameter names
    try:
        param_names = get_parameter_names(mechanism)
    except:
        print(f"⚠️  Could not get parameter names for {mechanism}")
        return
    
    # Convert params list to numpy array
    params_array = np.array([p for p in params if p is not None])
    n_params = len(param_names)
    n_runs = len(params_array)
    
    # Calculate CV percentage
    cv_percentages = []
    for i in range(n_params):
        param_values = params_array[:, i]
        mean_val = np.mean(param_values)
        std_val = np.std(param_values)
        cv = (std_val / mean_val * 100) if mean_val != 0 else 0
        cv_percentages.append(cv)
    
    # Create subplots - determine grid size
    n_cols = 4
    n_rows = int(np.ceil(n_params / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    fig.suptitle(f'Parameter Stability: {mechanism} ({n_runs} runs)', fontsize=18, fontweight='bold')
    
    # Flatten axes for easy iteration
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    # Get parameter bounds for this mechanism
    try:
        param_bounds = get_parameter_bounds(mechanism)
    except:
        param_bounds = None
    
    # Create color palette for different runs
    colors = plt.cm.tab10(np.linspace(0, 1, n_runs))
    
    # Plot each parameter
    for i in range(n_params):
        ax = axes[i]
        param_values = params_array[:, i]
        
        # Create box plot
        bp = ax.boxplot([param_values], positions=[0], widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='darkblue', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))
        
        # Overlay individual points with jitter - each run gets its own color
        x_jitter = np.random.normal(0, 0.04, size=len(param_values))
        for j, (x, y, color) in enumerate(zip(x_jitter, param_values, colors)):
            ax.scatter(x, y, alpha=0.8, s=100, color=color, edgecolors='black', linewidths=1.2, zorder=3)
        
        # Set title and labels with larger fonts
        ax.set_title(param_names[i], fontsize=14, fontweight='bold')
        ax.set_ylabel(param_names[i], fontsize=12)
        ax.set_xlabel(f'CV: {cv_percentages[i]:.1f}%', fontsize=11)
        ax.set_xticks([])
        ax.tick_params(axis='y', labelsize=11)
        
        # Set y-axis to show full parameter bounds if available, otherwise data range
        if param_bounds and i < len(param_bounds):
            min_bound, max_bound = param_bounds[i]
            ax.set_ylim(min_bound, max_bound)
        else:
            # Fallback to data range with padding
            y_range = param_values.max() - param_values.min()
            if y_range > 0:
                ax.set_ylim(param_values.min() - 0.1 * y_range, param_values.max() + 0.1 * y_range)
    
    # Hide unused subplots
    for i in range(n_params, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_plots:
        if run_id:
            filename = f'parameter_distributions_{mechanism}_{run_id}.png'
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'parameter_distributions_{mechanism}_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   📊 Saved: {filename}")
    
    plt.close()


def create_summary_table_with_runid(all_results, run_id=None, save_table=True):
    """
    Wrapper for create_summary_table that uses run_id for filename.
    """
    # Call original function without saving
    df_summary = create_summary_table(all_results, save_table=False)
    
    # Save with run_id if requested
    if save_table:
        if run_id:
            filename = f'model_comparison_cv_summary_{run_id}.csv'
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'model_comparison_cv_summary_{timestamp}.csv'
        df_summary.to_csv(filename, index=False)
        print(f"\n💾 Summary table saved as: {filename}")
    
    return df_summary


def create_comparison_plots_with_runid(all_results, run_id=None, save_plots=True):
    """
    Wrapper for create_comparison_plots that uses run_id for filename.
    """
    # Filter out failed mechanisms
    successful_results = [r for r in all_results if r['success']]
    
    if not successful_results:
        print("No successful results to plot!")
        return
    
    # Define mechanism ordering
    mechanism_order = [
        'simple',
        'fixed_burst',
        'feedback_onion', 
        'time_varying_k',
        'fixed_burst_feedback_onion',
        'time_varying_k_fixed_burst',
        'time_varying_k_feedback_onion',
        'time_varying_k_combined'
    ]
    
    # Create DataFrame for plotting
    plot_data = []
    for result in successful_results:
        mechanism = result['mechanism']
        for val_emd in result['val_emds']:
            plot_data.append({
                'mechanism': mechanism,
                'Validation EMD': val_emd
            })
    
    df = pd.DataFrame(plot_data)
    
    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Comparison: Cross-Validation with EMD', fontsize=16, y=0.98)
    
    # 1. Mean Validation EMD comparison (bar plot with error bars)
    mean_data = pd.DataFrame(successful_results)
    mean_data['order'] = mean_data['mechanism'].map({m: i for i, m in enumerate(mechanism_order)})
    mean_data = mean_data.sort_values('order')
    
    bars = ax1.bar(range(len(mean_data)), mean_data['mean_val_emd'], 
                   yerr=mean_data['std_val_emd'], capsize=5, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Mechanism', fontsize=12)
    ax1.set_ylabel('Validation EMD (minutes)', fontsize=12)
    ax1.set_title('Mean Validation EMD Comparison (Lower is Better)', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(mean_data)))
    ax1.set_xticklabels(mean_data['mechanism'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, val, std) in enumerate(zip(bars, mean_data['mean_val_emd'], mean_data['std_val_emd'])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 2,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Validation EMD distribution boxplot
    if len(df) > 0:
        ordered_mechanisms = [m for m in mechanism_order if m in df['mechanism'].unique()]
        emd_data = [df[df['mechanism'] == mech]['Validation EMD'].values for mech in ordered_mechanisms]
        
        if HAS_SEABORN:
            df['mechanism'] = pd.Categorical(df['mechanism'], categories=mechanism_order, ordered=True)
            df_sorted = df.sort_values('mechanism')
            sns.boxplot(data=df_sorted, x='mechanism', y='Validation EMD', ax=ax2, palette='Set2')
        else:
            bp = ax2.boxplot(emd_data, labels=ordered_mechanisms, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
        
        ax2.set_title('Validation EMD Distribution by Mechanism', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Mechanism', fontsize=12)
        ax2.set_ylabel('Validation EMD (minutes)', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_plots:
        if run_id:
            filename = f'model_comparison_cv_emd_{run_id}.png'
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'model_comparison_cv_emd_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 Plot saved as: {filename}")
    
    plt.close()


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
    
    summary_df = create_summary_table_with_runid(all_results, run_id=args.run_id, save_table=True)
    create_comparison_plots_with_runid(all_results, run_id=args.run_id, save_plots=True)
    
    # Generate parameter distribution plots for each mechanism
    print(f"\n📊 Generating parameter distribution plots...")
    for result in all_results:
        if result['success'] and result['params']:
            print(f"   Plotting {result['mechanism']}...")
            plot_parameter_distributions(result, run_id=args.run_id, save_plots=True)
    
    print(f"\n🎉 Aggregation complete!")
    if args.run_id:
        print(f"📁 Results saved with run ID: {args.run_id}")
    
    if args.run_id:
        print(f"📋 Run ID: {args.run_id}")
    
    print(f"\n💡 Interpretation:")
    print(f"   - Lower Validation EMD indicates better model fit to unseen data")
    print(f"   - Use the summary table to identify the best mechanism(s)")
    print(f"   - Individual fold results are in the original CSV files")


if __name__ == "__main__":
    main()
