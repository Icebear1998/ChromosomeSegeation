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
from ModelComparison import (
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
    # Search in ModelComparisonEMDResults folder
    search_dir = 'ModelComparisonEMDResults'
    
    if run_id:
        # Find files with specific run_id
        pattern = f'{search_dir}/cv_results_*_{run_id}.csv'
    else:
        # Find all cv_results files
        pattern = f'{search_dir}/cv_results_*.csv'
    
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
                print(f"  Warning: Unexpected filename format: {csv_file}")
                continue
            
            # Load CSV
            df = pd.read_csv(csv_file)
            
            if len(df) == 0:
                print(f"  Warning: Empty CSV file: {csv_file}")
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
            print(f" Loaded: {mechanism} (n={len(df)} folds, Val EMD={result['mean_val_emd']:.2f})")
            
        except Exception as e:
            print(f"❌ Error loading {csv_file}: {e}")
            continue
    
    return all_results


def plot_parameter_matrix(all_results, run_id=None, save_plots=True):
    """
    Create a matrix plot of parameter distributions across all mechanisms.
    Each row shows one parameter across all mechanisms.
    
    Args:
        all_results (list): List of results dictionaries for all mechanisms
        run_id (str): Run ID for filename (optional)
        save_plots (bool): Whether to save the plot
    """
    # Filter results with parameter data
    results_with_params = [r for r in all_results if r.get('params') and any(p is not None for p in r['params'])]
    
    if not results_with_params:
        print("  No parameter data available for any mechanism")
        return
        
    # User-specified mechanism order
    desired_order = [
        'simple',
        'fixed_burst',
        'feedback_onion',
        'fixed_burst_feedback_onion',
        'time_varying_k',
        'time_varying_k_fixed_burst',
        'time_varying_k_feedback_onion',
        'time_varying_k_combined'
    ]
    
    # Sort results to match desired order
    # (Put any unspecified mechanisms at the end)
    def sort_key(result):
        mech = result['mechanism']
        if mech in desired_order:
            return desired_order.index(mech)
        return len(desired_order) + 1
        
    results_with_params.sort(key=sort_key)
    
    # Define parameter order (all possible parameters)
    # Note: k and k_max are merged into 'k/k_max'
    # Added n1/N1, n2/N2, n3/N3 at the bottom
    param_order = [
        'n2', 'N2',           # Population sizes
        'k/k_max', 'tau',     # Rate parameters (k and k_max merged)
        'r21', 'r23',         # Ratios
        'R21', 'R23',         # Chromosome-specific ratios
        'burst_size',         # Burst
        'n_inner',            # Feedback
        'alpha',              # Mutant parameters
        'beta_k', 'beta_k1', 'beta_k2', 'beta_k3', # New beta params
        'beta_tau', 'beta_tau2',
        'n1/N1', 'n2/N2', 'n3/N3' # Derived concentrations
    ]
    
    # Collect all unique parameters and mechanisms
    all_param_names = []
    
    # Process mechanism names for display (Renaming)
    mechanism_names = []
    for r in results_with_params:
        name = r['mechanism']
        # Replace 'feedback_onion' with 'steric_hindrance'
        display_name = name.replace('feedback_onion', 'steric_hindrance')
        mechanism_names.append(display_name)
        
    n_mechanisms = len(mechanism_names)
    
    # Debug: print mechanism names
    print(f"DEBUG: Mechanism names for plotting: {mechanism_names}")
    
    for result in results_with_params:
        try:
            param_names = get_parameter_names(result['mechanism'])
            for name in param_names:
                # Map k and k_max to 'k/k_max'
                if name in ['k', 'k_max']:
                    display_name = 'k/k_max'
                else:
                    display_name = name
                    
                if display_name not in all_param_names:
                    all_param_names.append(display_name)
        except:
            continue
            
    # Add derived parameters to the list if check passes (they rely on other params)
    # Since all mechanisms have n2, N2, etc., we assume these can be calculated
    all_param_names.extend(['n1/N1', 'n2/N2', 'n3/N3'])
    
    # Sort parameters by defined order
    all_param_names = [p for p in param_order if p in all_param_names]
    
    n_params = len(all_param_names)
    
    # Create figure with one row per parameter
    fig, axes = plt.subplots(n_params, 1, 
                            figsize=(max(12, 1.5 * n_mechanisms), 2.0 * n_params),
                            sharex=False,
                            squeeze=False)
    
    fig.suptitle('Parameter Distributions Across Mechanisms', fontsize=18, fontweight='bold', y=0.99)
    
    # Iterate through parameters (rows)
    for param_idx, param_name in enumerate(all_param_names):
        ax = axes[param_idx, 0]
        
        # Collect data for this parameter across all mechanisms
        all_box_data = [None] * n_mechanisms  # Initialize with None for all positions
        all_point_data = [None] * n_mechanisms
        param_bound = None # Reset for each parameter
        
        for mech_idx, result in enumerate(results_with_params):
            mechanism = result['mechanism']
            params = result['params']
            
            if not params or all(p is None for p in params):
                continue
            
            try:
                param_names = get_parameter_names(mechanism)
                param_bounds = get_parameter_bounds(mechanism)
            except:
                continue
            
            # Helper to get column values
            def get_col(name):
                if name in param_names:
                    idx = param_names.index(name)
                    # Filter out None entries from params_array before indexing
                    valid_params = [p for p in params if p is not None]
                    if valid_params:
                        return np.array([p[idx] for p in valid_params])
                return None
            
            param_values = None # Initialize param_values for this mechanism and parameter
            
            # Handle Derived Parameters
            if param_name in ['n1/N1', 'n2/N2', 'n3/N3']:
                n2_vals = get_col('n2')
                N2_vals = get_col('N2')
                
                if n2_vals is not None and N2_vals is not None and len(n2_vals) == len(N2_vals):
                    if param_name == 'n2/N2':
                        param_values = n2_vals / N2_vals
                    elif param_name == 'n1/N1':
                        r21 = get_col('r21')
                        R21 = get_col('R21')
                        if r21 is not None and R21 is not None and len(r21) == len(n2_vals) and len(R21) == len(N2_vals):
                            n1_vals = np.maximum(r21 * n2_vals, 1.0)
                            N1_vals = np.maximum(R21 * N2_vals, 1.0)
                            param_values = n1_vals / N1_vals
                        else:
                            param_values = None
                    elif param_name == 'n3/N3':
                        r23 = get_col('r23')
                        R23 = get_col('R23')
                        if r23 is not None and R23 is not None and len(r23) == len(n2_vals) and len(R23) == len(N2_vals):
                            n3_vals = np.maximum(r23 * n2_vals, 1.0)
                            N3_vals = np.maximum(R23 * N2_vals, 1.0)
                            param_values = n3_vals / N3_vals
                        else:
                            param_values = None
                
            else:
                # Regular Parameters
                # Check if this mechanism has this parameter
                # For 'k/k_max', check for either 'k' or 'k_max'
                if param_name == 'k/k_max':
                    actual_param_name = 'k' if 'k' in param_names else ('k_max' if 'k_max' in param_names else None)
                else:
                    actual_param_name = param_name if param_name in param_names else None
                
                if actual_param_name:
                    param_local_idx = param_names.index(actual_param_name)
                    
                    # Get parameter values
                    params_array = np.array([p for p in params if p is not None])
                    if params_array.size > 0: # Check if array is not empty
                        param_values = params_array[:, param_local_idx]
                    else:
                        param_values = None
                    
                    # Get bounds (use first valid one we find for regular params)
                    if param_bound is None and param_bounds and param_local_idx < len(param_bounds):
                        param_bound = param_bounds[param_local_idx]
            
            # Store values if found
            if param_values is not None and len(param_values) > 0:
                all_box_data[mech_idx] = param_values
                all_point_data[mech_idx] = param_values
        
        # Filter out None values but track positions
        valid_positions = [i for i, data in enumerate(all_box_data) if data is not None]
        valid_box_data = [all_box_data[i] for i in valid_positions]
        valid_point_data = [all_point_data[i] for i in valid_positions]
        
        # Zebra striping for background
        if param_idx % 2 == 0:
            ax.set_facecolor('#f8f9fa')
            
        if valid_box_data:
            # Create box plots at consistent positions
            # Style: Lighter fill, thinner lines
            bp = ax.boxplot(valid_box_data, positions=valid_positions, widths=0.4, patch_artist=True,
                           boxprops=dict(facecolor='#e0e0e0', alpha=0.5, edgecolor='#666666'),
                           medianprops=dict(color='#d62728', linewidth=1.5),
                           whiskerprops=dict(linewidth=1, color='#666666'),
                           capprops=dict(linewidth=1, color='#666666'),
                           flierprops=dict(marker='o', markersize=3, alpha=0.5))
            
            # Overlay colored points for each mechanism
            # Use a nice palette
            colors = plt.cm.tab10(np.linspace(0, 1, 10))  
            
            for pos, param_values in zip(valid_positions, valid_point_data):
                n_points = len(param_values)
                # Jitter
                x_jitter = np.random.normal(pos, 0.05, size=n_points)
                x_jitter = np.clip(x_jitter, pos - 0.2, pos + 0.2)
                
                for point_idx, (x, y) in enumerate(zip(x_jitter, param_values)):
                    color = colors[point_idx % len(colors)]
                    ax.scatter(x, y, alpha=0.7, s=40, color=color, 
                              edgecolors='white', linewidths=0.5, zorder=3)
            
            # Set y-axis to parameter bounds
            if param_bound:
                if param_name in ['n1/N1', 'n2/N2', 'n3/N3']:
                    # Use auto-scaling for derived params or loose fixed limits if needed
                    # Typically concentration is < 0.1
                    # Let's check max value
                    max_val = max([np.max(d) for d in valid_box_data])
                    ax.set_ylim(0, max(max_val * 1.2, 0.05))
                else:
                    min_bound, max_bound = param_bound
                    ax.set_ylim(min_bound, max_bound)
            
            # Y-Axis Label
            ax.set_ylabel(param_name, fontsize=12, fontweight='bold', rotation=90, labelpad=10)
            ax.tick_params(axis='y', labelsize=10)
            
            # Grid
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')
            
            # Add vertical lines to separate mechanisms clearly
            for i in range(n_mechanisms - 1):
                ax.axvline(i + 0.5, color='gray', linestyle=':', alpha=0.3)

        else:
            # No mechanisms have this parameter
            ax.text(0.5, 0.5, f'{param_name} (Not used)', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=10, color='gray', style='italic')
            ax.set_yticks([])
            ax.set_ylabel(param_name, fontsize=12, fontweight='bold', color='gray')
            
        # Ensure x-limits match for all rows using the same range
        ax.set_xlim(-0.5, n_mechanisms - 0.5)
        
        # Hide default x-ticks for all plots initially
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.set_xticklabels([])

    ax_top = axes[0, 0]
    ax_top.xaxis.set_label_position('top')
    ax_top.xaxis.tick_top()
    ax_top.tick_params(axis='x', which='both', top=True, labeltop=True)
    ax_top.set_xticks(range(n_mechanisms))
    ax_top.set_xticklabels(mechanism_names, rotation=45, ha='left', fontsize=11, fontweight='bold')

    # Adjust layout
    # Increased top margin for rotated labels
    plt.subplots_adjust(hspace=0.1, left=0.1, right=0.95, top=0.88, bottom=0.05)
    
    if save_plots:
        if run_id:
            filename = f'ModelComparisonEMDResults/parameter_matrix_{run_id}'
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'ModelComparisonEMDResults/parameter_matrix_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        #plt.savefig(filename, dpi=300, bbox_inches='tight', format='svg')
        print(f"\n Parameter matrix saved as: {filename}")
    
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
            filename = f'ModelComparisonEMDResults/model_comparison_cv_summary_{run_id}.csv'
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'ModelComparisonEMDResults/model_comparison_cv_summary_{timestamp}.csv'
        df_summary.to_csv(filename, index=False)
        print(f"\n Summary table saved as: {filename}")
    
    return df_summary


def create_comparison_plots_with_runid(all_results, run_id=None, save_plots=True):
    """
    Wrapper for create_comparison_plots that uses run_id for filename.
    Creates two separate plots instead of subplots.
    """
    # Filter out failed mechanisms
    successful_results = [r for r in all_results if r['success']]
    
    if not successful_results:
        print("No successful results to plot!")
        return
    
    # Define mechanism ordering (updated with new order and names)
    mechanism_order = [
        'simple',
        'fixed_burst',
        'steric_hindrance', 
        'fixed_burst_steric_hindrance',
        'time_varying_k',
        'time_varying_k_fixed_burst',
        'time_varying_k_steric_hindrance',
        'time_varying_k_combined'
    ]
    
    # Create DataFrame for plotting
    plot_data = []
    for result in successful_results:
        # Rename mechanism for display
        mechanism = result['mechanism'].replace('feedback_onion', 'steric_hindrance')
        
        for val_emd in result['val_emds']:
            plot_data.append({
                'mechanism': mechanism,
                'Validation EMD': val_emd
            })
    
    df = pd.DataFrame(plot_data)
    
    # ========== PLOT 1: Mean Validation EMD - DOT AND WHISKER ==========
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    mean_data = pd.DataFrame(successful_results)
    
    # Calculate SEM (Standard Error of Mean) instead of Std
    mean_data['sem_val_emd'] = mean_data.apply(
        lambda row: np.std(row['val_emds']) / np.sqrt(len(row['val_emds'])), axis=1
    )
    
    # Rename in mean_data as well
    mean_data['mechanism'] = mean_data['mechanism'].apply(lambda x: x.replace('feedback_onion', 'steric_hindrance'))
    
    mean_data['order'] = mean_data['mechanism'].map({m: i for i, m in enumerate(mechanism_order)})
    
    # Handle unknown mechanisms (assign order = len)
    mean_data['order'] = mean_data['order'].fillna(len(mechanism_order))
    
    mean_data = mean_data.sort_values('order')
    
    # DOT AND WHISKER PLOT (no connecting lines, no bars)
    x_positions = range(len(mean_data))
    ax1.errorbar(x_positions, mean_data['mean_val_emd'], 
                 yerr=mean_data['sem_val_emd'], 
                 fmt='o',  # Dots only, no connecting line
                 markersize=8,
                 capsize=5, 
                 capthick=2,
                 elinewidth=2,
                 color='steelblue',
                 markerfacecolor='steelblue',
                 markeredgecolor='darkblue',
                 markeredgewidth=1.5,
                 alpha=0.8)
    
    ax1.set_xlabel('Mechanism', fontsize=12)
    ax1.set_ylabel('Total validation EMD (min)', fontsize=12)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(mean_data['mechanism'], rotation=45, ha='right')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # DO NOT start y-axis from 0 - let matplotlib auto-scale
    ax1.set_ylim(bottom=None)
    
    # Add value labels
    for i, (x, val, sem) in enumerate(zip(x_positions, mean_data['mean_val_emd'], mean_data['sem_val_emd'])):
        ax1.text(x, val + sem + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.02,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_plots:
        if run_id:
            filename1 = f'ModelComparisonEMDResults/model_comparison_mean_emd_{run_id}.pdf'
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename1 = f'ModelComparisonEMDResults/model_comparison_mean_emd_{timestamp}.pdf'
        plt.savefig(filename1, dpi=300, bbox_inches='tight', format='pdf')
        #plt.savefig(filename1, dpi=300, bbox_inches='tight', format='svg')
        print(f"\n Mean EMD plot saved as: {filename1}")
    
    plt.close()
    
    # ========== PLOT 2: Validation EMD Distribution Boxplot ==========
    if len(df) > 0:
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        
        ordered_mechanisms = [m for m in mechanism_order if m in df['mechanism'].unique()]
        # Add any remaining mechanisms not in the order list
        existing_mechs = set(df['mechanism'].unique())
        ordered_mechs_set = set(ordered_mechanisms)
        remaining = list(existing_mechs - ordered_mechs_set)
        ordered_mechanisms.extend(remaining)
        
        emd_data = [df[df['mechanism'] == mech]['Validation EMD'].values for mech in ordered_mechanisms]
        
        if HAS_SEABORN:
            # Update categories for seaborn
            df['mechanism'] = pd.Categorical(df['mechanism'], categories=ordered_mechanisms, ordered=True)
            df_sorted = df.sort_values('mechanism')
            sns.boxplot(data=df_sorted, x='mechanism', y='Validation EMD', ax=ax2, palette='Set2')
        else:
            bp = ax2.boxplot(emd_data, labels=ordered_mechanisms, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
        
        ax2.set_xlabel('Mechanism', fontsize=12)
        ax2.set_ylabel('Total validation EMD (min)', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_plots:
            if run_id:
                filename2 = f'ModelComparisonEMDResults/model_comparison_boxplot_emd_{run_id}.pdf'
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename2 = f'ModelComparisonEMDResults/model_comparison_boxplot_emd_{run_id}.pdf'
            plt.savefig(filename2, dpi=300, bbox_inches='tight', format='pdf')
            #plt.savefig(filename2, dpi=300, bbox_inches='tight', format='svg')
            print(f" Boxplot distribution saved as: {filename2}")
        
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
        print(f"\n Using custom pattern: {args.pattern}")
        csv_files = glob.glob(args.pattern)
    elif args.run_id:
        print(f"\n Finding results for Run ID: {args.run_id}")
        csv_files = find_cv_result_files(args.run_id)
    else:
        print("\n  WARNING: No run ID specified!")
        print("   This will aggregate ALL cv_results_*.csv files in the directory.")
        print("   Results from different runs may be mixed.")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        csv_files = find_cv_result_files(None)
    
    if not csv_files:
        print(f"\n No CV result files found!")
        if args.run_id:
            print(f"   Searched for: ModelComparisonEMDResults/cv_results_*_{args.run_id}.csv")
        print("\n Make sure the SLURM job array has completed successfully.")
        return
    
    print(f"\n Found {len(csv_files)} CV result files:")
    for f in csv_files:
        print(f"   - {f}")
    
    # Load and aggregate results
    print(f"\n Loading and aggregating results...")
    all_results = load_cv_results(csv_files)
    
    if not all_results:
        print(f"\n❌ No results could be loaded!")
        return
    
    print(f"\n Successfully loaded {len(all_results)} mechanism results")
    
    # Generate summary table and plots
    print(f"\n Generating summary and visualizations...")
    
    summary_df = create_summary_table_with_runid(all_results, run_id=args.run_id, save_table=True)
    create_comparison_plots_with_runid(all_results, run_id=args.run_id, save_plots=True)
    
    # Generate parameter matrix plot (all mechanisms in one plot)
    print(f"\n Generating parameter matrix plot...")
    plot_parameter_matrix(all_results, run_id=args.run_id, save_plots=True)
    
    print(f"\n Aggregation complete!")
    if args.run_id:
        print(f" Results saved with run ID: {args.run_id}")
    
    if args.run_id:
        print(f" Run ID: {args.run_id}")
    
    print(f"\n Interpretation:")
    print(f"   - Lower Validation EMD indicates better model fit to unseen data")
    print(f"   - Use the summary table to identify the best mechanism(s)")
    print(f"   - Individual fold results are in the original CSV files")


if __name__ == "__main__":
    main()
