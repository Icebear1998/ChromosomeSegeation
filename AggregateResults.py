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
    
    # Define mechanism ordering (time-varying only)
    mechanism_order = [
        # Standard time-varying mechanisms
        'time_varying_k',
        'time_varying_k_fixed_burst',
        'time_varying_k_steric_hindrance',
        'time_varying_k_combined',
        # Feedback variants
        'time_varying_k_wfeedback',
        'time_varying_k_fixed_burst_wfeedback',
        'time_varying_k_steric_hindrance_wfeedback',
        'time_varying_k_combined_wfeedback'
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
    
    # ========== PLOT 1: Mean Validation EMD - DOT AND WHISKER ==========
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    mean_data = pd.DataFrame(successful_results)
    
    # Calculate SEM (Standard Error of Mean) instead of Std
    mean_data['sem_val_emd'] = mean_data.apply(
        lambda row: np.std(row['val_emds']) / np.sqrt(len(row['val_emds'])), axis=1
    )
    

    
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
