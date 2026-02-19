#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from multiprocessing import cpu_count
import sys
import os
import argparse


# Optional seaborn import
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Import cross-validation utilities
from CrossValidation import run_cross_validation
from simulation_utils import load_experimental_data


def get_parameter_count(mechanism):
    """
    Get the number of parameters for each mechanism.
    
    Args:
        mechanism (str): Mechanism name
    
    Returns:
        int: Number of parameters
    """
    try:
        from simulation_utils import get_parameter_names
        # Get parameter names and return length
        names = get_parameter_names(mechanism)
        return len(names)
    except Exception as e:
        print(f"Warning: Could not get parameter count for {mechanism}: {e}")
        return 0


def run_single_cv(mechanism, k_folds=5, n_simulations=2000, max_iter=1000, tol=0.01, run_id=None):
    """
    Run cross-validation for a single mechanism.
    
    Args:
        mechanism (str): Mechanism name
        k_folds (int): Number of folds for cross-validation
        n_simulations (int): Number of simulations per evaluation
        max_iter (int): Maximum iterations for differential evolution
        tol (float): Tolerance for optimization convergence
        run_id (str): Optional run identifier to distinguish result sets
    
    Returns:
        dict: Cross-validation results including mean and std of validation EMD
    """
    print(f"\n{'='*60}")
    print(f"Running {k_folds}-Fold Cross-Validation for: {mechanism.upper()}")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    try:
        # Run cross-validation (this function prints its own progress)
        run_cross_validation(mechanism, k_folds=k_folds, n_simulations=n_simulations, max_iter=max_iter, tol=tol)
        
        # Load the results from the CSV file created by run_cross_validation
        csv_file = 'ModelComparisonEMDResults/cv_results_{}.csv'.format(mechanism)
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            
            val_emds = df['val_emd'].values
            train_emds = df['train_emd'].values
            
            # Rename file to include run_id if provided
            if run_id:
                new_csv_file = 'ModelComparisonEMDResults/cv_results_{}_{}.csv'.format(mechanism, run_id)
                os.rename(csv_file, new_csv_file)
                csv_file = new_csv_file
            
            result = {
                'mechanism': mechanism,
                'n_params': get_parameter_count(mechanism),
                'k_folds': k_folds,
                'mean_val_emd': np.mean(val_emds),
                'std_val_emd': np.std(val_emds),
                'mean_train_emd': np.mean(train_emds),
                'std_train_emd': np.std(train_emds),
                'val_emds': val_emds.tolist(),
                'train_emds': train_emds.tolist(),
                'success': True,
                'csv_file': csv_file
            }
            
            print(f"\n Cross-Validation Complete for {mechanism}")
            print(f"   Mean Validation EMD: {result['mean_val_emd']:.2f} ± {result['std_val_emd']:.2f}")
            print(f"   Mean Training EMD: {result['mean_train_emd']:.2f} ± {result['std_train_emd']:.2f}")
            print(f"   Results saved to: {csv_file}")
            sys.stdout.flush()
            
            return result
        else:
            raise FileNotFoundError(f"CV results file not found: {csv_file}")
            
    except Exception as e:
        print(f" Error with {mechanism}: {e}")
        sys.stdout.flush()
        return {
            'mechanism': mechanism,
            'n_params': get_parameter_count(mechanism),
            'k_folds': k_folds,
            'mean_val_emd': np.nan,
            'std_val_emd': np.nan,
            'mean_train_emd': np.nan,
            'std_train_emd': np.nan,
            'val_emds': [],
            'train_emds': [],
            'success': False
        }


def create_comparison_plots(all_results, save_plots=True):
    """
    Create visualization plots for cross-validation model comparison.
    
    Args:
        all_results (list): List of CV results dictionaries
        save_plots (bool): Whether to save plots to files
    """
    # Filter out failed mechanisms
    successful_results = [r for r in all_results if r['success']]
    
    if not successful_results:
        print("No successful results to plot!")
        return
    
    # Define mechanism ordering (by complexity/number of additional mechanisms)
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
    
    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Comparison: Cross-Validation with EMD', fontsize=16, y=0.98)
    
    # 1. Mean Validation EMD comparison (bar plot with error bars)
    # Sort by predefined order
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
    
    # Add value labels on bars
    for i, (bar, val, std) in enumerate(zip(bars, mean_data['mean_val_emd'], mean_data['std_val_emd'])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 2,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Validation EMD distribution boxplot (ordered)
    if len(df) > 0:
        # Create ordered boxplot data
        ordered_mechanisms = [m for m in mechanism_order if m in df['mechanism'].unique()]
        emd_data = [df[df['mechanism'] == mech]['Validation EMD'].values for mech in ordered_mechanisms]
        
        if HAS_SEABORN:
            df['mechanism'] = pd.Categorical(df['mechanism'], categories=mechanism_order, ordered=True)
            df_sorted = df.sort_values('mechanism')
            sns.boxplot(data=df_sorted, x='mechanism', y='Validation EMD', ax=ax2, palette='Set2')
        else:
            # Fallback to matplotlib boxplot
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'ModelComparisonEMDResults/model_comparison_cv_emd_{timestamp}.pdf'
        plt.savefig(filename, dpi=300, bbox_inches='tight', format='pdf')
        print(f"\n📊 Plot saved as: {filename}")
    
    plt.close()


def create_summary_table(all_results, save_table=True):
    """
    Create and display a summary table of all cross-validation results.
    
    Args:
        all_results (list): List of CV results dictionaries
        save_table (bool): Whether to save table to CSV
    
    Returns:
        pd.DataFrame: Summary table
    """
    # Create summary DataFrame
    summary_data = []
    for result in all_results:
        summary_data.append({
            'Mechanism': result['mechanism'],
            'Parameters': result['n_params'],
            'K-Folds': result['k_folds'],
            'Mean Val EMD': f"{result['mean_val_emd']:.2f}" if not np.isnan(result['mean_val_emd']) else "N/A",
            'Std Val EMD': f"{result['std_val_emd']:.2f}" if not np.isnan(result['std_val_emd']) else "N/A",
            'Mean Train EMD': f"{result['mean_train_emd']:.2f}" if not np.isnan(result['mean_train_emd']) else "N/A",
            'Std Train EMD': f"{result['std_train_emd']:.2f}" if not np.isnan(result['std_train_emd']) else "N/A",
            'Status': 'Success' if result['success'] else 'Failed'
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Sort by mean validation EMD (best models first)
    successful_mask = df_summary['Mean Val EMD'] != "N/A"
    df_successful = df_summary[successful_mask].copy()
    df_failed = df_summary[~successful_mask].copy()
    
    if len(df_successful) > 0:
        df_successful['EMD_numeric'] = df_successful['Mean Val EMD'].astype(float)
        df_successful = df_successful.sort_values('EMD_numeric').drop('EMD_numeric', axis=1)
        df_summary = pd.concat([df_successful, df_failed], ignore_index=True)
    
    print(f"\n{'='*100}")
    print("CROSS-VALIDATION MODEL COMPARISON SUMMARY")
    print(f"{'='*100}")
    print(df_summary.to_string(index=False))
    
    # Identify best model
    if len(df_successful) > 0:
        print(f"\n{'='*60}")
        print("BEST MODEL:")
        print(f"{'='*60}")
        
        best_idx = df_successful['Mean Val EMD'].astype(float).idxmin()
        best_mechanism = df_summary.loc[best_idx, 'Mechanism']
        best_emd = df_summary.loc[best_idx, 'Mean Val EMD']
        best_std = df_summary.loc[best_idx, 'Std Val EMD']
        
        print(f"🏆 Best Model: {best_mechanism}")
        print(f"   Validation EMD: {best_emd} ± {best_std} minutes")
        print(f"   Parameters: {df_summary.loc[best_idx, 'Parameters']}")
    
    if save_table:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'ModelComparisonEMDResults/model_comparison_cv_summary_{timestamp}.csv'
        df_summary.to_csv(filename, index=False)
        print(f"\n💾 Summary table saved as: {filename}")
    
    return df_summary


def main():
    # ========== CONFIGURATION (Edit these for different runs) ==========
    K_FOLDS = 5              # Number of cross-validation folds
    N_SIMULATIONS = 5000     # Number of simulations per evaluation
    MAX_ITER = 2000          # Maximum iterations for optimization
    TOL = 0.01               # Tolerance for optimization convergence
    # ===================================================================
    
    # Parse command-line arguments (CLI args override config above)
    parser = argparse.ArgumentParser(description='Cross-Validation Model Comparison using EMD')
    parser.add_argument('--mechanism', type=str, default=None,
                       help='Single mechanism to test (for SLURM job array mode)')
    parser.add_argument('--run-id', type=str, default=None,
                       help='Run identifier to distinguish result sets')
    parser.add_argument('--k-folds', type=int, default=None,
                       help=f'Number of folds for cross-validation (default: {K_FOLDS} from config)')
    parser.add_argument('--n-simulations', type=int, default=None,
                       help=f'Number of simulations per evaluation (default: {N_SIMULATIONS} from config)')
    parser.add_argument('--max-iter', type=int, default=None,
                       help=f'Maximum iterations for optimization (default: {MAX_ITER} from config)')
    parser.add_argument('--tol', type=float, default=None,
                       help=f'Tolerance for optimization convergence (default: {TOL} from config)')
    
    args = parser.parse_args()
    
    # Use config values if CLI not provided
    k_folds = K_FOLDS
    n_simulations = N_SIMULATIONS
    max_iter = MAX_ITER
    tol = TOL
    
    # Single-mechanism mode (for SLURM job arrays)
    if args.mechanism:
        print("="*80)
        print(f"CROSS-VALIDATION FOR SINGLE MECHANISM: {args.mechanism}")
        print("="*80)
        print(f"Run ID: {args.run_id if args.run_id else 'Not specified'}")
        print(f"\n  Cross-Validation Configuration:")
        print(f"   K-Folds: {k_folds}")
        print(f"   Simulations per evaluation: {n_simulations}")
        print(f"   Optimization iterations per fold: {max_iter}")
        print(f"   Optimization tolerance: {tol}")
        sys.stdout.flush()
        
        start_time = datetime.now()
        
        # Run CV for single mechanism
        result = run_single_cv(
            args.mechanism, 
            k_folds=k_folds, 
            n_simulations=n_simulations,
            max_iter=max_iter,
            tol=tol,
            run_id=args.run_id
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n  Completed in: {duration}")
        print(f" Single-mechanism cross-validation complete")
        print(f" Results saved to: {result.get('csv_file', 'cv_results_*.csv')}")
        
        return
    
    # Full comparison mode (original behavior)
    print("="*80)
    print("CHROMOSOME SEGREGATION MODEL COMPARISON")
    print("Cross-Validation with Earth Mover's Distance (EMD)")
    print("="*80)
    sys.stdout.flush()
    
    # Display system information
    n_cpus = cpu_count()
    print(f"\n System Information:")
    print(f"   Available CPUs: {n_cpus}")
    sys.stdout.flush()
    
    # Load experimental data
    print("\n Loading experimental data...")
    sys.stdout.flush()
    datasets = load_experimental_data()
    if not datasets:
        print("❌ Error: Could not load experimental data!")
        sys.stdout.flush()
        return
    
    print(f" Loaded {len(datasets)} datasets: {list(datasets.keys())}")
    
    # Display dataset breakdown
    print(f"\n Dataset breakdown:")
    for name, data in datasets.items():
        t12_points = len(data['delta_t12'])
        t32_points = len(data['delta_t32'])
        total = t12_points + t32_points
        print(f"   {name}: {total} points (T1-T2: {t12_points}, T3-T2: {t32_points})")
    sys.stdout.flush()
    
    # Define mechanisms to compare
    mechanisms = [
        # Time-varying rate mechanisms
        'time_varying_k',
        'time_varying_k_fixed_burst',
        'time_varying_k_combined',
        'time_varying_k_steric_hindrance',

        # Time-varying with feedback mechanisms
        'time_varying_k_wfeedback',
        'time_varying_k_fixed_burst_wfeedback',
        'time_varying_k_combined_wfeedback',
        'time_varying_k_steric_hindrance_wfeedback'
    ]
    
    print(f"\n🔬 Comparing {len(mechanisms)} mechanisms:")
    for i, mech in enumerate(mechanisms, 1):
        param_count = get_parameter_count(mech)
        print(f"  {i}. {mech} ({param_count} parameters)")
    sys.stdout.flush()
    
    print(f"\n  Cross-Validation Configuration:")
    print(f"   K-Folds: {k_folds}")
    print(f"   Simulations per evaluation: {n_simulations}")
    print(f"   Optimization iterations per fold: {max_iter}")
    print(f"   Optimization tolerance: {tol}")
    print(f"   Run ID: {args.run_id if args.run_id else 'Auto-generated'}")
    sys.stdout.flush()
    
    # Run cross-validation for each mechanism
    all_results = []
    start_time = datetime.now()
    run_id = args.run_id  # Assign for use in loop
    
    for i, mechanism in enumerate(mechanisms, 1):
        mechanism_start = datetime.now()
        print(f"\n Progress: {i}/{len(mechanisms)} mechanisms")
        print(f" Started at: {mechanism_start.strftime('%H:%M:%S')}")
        sys.stdout.flush()
        
        try:
            result = run_single_cv(mechanism, k_folds=k_folds, n_simulations=n_simulations, max_iter=max_iter, tol=tol, run_id=run_id)
            all_results.append(result)
            
            mechanism_end = datetime.now()
            mechanism_duration = mechanism_end - mechanism_start
            print(f"⏱  Mechanism {mechanism} completed in: {mechanism_duration}")
            
            # Estimate remaining time
            avg_time_per_mechanism = (mechanism_end - start_time) / i
            remaining_mechanisms = len(mechanisms) - i
            estimated_remaining = avg_time_per_mechanism * remaining_mechanisms
            estimated_completion = mechanism_end + estimated_remaining
            
            if remaining_mechanisms > 0:
                print(f" Estimated completion: {estimated_completion.strftime('%H:%M:%S')} "
                      f"(~{estimated_remaining} remaining)")
            
        except Exception as e:
            print(f" Error with {mechanism}: {e}")
            all_results.append({
                'mechanism': mechanism,
                'n_params': get_parameter_count(mechanism),
                'k_folds': k_folds,
                'mean_val_emd': np.nan,
                'std_val_emd': np.nan,
                'mean_train_emd': np.nan,
                'std_train_emd': np.nan,
                'val_emds': [],
                'train_emds': [],
                'success': False
            })
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n  Total analysis time: {duration}")
    
    # Calculate statistics
    successful_mechanisms = sum(1 for r in all_results if r['success'])
    success_rate = (successful_mechanisms / len(mechanisms)) * 100
    
    print(f"\n Overall Statistics:")
    print(f"   Total mechanisms tested: {len(mechanisms)}")
    print(f"   Successful: {successful_mechanisms}")
    print(f"   Success rate: {success_rate:.1f}%")
    
    # Create summary table and plots
    print(f"\n Creating summary and visualizations...")
    summary_df = create_summary_table(all_results, save_table=True)
    create_comparison_plots(all_results, save_plots=True)
    
    print(f"\n Cross-validation model comparison complete!")
    print(f" Results saved with timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    print(f"\n Interpretation:")
    print(f"   - Lower Validation EMD indicates better model fit to unseen data")
    print(f"   - Lower Std EMD indicates more stable/robust predictions")
    print(f"   - CV eliminates the need for multiple optimization runs")


if __name__ == "__main__":
    main()
