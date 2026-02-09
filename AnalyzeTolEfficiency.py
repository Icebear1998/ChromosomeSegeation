#!/usr/bin/env python3
"""
Tolerance Analysis for EMD-based Optimization.

This script analyzes the efficiency of different tolerance (tol) values
for EMD-based optimization using cross-validation. It tests how quickly
the optimization converges and what EMD it achieves for different tol values.

Key features:
- Tests tol values from 0.05 to 0.001
- Uses N=10000 simulations for high fidelity
- Reports convergence status and average iterations
- Uses 5-fold cross-validation to get robust statistics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from scipy.optimize import differential_evolution

# Import cross-validation and simulation utilities
from CrossValidation import create_folds, objective_function
from simulation_utils import load_experimental_data, get_parameter_bounds


def run_single_fold_with_tol(mechanism, train_data, val_data, tol, n_simulations=10000, max_iterations=2000):
    """
    Run optimization for a single fold with a specific tolerance value.
    
    Args:
        mechanism (str): Mechanism name
        train_data (dict): Training data for this fold
        val_data (dict): Validation data for this fold
        tol (float): Tolerance value for differential evolution
        n_simulations (int): Number of simulations per evaluation
        max_iterations (int): Maximum iterations for DE
    
    Returns:
        dict: Results including Train/Val EMD, convergence status, and iteration count
    """
    bounds = get_parameter_bounds(mechanism)
    
    try:
        # Optimize on Training Data
        result = differential_evolution(
            objective_function,
            bounds,
            args=(mechanism, train_data, n_simulations),
            strategy='best1bin',
            maxiter=max_iterations,
            popsize=10,
            tol=tol,
            mutation=(0.5, 1),
            recombination=0.7,
            disp=False,
            workers=-1,  # Use all available CPUs for DE optimization
            polish=True
        )
        
        train_emd = result.fun
        
        # Validate on Held-out Data
        if result.success or result.nit >= max_iterations: # Even if not converged, check val (max iter reached)
             val_emd = objective_function(result.x, mechanism, val_data, n_simulations)
        else:
             val_emd = np.nan
        
        return {
            'success': True,
            'train_emd': train_emd,
            'val_emd': val_emd,
            'converged': result.success,
            'n_iterations': result.nit,
            'message': result.message
        }
        
    except Exception as e:
        return {
            'success': False,
            'train_emd': np.nan,
            'val_emd': np.nan,
            'converged': False,
            'n_iterations': 0,
            'message': str(e)
        }


def analyze_tol_efficiency(mechanism, tol_values, k_folds=5, n_simulations=10000, max_iterations=2000):
    """
    Analyze optimization efficiency for different tolerance values using cross-validation.
    Note: differential_evolution uses workers=-1 internally for parallelization.
    
    Args:
        mechanism (str): Mechanism name
        tol_values (list): List of tolerance values to test
        k_folds (int): Number of folds for cross-validation
        n_simulations (int): Number of simulations per evaluation
        max_iterations (int): Maximum iterations for DE
    
    Returns:
        dict: Results for each tolerance value
    """
    print(f"\n{'='*60}")
    print(f"Analyzing tolerance efficiency for: {mechanism}")
    print(f"{'='*60}")
    print(f"Tolerance values: {tol_values}")
    print(f"K-folds: {k_folds}")
    print(f"Simulations per evaluation: {n_simulations}")
    print(f"Max iterations: {max_iterations}")
    sys.stdout.flush()
    
    # Load and create folds
    datasets = load_experimental_data()
    folds = create_folds(datasets, k_folds=k_folds, seed=42)
    
    results = {}
    
    for tol in tol_values:
        print(f"\n--- Testing tol={tol} ---")
        sys.stdout.flush()
        
        fold_train_emds = []
        fold_val_emds = []
        fold_converged = []
        fold_iterations = []
        
        for k in range(k_folds):
            print(f"  Fold {k+1}/{k_folds}: ", end="", flush=True)
            
            train_data = folds[k]['train']
            val_data = folds[k]['val']
            
            fold_result = run_single_fold_with_tol(
                mechanism, train_data, val_data, tol, n_simulations, max_iterations
            )
            
            if fold_result['success']:
                fold_train_emds.append(fold_result['train_emd'])
                fold_val_emds.append(fold_result['val_emd'])
                fold_converged.append(fold_result['converged'])
                fold_iterations.append(fold_result['n_iterations'])
                
                status_icon = "✓" if fold_result['converged'] else "○"
                print(f"{status_icon} Train EMD={fold_result['train_emd']:.2f}, Val EMD={fold_result['val_emd']:.2f}, iter={fold_result['n_iterations']}")
            else:
                print(f"✗ Failed: {fold_result['message']}")
            
            sys.stdout.flush()
        
        # Calculate statistics
        if fold_train_emds:
            results[tol] = {
                'mean_train_emd': np.mean(fold_train_emds),
                'std_train_emd': np.std(fold_train_emds),
                'mean_val_emd': np.mean(fold_val_emds),
                'std_val_emd': np.std(fold_val_emds),
                'mean_iterations': np.mean(fold_iterations),
                'std_iterations': np.std(fold_iterations),
                'convergence_rate': sum(fold_converged) / len(fold_converged),
                'n_converged': sum(fold_converged),
                'n_folds': len(fold_train_emds),
                'all_train_emds': fold_train_emds,
                'all_val_emds': fold_val_emds,
                'all_iterations': fold_iterations
            }
            
            print(f"\n  Summary for tol={tol}:")
            print(f"    Mean Train EMD: {results[tol]['mean_train_emd']:.2f} ± {results[tol]['std_train_emd']:.2f}")
            print(f"    Mean Val EMD:   {results[tol]['mean_val_emd']:.2f} ± {results[tol]['std_val_emd']:.2f}")
            print(f"    Mean Iterations: {results[tol]['mean_iterations']:.1f} ± {results[tol]['std_iterations']:.1f}")
            print(f"    Convergence Rate: {results[tol]['convergence_rate']*100:.0f}% ({results[tol]['n_converged']}/{results[tol]['n_folds']})")
        else:
            results[tol] = {
                'mean_train_emd': np.nan,
                'std_train_emd': np.nan,
                'mean_val_emd': np.nan,
                'std_val_emd': np.nan,
                'mean_iterations': np.nan,
                'std_iterations': np.nan,
                'convergence_rate': 0.0,
                'n_converged': 0,
                'n_folds': 0,
                'all_train_emds': [],
                'all_val_emds': [],
                'all_iterations': []
            }
            print(f"\n  Summary for tol={tol}: All folds failed")
        
        sys.stdout.flush()
    
    return results


def create_tol_analysis_plots(mechanism, tol_values, results, save_plots=True):
    """
    Create visualization plots for tolerance analysis.
    
    Args:
        mechanism (str): Mechanism name
        tol_values (list): List of tolerance values tested
        results (dict): Results from analyze_tol_efficiency
        save_plots (bool): Whether to save plots
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Tolerance Analysis for {mechanism} (EMD-based Optimization)', fontsize=16, y=0.98)
    
    # Extract data
    mean_val_emds = [results[tol]['mean_val_emd'] for tol in tol_values]
    std_val_emds = [results[tol]['std_val_emd'] for tol in tol_values]
    mean_train_emds = [results[tol]['mean_train_emd'] for tol in tol_values]
    
    mean_iters = [results[tol]['mean_iterations'] for tol in tol_values]
    std_iters = [results[tol]['std_iterations'] for tol in tol_values]
    conv_rates = [results[tol]['convergence_rate'] * 100 for tol in tol_values]
    n_converged = [results[tol]['n_converged'] for tol in tol_values]
    n_folds = [results[tol]['n_folds'] for tol in tol_values]
    
    # 1. Mean EMD vs Tolerance (Validation + Train)
    ax1.errorbar(range(len(tol_values)), mean_val_emds, yerr=std_val_emds, 
                 marker='o', capsize=5, linewidth=2, markersize=8, color='steelblue', label='Validation EMD')
    ax1.plot(range(len(tol_values)), mean_train_emds, 
             marker='x', linestyle='--', color='gray', alpha=0.7, label='Training EMD')
             
    ax1.set_xlabel('Tolerance Value', fontsize=12)
    ax1.set_ylabel('Mean EMD (minutes)', fontsize=12)
    ax1.set_title('Mean EMD vs Tolerance (Lower is Better)', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(tol_values)))
    ax1.set_xticklabels([f'{tol:.3f}' for tol in tol_values], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels for Validation EMD
    for i, (emd, std) in enumerate(zip(mean_val_emds, std_val_emds)):
        if not np.isnan(emd):
            ax1.text(i, emd + std + 1, f'{emd:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Mean Iterations vs Tolerance
    ax2.errorbar(range(len(tol_values)), mean_iters, yerr=std_iters,
                 marker='s', capsize=5, linewidth=2, markersize=8, color='coral')
    ax2.set_xlabel('Tolerance Value', fontsize=12)
    ax2.set_ylabel('Mean Iterations to Convergence', fontsize=12)
    ax2.set_title('Iterations vs Tolerance (Fewer is Faster)', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(tol_values)))
    ax2.set_xticklabels([f'{tol:.3f}' for tol in tol_values], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (iters, std) in enumerate(zip(mean_iters, std_iters)):
        if not np.isnan(iters):
            ax2.text(i, iters + std + 5, f'{iters:.0f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Convergence Rate
    bars = ax3.bar(range(len(tol_values)), conv_rates, alpha=0.7, color='forestgreen')
    ax3.set_xlabel('Tolerance Value', fontsize=12)
    ax3.set_ylabel('Convergence Rate (%)', fontsize=12)
    ax3.set_title('Convergence Rate vs Tolerance', fontsize=13, fontweight='bold')
    ax3.set_xticks(range(len(tol_values)))
    ax3.set_xticklabels([f'{tol:.3f}' for tol in tol_values], rotation=45)
    ax3.set_ylim([0, 105])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add convergence count labels
    for i, (bar, conv, total) in enumerate(zip(bars, n_converged, n_folds)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height + 2,
                f'{conv}/{total}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Val EMD vs Iterations scatter plot
    for tol, result in results.items():
        if result['all_val_emds']:
            ax4.scatter(result['all_iterations'], result['all_val_emds'], 
                       label=f'tol={tol:.3f}', alpha=0.7, s=100)
    
    ax4.set_xlabel('Number of Iterations', fontsize=12)
    ax4.set_ylabel('Total EMD (sec)', fontsize=12)
    ax4.set_title('Validation EMD vs Iterations (All Folds)', fontsize=13, fontweight='bold')
    ax4.legend()
    
    plt.tight_layout()
    
    if save_plots:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'tol_efficiency_emd_{mechanism}_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n Plot saved as: {filename}")
    
    plt.close()


def create_summary_table(mechanism, tol_values, results, save_table=True):
    """
    Create and display a summary table of tolerance analysis results.
    
    Args:
        mechanism (str): Mechanism name
        tol_values (list): List of tolerance values tested
        results (dict): Results from analyze_tol_efficiency
        save_table (bool): Whether to save table to CSV
    
    Returns:
        pd.DataFrame: Summary table
    """
    summary_data = []
    for tol in tol_values:
        r = results[tol]
        summary_data.append({
            'Tolerance': tol,
            'Mean Train EMD': f"{r['mean_train_emd']:.2f}" if not np.isnan(r['mean_train_emd']) else "N/A",
            'Mean Val EMD': f"{r['mean_val_emd']:.2f}" if not np.isnan(r['mean_val_emd']) else "N/A",
            'Std Val EMD': f"{r['std_val_emd']:.2f}" if not np.isnan(r['std_val_emd']) else "N/A",
            'Mean Iterations': f"{r['mean_iterations']:.1f}" if not np.isnan(r['mean_iterations']) else "N/A",
            'Std Iterations': f"{r['std_iterations']:.1f}" if not np.isnan(r['std_iterations']) else "N/A",
            'Convergence Rate': f"{r['convergence_rate']*100:.0f}%",
            'Converged/Total': f"{r['n_converged']}/{r['n_folds']}"
        })
    
    df = pd.DataFrame(summary_data)
    
    print(f"\n{'='*100}")
    print(f"TOLERANCE ANALYSIS SUMMARY - {mechanism}")
    print(f"{'='*100}")
    print(df.to_string(index=False))
    
    if save_table:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'tol_efficiency_summary_{mechanism}_{timestamp}.csv'
        df.to_csv(filename, index=False)
        print(f"\n Summary table saved as: {filename}")
    
    return df


def main():
    """
    Main function to run tolerance analysis.
    """
    print("="*80)
    print("TOLERANCE EFFICIENCY ANALYSIS - EMD OPTIMIZATION")
    print("="*80)
    sys.stdout.flush()
    
    # Configuration
    mechanism = 'simple'  # Can be changed to test other mechanisms
    
    # Test range: 0.05 to 0.001
    tol_values = [0.05, 0.01, 0.005]#, 0.003, 0.001]
    
    k_folds = 5
    n_simulations = 10000
    max_iterations = 1000
    
    print(f"\n Configuration:")
    print(f"   Mechanism: {mechanism}")
    print(f"   Tolerance values: {tol_values}")
    print(f"   K-folds: {k_folds}")
    print(f"   Simulations per evaluation: {n_simulations}")
    print(f"   Max iterations: {max_iterations}")
    sys.stdout.flush()
    
    # Run analysis
    from datetime import datetime
    start_time = datetime.now()
    
    results = analyze_tol_efficiency(
        mechanism, tol_values, k_folds, n_simulations, max_iterations
    )
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n  Total analysis time: {duration}")
    
    # Create summary and visualizations
    print(f"\n Creating summary and visualizations...")
    summary_df = create_summary_table(mechanism, tol_values, results, save_table=True)
    create_tol_analysis_plots(mechanism, tol_values, results, save_plots=True)
    
    print(f"\n Tolerance analysis complete!")
    print(f"\n Recommendations:")
    
    # Find best tolerance
    valid_tols = [(tol, results[tol]['mean_val_emd']) for tol in tol_values 
                  if not np.isnan(results[tol]['mean_val_emd']) and results[tol]['convergence_rate'] == 1.0]
    
    if valid_tols:
        # Best = lowest Validation EMD among fully converged
        best_tol, best_val_emd = min(valid_tols, key=lambda x: x[1])
        print(f"    Best tolerance: {best_tol} (Val EMD={best_val_emd:.2f}, 100% convergence)")
        
        # Fastest = fewest iterations among fully converged
        fastest_tol = min([tol for tol, _ in valid_tols], 
                         key=lambda t: results[t]['mean_iterations'])
        print(f"    Fastest tolerance: {fastest_tol} (Avg {results[fastest_tol]['mean_iterations']:.0f} iterations)")
    else:
        print(f"     No tolerance value achieved 100% convergence rate")


if __name__ == "__main__":
    main()
