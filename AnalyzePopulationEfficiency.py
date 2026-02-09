#!/usr/bin/env python3
"""
Population Size Analysis for EMD-based Optimization.

This script analyzes the efficiency of different population size (popsize) values
for EMD-based optimization using cross-validation. It tests how population size
affects optimization quality and computational cost.

Key features:
- Tests popsize values from 5 to 30
- Uses N=10000 simulations for high fidelity
- Reports convergence status and average iterations
- Uses 5-fold cross-validation to get robust statistics
- Tracks both training and validation EMD
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


def run_single_fold_with_popsize(mechanism, train_data, val_data, popsize, n_simulations=10000, max_iterations=2000, tol=0.01):
    """
    Run optimization for a single fold with a specific population size.
    
    Args:
        mechanism (str): Mechanism name
        train_data (dict): Training data for this fold
        val_data (dict): Validation data for this fold
        popsize (int): Population size for differential evolution
        n_simulations (int): Number of simulations per evaluation
        max_iterations (int): Maximum iterations for DE
        tol (float): Tolerance for convergence
    
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
            popsize=popsize,
            tol=tol,
            disp=False,
            workers=-1,  # Use all available CPUs for DE optimization
            polish=True
        )
        
        train_emd = result.fun
        
        # Validate on Held-out Data
        if result.success or result.nit >= max_iterations:
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


def analyze_popsize_efficiency(mechanism, popsize_values, k_folds=5, n_simulations=10000, max_iterations=2000, tol=0.01):
    """
    Analyze optimization efficiency for different population sizes using cross-validation.
    
    Args:
        mechanism (str): Mechanism name
        popsize_values (list): List of population sizes to test
        k_folds (int): Number of folds for cross-validation
        n_simulations (int): Number of simulations per evaluation
        max_iterations (int): Maximum iterations for DE
        tol (float): Tolerance for convergence
    
    Returns:
        dict: Results for each population size
    """
    print(f"\n{'='*60}")
    print(f"Analyzing population size efficiency for: {mechanism}")
    print(f"{'='*60}")
    print(f"Population sizes: {popsize_values}")
    print(f"K-folds: {k_folds}")
    print(f"Simulations per evaluation: {n_simulations}")
    print(f"Max iterations: {max_iterations}")
    print(f"Tolerance: {tol}")
    sys.stdout.flush()
    
    # Load and create folds
    datasets = load_experimental_data()
    folds = create_folds(datasets, k_folds=k_folds, seed=42)
    
    results = {}
    
    for popsize in popsize_values:
        print(f"\n--- Testing popsize={popsize} ---")
        sys.stdout.flush()
        
        fold_train_emds = []
        fold_val_emds = []
        fold_converged = []
        fold_iterations = []
        
        for k in range(k_folds):
            print(f"  Fold {k+1}/{k_folds}: ", end="", flush=True)
            
            train_data = folds[k]['train']
            val_data = folds[k]['val']
            
            fold_result = run_single_fold_with_popsize(
                mechanism, train_data, val_data, popsize, n_simulations, max_iterations, tol
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
            results[popsize] = {
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
            
            print(f"\n  Summary for popsize={popsize}:")
            print(f"    Mean Train EMD: {results[popsize]['mean_train_emd']:.2f} ± {results[popsize]['std_train_emd']:.2f}")
            print(f"    Mean Val EMD:   {results[popsize]['mean_val_emd']:.2f} ± {results[popsize]['std_val_emd']:.2f}")
            print(f"    Mean Iterations: {results[popsize]['mean_iterations']:.1f} ± {results[popsize]['std_iterations']:.1f}")
            print(f"    Convergence Rate: {results[popsize]['convergence_rate']*100:.0f}% ({results[popsize]['n_converged']}/{results[popsize]['n_folds']})")
        else:
            results[popsize] = {
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
            print(f"\n  Summary for popsize={popsize}: All folds failed")
        
        sys.stdout.flush()
    
    return results


def create_popsize_analysis_plots(mechanism, popsize_values, results, save_plots=True):
    """
    Create visualization plots for population size analysis.
    
    Args:
        mechanism (str): Mechanism name
        popsize_values (list): List of population sizes tested
        results (dict): Results from analyze_popsize_efficiency
        save_plots (bool): Whether to save plots
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Population Size Analysis for {mechanism} (EMD-based Optimization)', fontsize=16, y=0.98)
    
    # Extract data
    mean_val_emds = [results[popsize]['mean_val_emd'] for popsize in popsize_values]
    std_val_emds = [results[popsize]['std_val_emd'] for popsize in popsize_values]
    mean_train_emds = [results[popsize]['mean_train_emd'] for popsize in popsize_values]
    
    mean_iters = [results[popsize]['mean_iterations'] for popsize in popsize_values]
    std_iters = [results[popsize]['std_iterations'] for popsize in popsize_values]
    conv_rates = [results[popsize]['convergence_rate'] * 100 for popsize in popsize_values]
    n_converged = [results[popsize]['n_converged'] for popsize in popsize_values]
    n_folds = [results[popsize]['n_folds'] for popsize in popsize_values]
    
    # 1. Mean EMD vs Population Size (Validation + Train)
    ax1.errorbar(popsize_values, mean_val_emds, yerr=std_val_emds, 
                 marker='o', capsize=5, linewidth=2, markersize=8, color='steelblue', label='Validation EMD')
    ax1.plot(popsize_values, mean_train_emds, 
             marker='x', linestyle='--', color='gray', alpha=0.7, label='Training EMD')
             
    ax1.set_xlabel('Population Size', fontsize=12)
    ax1.set_ylabel('Mean EMD (minutes)', fontsize=12)
    ax1.set_title('Mean EMD vs Population Size (Lower is Better)', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels for Validation EMD
    for i, (popsize, emd, std) in enumerate(zip(popsize_values, mean_val_emds, std_val_emds)):
        if not np.isnan(emd):
            ax1.text(popsize, emd + std + 1, f'{emd:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Mean Iterations vs Population Size
    ax2.errorbar(popsize_values, mean_iters, yerr=std_iters,
                 marker='s', capsize=5, linewidth=2, markersize=8, color='coral')
    ax2.set_xlabel('Population Size', fontsize=12)
    ax2.set_ylabel('Mean Iterations to Convergence', fontsize=12)
    ax2.set_title('Iterations vs Population Size', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (popsize, iters, std) in enumerate(zip(popsize_values, mean_iters, std_iters)):
        if not np.isnan(iters):
            ax2.text(popsize, iters + std + 5, f'{iters:.0f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Convergence Rate
    bars = ax3.bar(range(len(popsize_values)), conv_rates, alpha=0.7, color='forestgreen')
    ax3.set_xlabel('Population Size', fontsize=12)
    ax3.set_ylabel('Convergence Rate (%)', fontsize=12)
    ax3.set_title('Convergence Rate vs Population Size', fontsize=13, fontweight='bold')
    ax3.set_xticks(range(len(popsize_values)))
    ax3.set_xticklabels(popsize_values)
    ax3.set_ylim([0, 105])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add convergence count labels
    for i, (bar, conv, total) in enumerate(zip(bars, n_converged, n_folds)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height + 2,
                f'{conv}/{total}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Val EMD vs Iterations scatter plot
    for popsize, result in results.items():
        if result['all_val_emds']:
            ax4.scatter(result['all_iterations'], result['all_val_emds'], 
                       label=f'popsize={popsize}', alpha=0.7, s=100)
    
    ax4.set_xlabel('Number of Iterations', fontsize=12)
    ax4.set_ylabel('Validation EMD (minutes)', fontsize=12)
    ax4.set_title('Validation EMD vs Iterations (All Folds)', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'popsize_efficiency_emd_{mechanism}_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 Plot saved as: {filename}")
    
    plt.close()


def create_summary_table(mechanism, popsize_values, results, save_table=True):
    """
    Create and display a summary table of population size analysis results.
    
    Args:
        mechanism (str): Mechanism name
        popsize_values (list): List of population sizes tested
        results (dict): Results from analyze_popsize_efficiency
        save_table (bool): Whether to save table to CSV
    
    Returns:
        pd.DataFrame: Summary table
    """
    summary_data = []
    for popsize in popsize_values:
        r = results[popsize]
        summary_data.append({
            'PopSize': popsize,
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
    print(f"POPULATION SIZE ANALYSIS SUMMARY - {mechanism}")
    print(f"{'='*100}")
    print(df.to_string(index=False))
    
    if save_table:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'popsize_efficiency_summary_{mechanism}_{timestamp}.csv'
        df.to_csv(filename, index=False)
        print(f"\n💾 Summary table saved as: {filename}")
    
    return df


def main():
    """
    Main function to run population size analysis.
    """
    print("="*80)
    print("POPULATION SIZE EFFICIENCY ANALYSIS - EMD OPTIMIZATION")
    print("="*80)
    sys.stdout.flush()
    
    # Configuration
    mechanism = 'time_varying_k'  # Can be changed to test other mechanisms
    
    # Test range: 5 to 30
    popsize_values = [5, 10, 15, 20]
    
    k_folds = 5
    n_simulations = 10000
    max_iterations = 1000
    tol = 0.01  # Use efficient tolerance found from previous analysis
    
    print(f"\n📋 Configuration:")
    print(f"   Mechanism: {mechanism}")
    print(f"   Population sizes: {popsize_values}")
    print(f"   K-folds: {k_folds}")
    print(f"   Simulations per evaluation: {n_simulations}")
    print(f"   Max iterations: {max_iterations}")
    print(f"   Tolerance: {tol}")
    sys.stdout.flush()
    
    # Run analysis
    from datetime import datetime
    start_time = datetime.now()
    
    results = analyze_popsize_efficiency(
        mechanism, popsize_values, k_folds, n_simulations, max_iterations, tol
    )
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n⏱️  Total analysis time: {duration}")
    
    # Create summary and visualizations
    print(f"\n📊 Creating summary and visualizations...")
    summary_df = create_summary_table(mechanism, popsize_values, results, save_table=True)
    create_popsize_analysis_plots(mechanism, popsize_values, results, save_plots=True)
    
    print(f"\n🎉 Population size analysis complete!")
    print(f"\n💡 Recommendations:")
    
    # Find best population size
    valid_popsizes = [(popsize, results[popsize]['mean_val_emd']) for popsize in popsize_values 
                      if not np.isnan(results[popsize]['mean_val_emd']) and results[popsize]['convergence_rate'] == 1.0]
    
    if valid_popsizes:
        # Best = lowest Validation EMD among fully converged
        best_popsize, best_val_emd = min(valid_popsizes, key=lambda x: x[1])
        print(f"   🏆 Best population size: {best_popsize} (Val EMD={best_val_emd:.2f}, 100% convergence)")
        
        # Fastest = fewest iterations among fully converged
        fastest_popsize = min([popsize for popsize, _ in valid_popsizes], 
                             key=lambda p: results[p]['mean_iterations'])
        print(f"   ⚡ Fastest population size: {fastest_popsize} (Avg {results[fastest_popsize]['mean_iterations']:.0f} iterations)")
    else:
        print(f"   ⚠️  No population size achieved 100% convergence rate")


if __name__ == "__main__":
    main()
