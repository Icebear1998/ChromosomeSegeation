#!/usr/bin/env python3
"""
Model Comparison using AIC and BIC for Chromosome Segregation Mechanisms.

This script compares 8 different mechanisms using Akaike Information Criterion (AIC) 
and Bayesian Information Criterion (BIC) by running multiple optimization runs 
for each mechanism and averaging the results.

Mechanisms compared:
1. Constant rate mechanisms (MultiMechanismSimulation):
   - simple: Constant degradation rate
   - fixed_burst: Constant rate with fixed burst sizes
   - feedback_onion: Constant rate with onion feedback
   - fixed_burst_feedback_onion: Constant rate with fixed bursts and onion feedback

2. Time-varying rate mechanisms (MultiMechanismSimulationTimevary):
   - time_varying_k: Time-varying degradation rate
   - time_varying_k_fixed_burst: Time-varying rate with fixed bursts
   - time_varying_k_feedback_onion: Time-varying rate with onion feedback
   - time_varying_k_combined: Time-varying rate with fixed bursts and onion feedback
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool, cpu_count
import functools

# Optional seaborn import
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Import optimization functions
from SimulationOptimization_join import run_optimization
from simulation_utils import load_experimental_data, get_parameter_bounds
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'SecondVersion'))
from MoMOptimization_join import joint_objective, get_mechanism_info
from scipy.optimize import differential_evolution


def get_parameter_count(mechanism):
    """
    Get the number of parameters for each mechanism.
    
    Args:
        mechanism (str): Mechanism name
    
    Returns:
        int: Number of parameters
    """
    # Parameter counts based on the current implementation
    param_counts = {
        # Constant rate mechanisms (MoM-based) - actual parameter counts from get_mechanism_info
        'simple': 11,  # n2, N2, k, r21, r23, R21, R23, alpha, beta_k, beta2_k, beta3_k
        'fixed_burst': 12,  # adds burst_size
        'feedback_onion': 12,  # adds n_inner
        'fixed_burst_feedback_onion': 13,  # adds burst_size, n_inner
        
        # Time-varying rate mechanisms (simulation-based)
        'time_varying_k': 12,  # n2, N2, k_max, tau, r21, r23, R21, R23, alpha, beta_k, beta_tau, beta_tau2
        'time_varying_k_fixed_burst': 13,  # adds burst_size
        'time_varying_k_feedback_onion': 13,  # adds n_inner
        'time_varying_k_combined': 14,  # adds burst_size, n_inner
    }
    
    return param_counts.get(mechanism, 0)


def get_total_data_points(datasets):
    """
    Calculate total number of data points across all datasets.
    
    Args:
        datasets (dict): Experimental datasets
    
    Returns:
        int: Total number of data points
    """
    total_points = 0
    for dataset_name, data_dict in datasets.items():
        total_points += len(data_dict['delta_t12']) + len(data_dict['delta_t32'])
    
    return total_points


def get_mom_data_points(datasets):
    """
    Calculate total data points for MoM optimization (now includes Velcade).
    MoM mechanisms now use all 5 datasets.
    """
    mom_datasets = ['wildtype', 'threshold', 'degrade', 'degradeAPC', 'velcade']
    total_points = 0
    for dataset_name in mom_datasets:
        if dataset_name in datasets:
            data_dict = datasets[dataset_name]
            total_points += len(data_dict['delta_t12']) + len(data_dict['delta_t32'])
    return total_points


def run_mom_optimization(mechanism, datasets, max_iterations=200, selected_strains=None):
    """
    Wrapper function for MoM optimization to match the interface expected by the comparison script.
    Now uses all 5 datasets including Velcade for MoM mechanisms.
    
    Args:
        mechanism (str): Mechanism name
        datasets (dict): Experimental datasets
        max_iterations (int): Maximum iterations for optimization
        selected_strains (list): List of strain names (not used, kept for interface compatibility)
    
    Returns:
        dict: Results dictionary matching the simulation optimization interface
    """
    try:
        import pandas as pd
        
        # Load data in the format expected by MoM optimization (original 4 datasets only)
        df = pd.read_excel("Data/All_strains_SCStimes.xlsx")
        data_wt12 = df['wildtype12'].dropna().values
        data_wt32 = df['wildtype32'].dropna().values
        data_threshold12 = df['threshold12'].dropna().values
        data_threshold32 = df['threshold32'].dropna().values
        data_degrate12 = df['degRade12'].dropna().values
        data_degrate32 = df['degRade32'].dropna().values
        # Handle missing initialProteins columns
        data_initial12 = df['initialProteins12'].dropna().values if 'initialProteins12' in df.columns else np.array([])
        data_initial32 = df['initialProteins32'].dropna().values if 'initialProteins32' in df.columns else np.array([])
        data_degrateAPC12 = df['degRadeAPC12'].dropna().values
        data_degrateAPC32 = df['degRadeAPC32'].dropna().values
        # Load Velcade data for MoM optimization
        data_velcade12 = df['degRadeVel12'].dropna().values
        data_velcade32 = df['degRadeVel32'].dropna().values
        
        # Get mechanism info
        gamma_mode = 'separate'  # Use separate gamma for each chromosome
        mechanism_info = get_mechanism_info(mechanism, gamma_mode)
        bounds = mechanism_info['bounds']
        
        # Run optimization using updated joint_objective (5 datasets including Velcade)
        result = differential_evolution(
            joint_objective,
            bounds,
            args=(mechanism, mechanism_info, data_wt12, data_wt32,
                  data_threshold12, data_threshold32,
                  data_degrate12, data_degrate32,
                  data_initial12, data_initial32,
                  data_degrateAPC12, data_degrateAPC32,
                  data_velcade12, data_velcade32),
            maxiter=max_iterations,
            popsize=15,
            seed=42,
            disp=False
        )
        
        # Convert to expected format
        return {
            'success': True,
            'converged': result.success,
            'nll': result.fun,
            'params': dict(zip(mechanism_info['params'], result.x)),
            'result': result,
            'message': result.message if not result.success else "Converged successfully"
        }
        
    except Exception as e:
        return {
            'success': False,
            'converged': False,
            'nll': np.inf,
            'params': {},
            'result': None,
            'message': f"MoM optimization failed: {e}"
        }


def calculate_aic_bic(nll, n_params, n_data):
    """
    Calculate AIC and BIC from negative log-likelihood.
    
    Args:
        nll (float): Negative log-likelihood
        n_params (int): Number of parameters
        n_data (int): Number of data points
    
    Returns:
        tuple: (AIC, BIC)
    """
    aic = 2 * n_params + 2 * nll
    bic = np.log(n_data) * n_params + 2 * nll
    return aic, bic


def run_single_optimization(args):
    """
    Run a single optimization for parallel processing.
    
    Args:
        args (tuple): (mechanism, datasets, num_simulations, run_number, seed)
    
    Returns:
        dict: Single optimization result
    """
    mechanism, datasets, num_simulations, run_number, seed = args
    
    try:
        # Set unique seed for each run
        np.random.seed(seed + run_number)
        
        # Choose optimization method based on mechanism type
        if mechanism.startswith('time_varying_k'):
            # Use simulation-based optimization for time-varying mechanisms
            result = run_optimization(
                mechanism, datasets, 
                max_iterations=200,
                num_simulations=num_simulations,
                selected_strains=None,
                use_parallel=False  # Disable nested parallelism
            )
        else:
            # Use MoM-based optimization for constant rate mechanisms
            result = run_mom_optimization(
                mechanism, datasets,
                max_iterations=200,
                selected_strains=None
            )
        
        return {
            'run_number': run_number,
            'success': result['success'],
            'converged': result.get('converged', False),
            'nll': result['nll'],
            'message': result.get('message', ''),
            'params': result.get('params', {})
        }
        
    except Exception as e:
        return {
            'run_number': run_number,
            'success': False,
            'converged': False,
            'nll': np.inf,
            'message': f"Error: {e}",
            'params': {}
        }


def run_mechanism_comparison(mechanism, datasets, num_runs=10, num_simulations=500, n_processes=None):
    """
    Run multiple optimization runs for a single mechanism and collect AIC/BIC statistics.
    Uses parallel processing for faster execution.
    
    Args:
        mechanism (str): Mechanism name
        datasets (dict): Experimental datasets
        num_runs (int): Number of optimization runs
        num_simulations (int): Number of simulations per evaluation
        n_processes (int): Number of parallel processes (None for auto-detect)
    
    Returns:
        dict: Results including AIC/BIC statistics
    """
    print(f"\n{'='*60}")
    print(f"Running comparison for: {mechanism.upper()}")
    print(f"{'='*60}")
    
    n_params = get_parameter_count(mechanism)
    # Use appropriate data point calculation based on mechanism type
    if mechanism.startswith('time_varying_k'):
        n_data = get_total_data_points(datasets)  # All 5 datasets
    else:
        n_data = get_mom_data_points(datasets)    # All 5 datasets now
    
    print(f"Parameters: {n_params}, Data points: {n_data}")
    
    # Determine number of processes
    if n_processes is None:
        n_processes = min(num_runs, cpu_count())
    
    print(f"Running {num_runs} optimization runs using {n_processes} parallel processes...")
    
    # Prepare arguments for parallel processing
    base_seed = 42
    args_list = [
        (mechanism, datasets, num_simulations, run_num, base_seed)
        for run_num in range(1, num_runs + 1)
    ]
    
    # Run optimizations in parallel
    aic_values = []
    bic_values = []
    nll_values = []
    converged_runs = 0
    failed_runs = 0
    
    try:
        with Pool(processes=n_processes) as pool:
            print(f"🚀 Starting parallel optimization with {n_processes} processes...")
            results = pool.map(run_single_optimization, args_list)
        
        # Process results
        for result in results:
            run_num = result['run_number']
            
            if result['success'] and result['converged']:
                nll = result['nll']
                aic, bic = calculate_aic_bic(nll, n_params, n_data)
                
                aic_values.append(aic)
                bic_values.append(bic)
                nll_values.append(nll)
                converged_runs += 1
                
                print(f"✅ Run {run_num} converged: NLL={nll:.2f}, AIC={aic:.2f}, BIC={bic:.2f}")
            else:
                failed_runs += 1
                print(f"❌ Run {run_num} failed: {result['message']}")
                
    except Exception as e:
        print(f"❌ Parallel processing error: {e}")
        print("Falling back to sequential processing...")
        
        # Fallback to sequential processing
        for args in args_list:
            result = run_single_optimization(args)
            run_num = result['run_number']
            
            if result['success'] and result['converged']:
                nll = result['nll']
                aic, bic = calculate_aic_bic(nll, n_params, n_data)
                
                aic_values.append(aic)
                bic_values.append(bic)
                nll_values.append(nll)
                converged_runs += 1
                
                print(f"✅ Run {run_num} converged: NLL={nll:.2f}, AIC={aic:.2f}, BIC={bic:.2f}")
            else:
                failed_runs += 1
                print(f"❌ Run {run_num} failed: {result['message']}")
    
    # Calculate statistics
    if aic_values:
        results = {
            'mechanism': mechanism,
            'n_params': n_params,
            'n_data': n_data,
            'converged_runs': converged_runs,
            'failed_runs': failed_runs,
            'convergence_rate': converged_runs / num_runs,
            'mean_nll': np.mean(nll_values),
            'std_nll': np.std(nll_values),
            'mean_aic': np.mean(aic_values),
            'std_aic': np.std(aic_values),
            'mean_bic': np.mean(bic_values),
            'std_bic': np.std(bic_values),
            'min_aic': np.min(aic_values),
            'min_bic': np.min(bic_values),
            'aic_values': aic_values,
            'bic_values': bic_values,
            'nll_values': nll_values
        }
        
        print(f"\n📊 Summary for {mechanism}:")
        print(f"  Convergence rate: {results['convergence_rate']*100:.1f}% ({converged_runs}/{num_runs})")
        print(f"  Mean AIC: {results['mean_aic']:.2f} ± {results['std_aic']:.2f}")
        print(f"  Mean BIC: {results['mean_bic']:.2f} ± {results['std_bic']:.2f}")
        print(f"  Mean NLL: {results['mean_nll']:.2f} ± {results['std_nll']:.2f}")
        
    else:
        print(f"❌ No successful runs for {mechanism}")
        results = {
            'mechanism': mechanism,
            'n_params': n_params,
            'n_data': n_data,
            'converged_runs': 0,
            'failed_runs': failed_runs,
            'convergence_rate': 0.0,
            'mean_nll': np.nan,
            'std_nll': np.nan,
            'mean_aic': np.nan,
            'std_aic': np.nan,
            'mean_bic': np.nan,
            'std_bic': np.nan,
            'min_aic': np.nan,
            'min_bic': np.nan,
            'aic_values': [],
            'bic_values': [],
            'nll_values': []
        }
    
    return results


def create_comparison_plots(all_results, save_plots=True):
    """
    Create visualization plots for model comparison.
    
    Args:
        all_results (list): List of results dictionaries
        save_plots (bool): Whether to save plots to files
    """
    # Filter out failed mechanisms
    successful_results = [r for r in all_results if r['converged_runs'] > 0]
    
    if not successful_results:
        print("No successful results to plot!")
        return
    
    # Create DataFrame for plotting
    plot_data = []
    for result in successful_results:
        mechanism = result['mechanism']
        for aic, bic in zip(result['aic_values'], result['bic_values']):
            plot_data.append({
                'mechanism': mechanism,
                'AIC': aic,
                'BIC': bic
            })
    
    df = pd.DataFrame(plot_data)
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Comparison: AIC and BIC Analysis', fontsize=16, y=0.98)
    
    # 1. Mean AIC comparison
    mean_data = pd.DataFrame(successful_results)
    mean_data = mean_data.sort_values('mean_aic')
    
    bars1 = ax1.bar(range(len(mean_data)), mean_data['mean_aic'], 
                    yerr=mean_data['std_aic'], capsize=5, alpha=0.7, color='skyblue')
    ax1.set_xlabel('Mechanism')
    ax1.set_ylabel('Mean AIC')
    ax1.set_title('Mean AIC Comparison (Lower is Better)')
    ax1.set_xticks(range(len(mean_data)))
    ax1.set_xticklabels(mean_data['mechanism'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val, std) in enumerate(zip(bars1, mean_data['mean_aic'], mean_data['std_aic'])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Mean BIC comparison
    mean_data_bic = mean_data.sort_values('mean_bic')
    
    bars2 = ax2.bar(range(len(mean_data_bic)), mean_data_bic['mean_bic'], 
                    yerr=mean_data_bic['std_bic'], capsize=5, alpha=0.7, color='lightcoral')
    ax2.set_xlabel('Mechanism')
    ax2.set_ylabel('Mean BIC')
    ax2.set_title('Mean BIC Comparison (Lower is Better)')
    ax2.set_xticks(range(len(mean_data_bic)))
    ax2.set_xticklabels(mean_data_bic['mechanism'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val, std) in enumerate(zip(bars2, mean_data_bic['mean_bic'], mean_data_bic['std_bic'])):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 3. AIC distribution boxplot
    if len(df) > 0:
        if HAS_SEABORN:
            sns.boxplot(data=df, x='mechanism', y='AIC', ax=ax3)
        else:
            # Fallback to matplotlib boxplot
            mechanisms_list = df['mechanism'].unique()
            aic_data = [df[df['mechanism'] == mech]['AIC'].values for mech in mechanisms_list]
            ax3.boxplot(aic_data, labels=mechanisms_list)
        ax3.set_title('AIC Distribution by Mechanism')
        ax3.set_xlabel('Mechanism')
        ax3.set_ylabel('AIC')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
    
    # 4. BIC distribution boxplot
    if len(df) > 0:
        if HAS_SEABORN:
            sns.boxplot(data=df, x='mechanism', y='BIC', ax=ax4)
        else:
            # Fallback to matplotlib boxplot
            mechanisms_list = df['mechanism'].unique()
            bic_data = [df[df['mechanism'] == mech]['BIC'].values for mech in mechanisms_list]
            ax4.boxplot(bic_data, labels=mechanisms_list)
        ax4.set_title('BIC Distribution by Mechanism')
        ax4.set_xlabel('Mechanism')
        ax4.set_ylabel('BIC')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    if save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'model_comparison_aic_bic_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as: {filename}")
    
    plt.show()


def create_summary_table(all_results, save_table=True):
    """
    Create and display a summary table of all results.
    
    Args:
        all_results (list): List of results dictionaries
        save_table (bool): Whether to save table to CSV
    """
    # Create summary DataFrame
    summary_data = []
    for result in all_results:
        summary_data.append({
            'Mechanism': result['mechanism'],
            'Parameters': result['n_params'],
            'Convergence Rate (%)': f"{result['convergence_rate']*100:.1f}",
            'Converged Runs': f"{result['converged_runs']}/10",
            'Mean NLL': f"{result['mean_nll']:.2f}" if not np.isnan(result['mean_nll']) else "N/A",
            'Mean AIC': f"{result['mean_aic']:.2f}" if not np.isnan(result['mean_aic']) else "N/A",
            'Std AIC': f"{result['std_aic']:.2f}" if not np.isnan(result['std_aic']) else "N/A",
            'Mean BIC': f"{result['mean_bic']:.2f}" if not np.isnan(result['mean_bic']) else "N/A",
            'Std BIC': f"{result['std_bic']:.2f}" if not np.isnan(result['std_bic']) else "N/A",
            'Min AIC': f"{result['min_aic']:.2f}" if not np.isnan(result['min_aic']) else "N/A",
            'Min BIC': f"{result['min_bic']:.2f}" if not np.isnan(result['min_bic']) else "N/A"
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Sort by mean AIC (best models first)
    successful_mask = df_summary['Mean AIC'] != "N/A"
    df_successful = df_summary[successful_mask].copy()
    df_failed = df_summary[~successful_mask].copy()
    
    if len(df_successful) > 0:
        df_successful['AIC_numeric'] = df_successful['Mean AIC'].astype(float)
        df_successful = df_successful.sort_values('AIC_numeric').drop('AIC_numeric', axis=1)
        df_summary = pd.concat([df_successful, df_failed], ignore_index=True)
    
    print(f"\n{'='*100}")
    print("MODEL COMPARISON SUMMARY TABLE")
    print(f"{'='*100}")
    print(df_summary.to_string(index=False))
    
    # Identify best models
    if len(df_successful) > 0:
        print(f"\n{'='*60}")
        print("BEST MODELS:")
        print(f"{'='*60}")
        
        best_aic_idx = df_successful['Mean AIC'].astype(float).idxmin()
        best_bic_idx = df_successful['Mean BIC'].astype(float).idxmin()
        
        print(f"🏆 Best AIC: {df_summary.loc[best_aic_idx, 'Mechanism']} (AIC = {df_summary.loc[best_aic_idx, 'Mean AIC']})")
        print(f"🏆 Best BIC: {df_summary.loc[best_bic_idx, 'Mechanism']} (BIC = {df_summary.loc[best_bic_idx, 'Mean BIC']})")
        
        # Check if same model wins both
        if best_aic_idx == best_bic_idx:
            print(f"🎯 CONSENSUS: {df_summary.loc[best_aic_idx, 'Mechanism']} is the best model by both AIC and BIC!")
    
    if save_table:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'model_comparison_summary_{timestamp}.csv'
        df_summary.to_csv(filename, index=False)
        print(f"\nSummary table saved as: {filename}")
    
    return df_summary


def main():
    """
    Main function to run the complete model comparison analysis.
    """
    print("="*80)
    print("CHROMOSOME SEGREGATION MODEL COMPARISON")
    print("AIC and BIC Analysis Across 8 Mechanisms")
    print("="*80)
    
    # Display system information
    n_cpus = cpu_count()
    print(f"\n💻 System Information:")
    print(f"   Available CPUs: {n_cpus}")
    print(f"   Parallel processing: ENABLED")
    
    # Load experimental data
    print("\n📊 Loading experimental data...")
    datasets = load_experimental_data()
    if not datasets:
        print("❌ Error: Could not load experimental data!")
        return
    
    print(f"✅ Loaded {len(datasets)} datasets: {list(datasets.keys())}")
    total_points = get_total_data_points(datasets)
    print(f"✅ Total data points: {total_points}")
    
    # Display dataset breakdown
    print(f"\n📈 Dataset breakdown:")
    for name, data in datasets.items():
        t12_points = len(data['delta_t12'])
        t32_points = len(data['delta_t32'])
        total = t12_points + t32_points
        print(f"   {name}: {total} points (T1-T2: {t12_points}, T3-T2: {t32_points})")
    
    # Define mechanisms to compare
    mechanisms = [
        # Constant rate mechanisms (MoM-based)
        'simple',
        'fixed_burst',
        'feedback_onion',
        'fixed_burst_feedback_onion',
        
        # Time-varying rate mechanisms (simulation-based)
        'time_varying_k',
        'time_varying_k_fixed_burst',
        'time_varying_k_feedback_onion', 
        'time_varying_k_combined'
    ]
    
    print(f"\n🔬 Comparing {len(mechanisms)} mechanisms:")
    for i, mech in enumerate(mechanisms, 1):
        param_count = get_parameter_count(mech)
        mech_type = "Simulation-based" if mech.startswith('time_varying_k') else "MoM-based"
        print(f"  {i}. {mech} ({param_count} parameters, {mech_type})")
    
    # Determine optimal number of processes per mechanism
    # Use fewer processes per mechanism to allow multiple mechanisms to run if needed
    processes_per_mechanism = max(1, min(10, n_cpus // 2))  # Reserve some CPUs
    
    print(f"\n⚙️  Parallel processing configuration:")
    print(f"   Processes per mechanism: {processes_per_mechanism}")
    print(f"   Runs per mechanism: 10")
    print(f"   Simulations per evaluation: 500")
    
    # Run comparison for each mechanism
    all_results = []
    start_time = datetime.now()
    
    for i, mechanism in enumerate(mechanisms, 1):
        mechanism_start = datetime.now()
        print(f"\n🚀 Progress: {i}/{len(mechanisms)} mechanisms")
        print(f"⏰ Started at: {mechanism_start.strftime('%H:%M:%S')}")
        
        try:
            result = run_mechanism_comparison(
                mechanism, datasets, 
                num_runs=10, 
                num_simulations=500,
                n_processes=processes_per_mechanism
            )
            all_results.append(result)
            
            mechanism_end = datetime.now()
            mechanism_duration = mechanism_end - mechanism_start
            print(f"⏱️  Mechanism {mechanism} completed in: {mechanism_duration}")
            
            # Estimate remaining time
            avg_time_per_mechanism = (mechanism_end - start_time) / i
            remaining_mechanisms = len(mechanisms) - i
            estimated_remaining = avg_time_per_mechanism * remaining_mechanisms
            estimated_completion = mechanism_end + estimated_remaining
            
            if remaining_mechanisms > 0:
                print(f"📅 Estimated completion: {estimated_completion.strftime('%H:%M:%S')} "
                      f"(~{estimated_remaining} remaining)")
            
        except Exception as e:
            print(f"❌ Error with {mechanism}: {e}")
            # Add failed result
            all_results.append({
                'mechanism': mechanism,
                'n_params': get_parameter_count(mechanism),
                'n_data': total_points,
                'converged_runs': 0,
                'failed_runs': 10,
                'convergence_rate': 0.0,
                'mean_nll': np.nan,
                'std_nll': np.nan,
                'mean_aic': np.nan,
                'std_aic': np.nan,
                'mean_bic': np.nan,
                'std_bic': np.nan,
                'min_aic': np.nan,
                'min_bic': np.nan,
                'aic_values': [],
                'bic_values': [],
                'nll_values': []
            })
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n⏱️  Total analysis time: {duration}")
    print(f"💻 Used {n_cpus} CPUs with parallel processing")
    
    # Calculate total optimization runs
    total_runs = len(mechanisms) * 10
    successful_runs = sum(r['converged_runs'] for r in all_results)
    overall_success_rate = (successful_runs / total_runs) * 100
    
    print(f"\n📈 Overall Statistics:")
    print(f"   Total optimization runs: {total_runs}")
    print(f"   Successful runs: {successful_runs}")
    print(f"   Overall success rate: {overall_success_rate:.1f}%")
    
    # Create summary table and plots
    print(f"\n📊 Creating summary and visualizations...")
    summary_df = create_summary_table(all_results, save_table=True)
    create_comparison_plots(all_results, save_plots=True)
    
    print(f"\n🎉 Model comparison analysis complete!")
    print(f"📁 Results saved with timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    print(f"🚀 Parallel processing utilized {processes_per_mechanism} processes per mechanism")


if __name__ == "__main__":
    main()
