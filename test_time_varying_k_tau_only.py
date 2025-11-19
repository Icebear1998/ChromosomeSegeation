#!/usr/bin/env python3
"""
Test time_varying_k mechanism by optimizing only tau parameter.
All other parameters are fixed from the optimized simple model.

Strategy:
1. Load pre-optimized simple model parameters
2. Create time_varying_k parameters with k_max = k from simple model
3. Optimize only tau using simulation-based optimization
4. Compare results to simple model baseline
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from simulation_utils import *
from Chromosomes_Theory import *
import warnings
warnings.filterwarnings('ignore')


def objective_tau_only(tau, fixed_params, mechanism, datasets, num_simulations=500):
    """
    Objective function that only varies tau, keeping all other parameters fixed.
    
    Args:
        tau (float or array): Single tau value to optimize
        fixed_params (dict): All other fixed parameters from simple model
        mechanism (str): 'time_varying_k'
        datasets (dict): Experimental datasets
        num_simulations (int): Number of simulations per evaluation
    
    Returns:
        float: Total negative log-likelihood
    """
    # Unpack tau (handle both scalar and array input)
    if isinstance(tau, np.ndarray):
        tau = tau[0]
    
    # Create full parameter vector for time_varying_k
    # Order: n2, N2, k_max, tau, r21, r23, R21, R23, alpha, beta_k, beta_tau, beta_tau2
    k_max = fixed_params['k']  # Use k from simple model as k_max
    k_1 = k_max / tau
    
    base_params = {
        'n1': fixed_params['n1'],
        'n2': fixed_params['n2'],
        'n3': fixed_params['n3'],
        'N1': fixed_params['N1'],
        'N2': fixed_params['N2'],
        'N3': fixed_params['N3'],
        'k_1': k_1,
        'k_max': k_max,
        'tau': tau
    }
    
    # Extract mutant parameters
    alpha = fixed_params['alpha']
    beta_k = fixed_params['beta_k']
    beta_tau = fixed_params['beta2_k']  # beta2_k affects tau
    beta_tau2 = fixed_params['beta3_k']  # beta3_k affects tau
    
    try:
        total_nll = 0
        
        for dataset_name, data_dict in datasets.items():
            # Apply mutant-specific modifications
            params, n0_list = apply_mutant_params(
                base_params, dataset_name, alpha, beta_k, beta_tau, beta_tau2
            )
            
            # Run simulations
            sim_delta_t12, sim_delta_t32 = run_simulation_for_dataset(
                mechanism, params, n0_list, num_simulations
            )
            
            if sim_delta_t12 is None or sim_delta_t32 is None:
                return 1e6
            
            # Extract experimental data
            exp_delta_t12 = data_dict['delta_t12']
            exp_delta_t32 = data_dict['delta_t32']
            
            # Create proper data dictionaries for likelihood calculation
            exp_data = {'delta_t12': exp_delta_t12, 'delta_t32': exp_delta_t32}
            sim_data = {'delta_t12': np.array(sim_delta_t12), 'delta_t32': np.array(sim_delta_t32)}
            
            nll_total_dataset = calculate_likelihood(exp_data, sim_data)
            
            if nll_total_dataset >= 1e6:
                return 1e6
            
            total_nll += nll_total_dataset
        
        return total_nll
    
    except Exception as e:
        print(f"Error in objective function: {e}")
        return 1e6


def optimize_tau_only(fixed_params, datasets, num_runs=5, num_simulations=500):
    """
    Optimize tau while keeping all other parameters fixed.
    
    Args:
        fixed_params (dict): Fixed parameters from simple model
        datasets (dict): Experimental datasets
        num_runs (int): Number of optimization runs with different tau initializations
        num_simulations (int): Number of simulations per evaluation
    
    Returns:
        dict: Best optimization results
    """
    print("="*80)
    print("OPTIMIZING TAU FOR TIME_VARYING_K MECHANISM")
    print("="*80)
    print(f"\nFixed parameters from simple model:")
    print(f"  n2={fixed_params['n2']:.2f}, N2={fixed_params['N2']:.1f}")
    print(f"  k={fixed_params['k']:.6f} (will be used as k_max)")
    print(f"  alpha={fixed_params['alpha']:.3f}, beta_k={fixed_params['beta_k']:.3f}")
    print(f"  beta2_k={fixed_params['beta2_k']:.3f}, beta3_k={fixed_params['beta3_k']:.3f}")
    
    print(f"\nRunning {num_runs} optimization attempts with different tau initializations...")
    print(f"Using {num_simulations} simulations per evaluation")
    
    # Define tau initial values to test
    tau_initial_values = [0.1, 1.0, 10.0, 50.0, 100.0]
    
    results = []
    
    for run in range(num_runs):
        initial_tau = tau_initial_values[run]
        
        print(f"\n{'='*70}")
        print(f"Run {run+1}/{num_runs}: Initial tau = {initial_tau:.1f}")
        print(f"{'='*70}")
        
        # Test initial tau
        initial_nll = objective_tau_only(
            initial_tau, fixed_params, 'time_varying_k', datasets, num_simulations=100
        )
        print(f"Initial NLL with tau={initial_tau:.1f}: {initial_nll:.2f}")
        
        # Optimize tau using differential_evolution
        print("Running optimization...")
        result = differential_evolution(
            objective_tau_only,
            bounds=[(0.1, 200.0)],  # tau bounds
            args=(fixed_params, 'time_varying_k', datasets, num_simulations),
            x0=[initial_tau],  # Initial guess
            maxiter=50,
            popsize=10,
            seed=42 + run,
            disp=True,
            atol=1e-6,
            tol=0.01
        )
        
        optimized_tau = result.x[0]
        nll = result.fun
        
        print(f"\n✓ Optimization completed")
        print(f"  Optimized tau: {optimized_tau:.4f}")
        print(f"  Final NLL: {nll:.6f}")
        print(f"  Converged: {result.success}")
        
        results.append({
            'run': run + 1,
            'initial_tau': initial_tau,
            'optimized_tau': optimized_tau,
            'nll': nll,
            'success': result.success,
            'k_max': fixed_params['k'],
            'k_1': fixed_params['k'] / optimized_tau
        })
    
    # Find best result
    valid_results = [r for r in results if r['success']]
    
    if not valid_results:
        print("\n❌ All optimization runs failed!")
        return None
    
    best_result = min(valid_results, key=lambda x: x['nll'])
    
    print(f"\n{'='*70}")
    print("OPTIMIZATION RESULTS SUMMARY")
    print(f"{'='*70}")
    
    print("\nAll runs:")
    for r in results:
        status = "✓" if r['success'] else "✗"
        print(f"  Run {r['run']}: tau={r['optimized_tau']:.4f}, NLL={r['nll']:.2f} {status}")
    
    print(f"\n{'='*70}")
    print("BEST RESULT:")
    print(f"{'='*70}")
    print(f"  Optimized tau: {best_result['optimized_tau']:.4f}")
    print(f"  k_max: {best_result['k_max']:.6f}")
    print(f"  k_1 (derived): {best_result['k_1']:.6f}")
    print(f"  Final NLL: {best_result['nll']:.6f}")
    
    return best_result


def main():
    """
    Main test routine.
    """
    print("="*80)
    print("TIME_VARYING_K: TAU-ONLY OPTIMIZATION TEST")
    print("="*80)
    print("\nTest if optimizing tau can improve upon simple model's fit")
    
    # Load experimental data
    print("\n📊 Loading experimental data...")
    datasets = load_experimental_data()
    if not datasets:
        print("Error: No datasets loaded!")
        return
    print("✅ Data loaded")
    
    # OPTION 1: Load pre-optimized simple model parameters from file
    # Uncomment and adjust the filename if you have saved parameters:
    # print("\n📁 Loading pre-optimized simple model parameters...")
    # with open('simulation_optimized_parameters_simple.txt', 'r') as f:
    #     # Parse parameters from file
    #     pass
    
    # OPTION 2: Use manually specified parameters from a previous run
    print("\n📋 Using pre-specified simple model parameters...")
    print("   (From previous MoM optimization)")
    
    # These are example values - replace with your actual optimized simple model parameters
    fixed_params = {
        'n2': 2.06,
        'N2': 934.43,
        'k': 0.053046,
        'r21': 0.46,
        'r23': 2.48,
        'R21': 0.51,
        'R23': 4.55,
        'alpha': 0.54,
        'beta_k': 0.45,
        'beta2_k': 4.0,   # beta_tau for APC mutant
        'beta3_k': 20.0,  # beta_tau2 for Velcade mutant
        # Derived parameters
        'n1': 0.46 * 2.06,
        'n3': 2.48 * 2.06,
        'N1': 0.51 * 934.43,
        'N3': 4.55 * 934.43
    }
    
    print(f"  n2={fixed_params['n2']:.2f}, N2={fixed_params['N2']:.1f}, k={fixed_params['k']:.6f}")
    
    # Baseline NLL (optional - can calculate with simple model)
    simple_nll = 6983.64  # Replace with actual value if known
    print(f"\nBaseline (simple model): NLL ≈ {simple_nll:.2f}")
    
    # Optimize tau
    print("\n" + "="*80)
    print("STARTING TAU OPTIMIZATION")
    print("="*80)
    
    best_result = optimize_tau_only(
        fixed_params, 
        datasets, 
        num_runs=5, 
        num_simulations=500
    )
    
    if best_result is None:
        print("\n❌ Optimization failed")
        return
    
    # Final comparison
    print("\n\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    improvement = simple_nll - best_result['nll']
    
    print(f"\nBaseline (simple):        NLL = {simple_nll:.2f}")
    print(f"Time_varying_k (optimized): NLL = {best_result['nll']:.2f}")
    print(f"Improvement:              {improvement:+.2f} NLL points")
    
    print(f"\nOptimal time-varying parameters:")
    print(f"  k_max: {best_result['k_max']:.6f}")
    print(f"  tau: {best_result['optimized_tau']:.4f}")
    print(f"  k_1: {best_result['k_1']:.6f}")
    
    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION:")
    print(f"{'='*70}")
    
    tau = best_result['optimized_tau']
    t_90 = 0.9 * tau
    
    print(f"\nDegradation rate function: k(t) = min(k_1 * t, k_max)")
    print(f"  where k_1 = k_max / tau = {best_result['k_1']:.6f}")
    print(f"  Time to reach 90% of k_max: {t_90:.2f} time units")
    
    if improvement > 1.0:
        print("\n✅ IMPROVEMENT: Time-varying degradation improves fit!")
        print("   This suggests the data shows evidence of time-dependent degradation.")
        
        if tau < 1.0:
            print("   Small tau → rapid ramp-up (nearly constant rate)")
        elif tau < 10.0:
            print("   Medium tau → moderate time-dependence")
        else:
            print("   Large tau → significant time-dependence")
    
    elif abs(improvement) < 1.0:
        print("\n✓ EQUIVALENT: Time-varying_k behaves like simple model.")
        print("  This suggests degradation rate is effectively constant.")
    
    else:
        print("\n⚠️  WORSE: Time-varying_k performs slightly worse.")
        print("  This might indicate overfitting or optimization issues.")
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    output_file = "time_varying_k_tau_optimization_results.txt"
    with open(output_file, 'w') as f:
        f.write("Time-Varying k: Tau-Only Optimization Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Simple model baseline NLL: {simple_nll:.6f}\n")
        f.write(f"Time-varying_k optimized NLL: {best_result['nll']:.6f}\n")
        f.write(f"Improvement: {improvement:+.6f}\n\n")
        f.write(f"Optimized tau: {best_result['optimized_tau']:.6f}\n")
        f.write(f"k_max (from simple k): {best_result['k_max']:.6f}\n")
        f.write(f"k_1 (derived): {best_result['k_1']:.6f}\n\n")
        f.write("Fixed parameters from simple model:\n")
        for key, value in fixed_params.items():
            f.write(f"  {key}: {value:.6f}\n")
    
    print(f"✅ Results saved to {output_file}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

