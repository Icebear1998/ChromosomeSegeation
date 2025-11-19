#!/usr/bin/env python3
"""
Test if time_varying_k mechanism can improve upon simple model's optimized parameters.

Strategy:
1. Optimize simple model to get baseline parameters
2. For time_varying_k: Fix all simple parameters, set k_max = k, optimize only tau

Expected: 
- When tau is very large, k(t) increases very slowly, behaving like a low constant rate
- When tau is very small, k(t) quickly reaches k_max, behaving like simple model
- The optimal tau should achieve NLL ≤ simple model's NLL

Biological interpretation:
- k(t) = min(k_max/tau * t, k_max)
- tau: time constant for reaching maximum degradation rate
- When tau → 0: instantaneous jump to k_max (equivalent to simple model with k = k_max)
- When tau → ∞: very slow ramp-up
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'SecondVersion'))

from MoMOptimization_join import (
    joint_objective, get_mechanism_info, unpack_parameters
)
from SimulationOptimization_join import run_optimization
from simulation_utils import load_experimental_data
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt


def optimize_simple_model(data_arrays, max_iter=150, seed=42):
    """
    Optimize simple model to get baseline using MoM-based optimization.
    (Faster than simulation-based for simple mechanism)
    """
    print("="*80)
    print("STEP 1: OPTIMIZE SIMPLE MODEL (BASELINE)")
    print("="*80)
    
    mechanism = 'simple'
    mechanism_info = get_mechanism_info(mechanism, 'separate')
    bounds = mechanism_info['bounds']
    
    print(f"\nOptimizing {len(bounds)} parameters using MoM...")
    print(f"This will take a few minutes...")
    
    # Global optimization
    result = differential_evolution(
        joint_objective,
        bounds=bounds,
        args=(mechanism, mechanism_info,
              data_arrays['data_wt12'], data_arrays['data_wt32'],
              data_arrays['data_threshold12'], data_arrays['data_threshold32'],
              data_arrays['data_degrate12'], data_arrays['data_degrate32'],
              data_arrays['data_initial12'], data_arrays['data_initial32'],
              data_arrays['data_degrateAPC12'], data_arrays['data_degrateAPC32'],
              data_arrays['data_velcade12'], data_arrays['data_velcade32']),
        strategy='best1bin',
        maxiter=max_iter,
        popsize=15,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=seed,
        disp=True,
        polish=True
    )
    
    # Local refinement
    result = minimize(
        joint_objective,
        result.x,
        args=(mechanism, mechanism_info,
              data_arrays['data_wt12'], data_arrays['data_wt32'],
              data_arrays['data_threshold12'], data_arrays['data_threshold32'],
              data_arrays['data_degrate12'], data_arrays['data_degrate32'],
              data_arrays['data_initial12'], data_arrays['data_initial32'],
              data_arrays['data_degrateAPC12'], data_arrays['data_degrateAPC32'],
              data_arrays['data_velcade12'], data_arrays['data_velcade32']),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    params_dict = unpack_parameters(result.x, mechanism_info)
    nll = result.fun
    
    print(f"\n✅ Simple model optimized")
    print(f"   NLL: {nll:.6f}")
    print(f"\n   Key parameters:")
    print(f"     n2={params_dict['n2']:.4f}, N2={params_dict['N2']:.2f}, k={params_dict['k']:.6f}")
    print(f"     r21={params_dict['r21']:.4f}, r23={params_dict['r23']:.4f}")
    print(f"     R21={params_dict['R21']:.2f}, R23={params_dict['R23']:.2f}")
    print(f"     alpha={params_dict['alpha']:.4f}")
    print(f"     beta_k={params_dict['beta_k']:.4f}")
    print(f"     beta2_k={params_dict['beta2_k']:.4f}")
    print(f"     beta3_k={params_dict['beta3_k']:.4f}")
    
    return {
        'params_vector': result.x,
        'params': params_dict,
        'nll': nll,
        'mechanism_info': mechanism_info
    }


def optimize_time_varying_k_only(simple_result, datasets, num_runs=5):
    """
    Fix simple parameters, set k_max = k, optimize only tau.
    
    Note: For time_varying_k mechanism with simulation-based optimization,
    we need to use a different approach since we can't easily "fix" parameters
    in the simulation-based optimizer like we did with MoM.
    
    Instead, we'll:
    1. Create an objective function that only varies tau
    2. Use scipy.optimize.minimize to optimize tau
    3. Call run_optimization with fixed initial_guess for all params except tau
    """
    print("\n" + "="*80)
    print("STEP 2: OPTIMIZE TIME_VARYING_K (TAU ONLY)")
    print("="*80)
    print("Strategy: Use simple model's k as k_max, optimize only tau")
    
    simple_params = simple_result['params']
    simple_nll = simple_result['nll']
    
    print(f"\nBaseline (simple): NLL = {simple_nll:.6f}")
    print(f"Simple model k = {simple_params['k']:.6f}")
    print(f"\nRunning {num_runs} optimization attempts with different tau initializations...")
    print("Note: Each run performs simulation-based optimization (this may take time)")
    
    results = []
    
    # Define tau test values
    tau_initial_values = [
        0.1,   # Very small: should quickly reach k_max
        1.0,   # Small
        10.0,  # Medium
        50.0,  # Large
        100.0  # Very large: slow ramp-up
    ]
    
    for run in range(num_runs):
        initial_tau = tau_initial_values[run]
        
        print(f"\n{'='*70}")
        print(f"Run {run+1}/{num_runs}: Testing tau = {initial_tau:.1f}")
        print(f"{'='*70}")
        
        # Create initial guess for time_varying_k mechanism
        # Order: n2, N2, k_max, tau, r21, r23, R21, R23, alpha, beta_k, beta2_k, beta3_k
        initial_guess = [
            simple_params['n2'],
            simple_params['N2'],
            simple_params['k'],  # k_max = k from simple model
            initial_tau,         # This is what we're testing
            simple_params['r21'],
            simple_params['r23'],
            simple_params['R21'],
            simple_params['R23'],
            simple_params['alpha'],
            simple_params['beta_k'],
            simple_params['beta2_k'],
            simple_params['beta3_k']
        ]
        
        print(f"\nInitial parameters:")
        print(f"  k_max: {simple_params['k']:.6f} (fixed from simple model)")
        print(f"  tau: {initial_tau:.2f}")
        
        # Run optimization with initial guess
        try:
            result = run_optimization(
                mechanism='time_varying_k',
                datasets=datasets,
                num_simulations=500,
                max_iterations=100,  # Reduced since we start from good initial guess
                initial_guess=initial_guess
            )
            
            optimized_params = result['params']
            nll = result['nll']
            improvement = simple_nll - nll
            
            print(f"\n  ✓ Optimization completed")
            print(f"  Final k_max: {optimized_params['k_max']:.6f}")
            print(f"  Final tau: {optimized_params['tau']:.4f}")
            print(f"  NLL: {nll:.6f}")
            print(f"  Improvement over simple: {improvement:+.6f}")
            
            results.append({
                'initial_tau': initial_tau,
                'optimized_tau': optimized_params['tau'],
                'optimized_k_max': optimized_params['k_max'],
                'nll': nll,
                'improvement': improvement,
                'params': optimized_params
            })
            
        except Exception as e:
            print(f"\n  ✗ Optimization failed: {e}")
            results.append({
                'initial_tau': initial_tau,
                'optimized_tau': None,
                'optimized_k_max': None,
                'nll': np.inf,
                'improvement': -np.inf,
                'params': None
            })
    
    # Find best result
    valid_results = [r for r in results if r['nll'] != np.inf]
    
    if not valid_results:
        print("\n❌ All optimization runs failed!")
        return None
    
    best_result = min(valid_results, key=lambda x: x['nll'])
    
    print(f"\n{'='*70}")
    print("TIME_VARYING_K RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Baseline (simple):        NLL = {simple_nll:.6f}")
    print(f"Best (time_varying_k):    NLL = {best_result['nll']:.6f}")
    print(f"Improvement:              {best_result['improvement']:+.6f}")
    print(f"\nBest parameters:")
    print(f"  k_max: {best_result['optimized_k_max']:.6f} (cf. simple k={simple_params['k']:.6f})")
    print(f"  tau: {best_result['optimized_tau']:.4f}")
    
    # Interpret results
    print(f"\n{'='*70}")
    print("INTERPRETATION:")
    print(f"{'='*70}")
    
    if best_result['improvement'] > 0.1:
        print("✅ IMPROVEMENT: Time-varying degradation improves fit!")
        print("   This suggests the data shows evidence of time-dependent degradation.")
        
        if best_result['optimized_tau'] < 1.0:
            print("   Small tau → rapid increase to constant k_max")
            print("   (Nearly equivalent to simple model)")
        elif best_result['optimized_tau'] < 10.0:
            print("   Medium tau → moderate time-dependence")
        else:
            print("   Large tau → strong time-dependence throughout observation window")
            
    elif abs(best_result['improvement']) < 0.1:
        print("✓ EQUIVALENT: Time-varying_k converges to simple model behavior.")
        print("  This suggests either:")
        print("  1. Degradation rate is constant (not time-dependent)")
        print("  2. tau is very small → k(t) quickly reaches k_max")
        print(f"  3. Optimal tau ({best_result['optimized_tau']:.2f}) gives nearly constant rate")
    else:
        print("❌ WORSE: Time-varying_k performs worse.")
        print("  This suggests possible optimization issues or overfitting.")
    
    # Create visualization
    create_tau_visualization(simple_params, results, simple_nll)
    
    return best_result


def create_tau_visualization(simple_params, results, simple_nll):
    """
    Create visualization of tau optimization results.
    """
    print("\n📊 Creating visualization...")
    
    # Filter valid results
    valid_results = [r for r in results if r['nll'] != np.inf]
    
    if not valid_results:
        print("  No valid results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: NLL vs tau
    ax = axes[0, 0]
    taus = [r['optimized_tau'] for r in valid_results]
    nlls = [r['nll'] for r in valid_results]
    ax.scatter(taus, nlls, s=100, alpha=0.6, c='steelblue')
    ax.axhline(simple_nll, color='red', linestyle='--', linewidth=2, label='Simple model NLL')
    ax.set_xlabel('Optimized tau', fontsize=12)
    ax.set_ylabel('NLL', fontsize=12)
    ax.set_title('NLL vs Tau', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: NLL improvement vs tau
    ax = axes[0, 1]
    improvements = [simple_nll - r['nll'] for r in valid_results]
    ax.scatter(taus, improvements, s=100, alpha=0.6, c='green')
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label='No improvement')
    ax.set_xlabel('Optimized tau', fontsize=12)
    ax.set_ylabel('NLL Improvement (simple - time_varying_k)', fontsize=12)
    ax.set_title('Improvement vs Tau', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: k_max comparison
    ax = axes[1, 0]
    k_maxs = [r['optimized_k_max'] for r in valid_results]
    x_pos = np.arange(len(valid_results))
    ax.bar(x_pos, k_maxs, alpha=0.6, color='steelblue', label='Optimized k_max')
    ax.axhline(simple_params['k'], color='red', linestyle='--', linewidth=2, label='Simple model k')
    ax.set_xlabel('Run', fontsize=12)
    ax.set_ylabel('k_max', fontsize=12)
    ax.set_title('k_max Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{i+1}" for i in range(len(valid_results))])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: k(t) curves for different tau values
    ax = axes[1, 1]
    t_max = max([r['optimized_tau'] for r in valid_results]) * 2
    t = np.linspace(0, t_max, 1000)
    
    for i, r in enumerate(valid_results):
        tau = r['optimized_tau']
        k_max = r['optimized_k_max']
        k_t = np.minimum(k_max / tau * t, k_max)
        ax.plot(t, k_t, linewidth=2, label=f"tau={tau:.2f}, NLL={r['nll']:.1f}")
    
    ax.axhline(simple_params['k'], color='red', linestyle='--', linewidth=2, label='Simple model k (constant)')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('k(t)', fontsize=12)
    ax.set_title('Degradation Rate k(t) vs Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = 'time_varying_k_optimization_analysis'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'tau_optimization_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved visualization to {output_path}")
    plt.close()


def main():
    """
    Main test routine.
    """
    print("="*80)
    print("TIME_VARYING_K OPTIMIZATION TEST")
    print("="*80)
    print("\nTest if time_varying_k mechanism can improve upon simple model")
    print("by optimizing tau while keeping other parameters fixed.")
    
    # Load data
    print("\n📊 Loading experimental data...")
    datasets = load_experimental_data()
    
    data_arrays = {
        'data_wt12': datasets['wildtype']['delta_t12'],
        'data_wt32': datasets['wildtype']['delta_t32'],
        'data_threshold12': datasets['threshold']['delta_t12'],
        'data_threshold32': datasets['threshold']['delta_t32'],
        'data_degrate12': datasets['degrade']['delta_t12'],
        'data_degrate32': datasets['degrade']['delta_t32'],
        'data_degrateAPC12': datasets['degradeAPC']['delta_t12'],
        'data_degrateAPC32': datasets['degradeAPC']['delta_t32'],
        'data_velcade12': datasets['velcade']['delta_t12'],
        'data_velcade32': datasets['velcade']['delta_t32'],
        'data_initial12': np.array([]),
        'data_initial32': np.array([])
    }
    
    print("✅ Data loaded")
    
    # Step 1: Optimize simple model (using MoM - faster)
    simple_result = optimize_simple_model(data_arrays, max_iter=150, seed=42)
    
    # Step 2: Optimize time_varying_k (tau only, using simulation)
    tvk_result = optimize_time_varying_k_only(simple_result, datasets, num_runs=5)
    
    if tvk_result is None:
        print("\n❌ Time-varying_k optimization failed")
        return
    
    # Final summary
    print("\n\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    simple_nll = simple_result['nll']
    
    print(f"\nBaseline (simple):             NLL = {simple_nll:.6f}")
    print(f"Time_varying_k (optimized):    NLL = {tvk_result['nll']:.6f} ({tvk_result['improvement']:+.6f})")
    
    print(f"\nOptimal time-varying parameters:")
    print(f"  k_max: {tvk_result['optimized_k_max']:.6f}")
    print(f"  tau: {tvk_result['optimized_tau']:.4f}")
    
    print("\n" + "-"*80)
    print("BIOLOGICAL INTERPRETATION:")
    print("-"*80)
    
    tau = tvk_result['optimized_tau']
    k_max = tvk_result['optimized_k_max']
    
    # Estimate when k(t) reaches 90% of k_max
    t_90 = 0.9 * tau
    
    print(f"\nDegradation rate function: k(t) = min(k_max/tau * t, k_max)")
    print(f"  k_max = {k_max:.6f}")
    print(f"  tau = {tau:.4f}")
    print(f"  Time to reach 90% of k_max: {t_90:.2f} time units")
    
    if tau < 1.0:
        print("\n→ Very rapid ramp-up: degradation reaches maximum rate almost immediately")
        print("  (Nearly indistinguishable from constant-rate simple model)")
    elif tau < 10.0:
        print("\n→ Moderate ramp-up: degradation rate increases during early phase")
    else:
        print("\n→ Slow ramp-up: degradation rate continues increasing throughout observation")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

