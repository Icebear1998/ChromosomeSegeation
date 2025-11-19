#!/usr/bin/env python3
"""
Test if complex mechanisms can improve upon simple model's optimized parameters.

Strategy:
1. Optimize simple model to get baseline parameters
2. For fixed_burst: Fix all simple parameters, optimize only burst_size
3. For feedback_onion: Fix all simple parameters, optimize only n_inner
4. For combined: Fix all simple parameters, optimize only burst_size and n_inner

Expected: Complex mechanisms should achieve NLL ≤ simple model's NLL
(since they have simple as a special case: burst_size=1, n_inner→∞)
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'SecondVersion'))

from MoMOptimization_join import (
    joint_objective, get_mechanism_info, unpack_parameters
)
from simulation_utils import load_experimental_data
from scipy.optimize import differential_evolution, minimize


def optimize_simple_model(data_arrays, max_iter=300, seed=42):
    """
    Optimize simple model to get baseline.
    """
    print("="*80)
    print("STEP 1: OPTIMIZE SIMPLE MODEL (BASELINE)")
    print("="*80)
    
    mechanism = 'simple'
    mechanism_info = get_mechanism_info(mechanism, 'separate')
    bounds = mechanism_info['bounds']
    
    print(f"\nOptimizing {len(bounds)} parameters...")
    
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
    
    return {
        'params_vector': result.x,
        'params_dict': params_dict,
        'nll': nll,
        'mechanism_info': mechanism_info
    }


def optimize_fixed_burst_only(simple_result, data_arrays, num_runs=5):
    """
    Fix simple parameters, optimize only burst_size.
    """
    print("\n" + "="*80)
    print("STEP 2: OPTIMIZE FIXED_BURST (BURST_SIZE ONLY)")
    print("="*80)
    print("Strategy: Use simple model's parameters, optimize only burst_size")
    
    mechanism = 'fixed_burst'
    mechanism_info = get_mechanism_info(mechanism, 'separate')
    
    # Get simple parameters
    simple_params = simple_result['params_vector']
    simple_nll = simple_result['nll']
    
    print(f"\nBaseline (simple): NLL = {simple_nll:.6f}")
    print(f"Running {num_runs} optimization attempts with different initializations...")
    
    results = []
    
    for run in range(num_runs):
        print(f"\n--- Run {run+1}/{num_runs} ---")
        
        # Create parameter vector: insert burst_size at position 7
        # Order: n2, N2, k, r21, r23, R21, R23, [burst_size], alpha, beta_k, beta2_k, beta3_k
        
        # Start with different initial burst_size values
        if run == 0:
            initial_burst = 1.0  # Start at boundary (should give simple)
        elif run == 1:
            initial_burst = 2.0  # Small burst
        elif run == 2:
            initial_burst = 5.0  # Medium burst
        elif run == 3:
            initial_burst = 10.0  # Large burst
        else:
            initial_burst = np.random.uniform(1.0, 20.0)  # Random
        
        initial_params = np.insert(simple_params, 7, initial_burst)
        
        # Only optimize burst_size (index 7)
        # Fix all other parameters by using very tight bounds
        bounds = []
        for i, val in enumerate(initial_params):
            if i == 7:  # burst_size
                bounds.append((1.0, 50.0))  # Allow optimization
            else:
                bounds.append((val - 1e-6, val + 1e-6))  # Fix parameter
        
        print(f"  Initial burst_size: {initial_burst:.2f}")
        
        # Optimize
        result = minimize(
            joint_objective,
            initial_params,
            args=(mechanism, mechanism_info,
                  data_arrays['data_wt12'], data_arrays['data_wt32'],
                  data_arrays['data_threshold12'], data_arrays['data_threshold32'],
                  data_arrays['data_degrate12'], data_arrays['data_degrate32'],
                  data_arrays['data_initial12'], data_arrays['data_initial32'],
                  data_arrays['data_degrateAPC12'], data_arrays['data_degrateAPC32'],
                  data_arrays['data_velcade12'], data_arrays['data_velcade32']),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        optimized_burst = result.x[7]
        nll = result.fun
        improvement = simple_nll - nll
        
        print(f"  Optimized burst_size: {optimized_burst:.4f}")
        print(f"  NLL: {nll:.6f} (improvement: {improvement:+.6f})")
        
        results.append({
            'initial_burst': initial_burst,
            'optimized_burst': optimized_burst,
            'nll': nll,
            'improvement': improvement
        })
    
    # Find best result
    best_result = min(results, key=lambda x: x['nll'])
    
    print(f"\n{'='*70}")
    print("FIXED_BURST RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Baseline (simple):     NLL = {simple_nll:.6f}")
    print(f"Best (fixed_burst):    NLL = {best_result['nll']:.6f}")
    print(f"Improvement:           {best_result['improvement']:+.6f}")
    print(f"Best burst_size:       {best_result['optimized_burst']:.4f}")
    
    if best_result['improvement'] > 0.1:
        print(f"\n✅ IMPROVEMENT: Fixed_burst mechanism improves fit!")
    elif abs(best_result['improvement']) < 0.1:
        print(f"\n✓ EQUIVALENT: Fixed_burst converges to simple (burst_size ≈ 1)")
    else:
        print(f"\n❌ WORSE: Fixed_burst performs worse (possible optimization issue)")
    
    return best_result


def optimize_feedback_onion_only(simple_result, data_arrays, num_runs=5):
    """
    Fix simple parameters, optimize only n_inner.
    """
    print("\n" + "="*80)
    print("STEP 3: OPTIMIZE FEEDBACK_ONION (N_INNER ONLY)")
    print("="*80)
    print("Strategy: Use simple model's parameters, optimize only n_inner")
    
    mechanism = 'feedback_onion'
    mechanism_info = get_mechanism_info(mechanism, 'separate')
    
    simple_params = simple_result['params_vector']
    simple_nll = simple_result['nll']
    simple_params_dict = simple_result['params_dict']
    
    print(f"\nBaseline (simple): NLL = {simple_nll:.6f}")
    print(f"N2 = {simple_params_dict['N2']:.2f} (reference for n_inner)")
    print(f"Running {num_runs} optimization attempts with different initializations...")
    
    results = []
    
    for run in range(num_runs):
        print(f"\n--- Run {run+1}/{num_runs} ---")
        
        # Create parameter vector: insert n_inner at position 7
        # Order: n2, N2, k, r21, r23, R21, R23, [n_inner], alpha, beta_k, beta2_k, beta3_k
        
        # Start with different initial n_inner values
        N2 = simple_params_dict['N2']
        if run == 0:
            initial_n_inner = 10.0 * N2  # Large (should give simple)
        elif run == 1:
            initial_n_inner = N2  # At N2
        elif run == 2:
            initial_n_inner = 0.5 * N2  # Below N2
        elif run == 3:
            initial_n_inner = 0.1 * N2  # Much below N2
        else:
            initial_n_inner = np.random.uniform(1.0, N2)  # Random
        
        initial_params = np.insert(simple_params, 7, initial_n_inner)
        
        # Only optimize n_inner (index 7)
        # NOTE: n_inner bound must be large enough to test "no feedback" case
        # where n_inner >> N (max N is ~1000, so we use 10000 as upper bound)
        bounds = []
        for i, val in enumerate(initial_params):
            if i == 7:  # n_inner
                bounds.append((1.0, 10000.0))  # Allow optimization with high upper bound
            else:
                bounds.append((val - 1e-6, val + 1e-6))  # Fix parameter
        
        print(f"  Initial n_inner: {initial_n_inner:.2f} ({initial_n_inner/N2:.2f} × N2)")
        
        # Optimize
        result = minimize(
            joint_objective,
            initial_params,
            args=(mechanism, mechanism_info,
                  data_arrays['data_wt12'], data_arrays['data_wt32'],
                  data_arrays['data_threshold12'], data_arrays['data_threshold32'],
                  data_arrays['data_degrate12'], data_arrays['data_degrate32'],
                  data_arrays['data_initial12'], data_arrays['data_initial32'],
                  data_arrays['data_degrateAPC12'], data_arrays['data_degrateAPC32'],
                  data_arrays['data_velcade12'], data_arrays['data_velcade32']),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        optimized_n_inner = result.x[7]
        nll = result.fun
        improvement = simple_nll - nll
        
        print(f"  Optimized n_inner: {optimized_n_inner:.4f} ({optimized_n_inner/N2:.2f} × N2)")
        print(f"  NLL: {nll:.6f} (improvement: {improvement:+.6f})")
        
        results.append({
            'initial_n_inner': initial_n_inner,
            'optimized_n_inner': optimized_n_inner,
            'nll': nll,
            'improvement': improvement
        })
    
    # Find best result
    best_result = min(results, key=lambda x: x['nll'])
    
    print(f"\n{'='*70}")
    print("FEEDBACK_ONION RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Baseline (simple):       NLL = {simple_nll:.6f}")
    print(f"Best (feedback_onion):   NLL = {best_result['nll']:.6f}")
    print(f"Improvement:             {best_result['improvement']:+.6f}")
    print(f"Best n_inner:            {best_result['optimized_n_inner']:.4f} ({best_result['optimized_n_inner']/N2:.2f} × N2)")
    
    if best_result['improvement'] > 0.1:
        print(f"\n✅ IMPROVEMENT: Feedback_onion mechanism improves fit!")
    elif abs(best_result['improvement']) < 0.1:
        print(f"\n✓ EQUIVALENT: Feedback_onion converges to simple (n_inner >> N)")
    else:
        print(f"\n❌ WORSE: Feedback_onion performs worse (possible optimization issue)")
    
    return best_result


def optimize_combined_only(simple_result, data_arrays, num_runs=5):
    """
    Fix simple parameters, optimize only burst_size and n_inner.
    """
    print("\n" + "="*80)
    print("STEP 4: OPTIMIZE COMBINED (BURST_SIZE + N_INNER)")
    print("="*80)
    print("Strategy: Use simple model's parameters, optimize burst_size and n_inner")
    
    mechanism = 'fixed_burst_feedback_onion'
    mechanism_info = get_mechanism_info(mechanism, 'separate')
    
    simple_params = simple_result['params_vector']
    simple_nll = simple_result['nll']
    simple_params_dict = simple_result['params_dict']
    N2 = simple_params_dict['N2']
    
    print(f"\nBaseline (simple): NLL = {simple_nll:.6f}")
    print(f"Running {num_runs} optimization attempts...")
    
    results = []
    
    for run in range(num_runs):
        print(f"\n--- Run {run+1}/{num_runs} ---")
        
        # Create parameter vector: insert burst_size and n_inner at positions 7 and 8
        # Order: n2, N2, k, r21, r23, R21, R23, [burst_size], [n_inner], alpha, beta_k, beta2_k, beta3_k
        
        if run == 0:
            initial_burst, initial_n_inner = 1.0, 10.0 * N2  # Boundary
        elif run == 1:
            initial_burst, initial_n_inner = 5.0, N2
        elif run == 2:
            initial_burst, initial_n_inner = 10.0, 0.5 * N2
        elif run == 3:
            initial_burst, initial_n_inner = 2.0, 0.2 * N2
        else:
            initial_burst = np.random.uniform(1.0, 20.0)
            initial_n_inner = np.random.uniform(1.0, N2)
        
        initial_params = np.insert(simple_params, 7, [initial_burst, initial_n_inner])
        
        # Optimize burst_size (7) and n_inner (8)
        # NOTE: n_inner bound must be large enough to test "no feedback" case
        bounds = []
        for i, val in enumerate(initial_params):
            if i == 7:  # burst_size
                bounds.append((1.0, 50.0))
            elif i == 8:  # n_inner
                bounds.append((1.0, 10000.0))  # High upper bound for testing
            else:
                bounds.append((val - 1e-6, val + 1e-6))  # Fix parameter
        
        print(f"  Initial: burst_size={initial_burst:.2f}, n_inner={initial_n_inner:.2f}")
        
        # Optimize
        result = minimize(
            joint_objective,
            initial_params,
            args=(mechanism, mechanism_info,
                  data_arrays['data_wt12'], data_arrays['data_wt32'],
                  data_arrays['data_threshold12'], data_arrays['data_threshold32'],
                  data_arrays['data_degrate12'], data_arrays['data_degrate32'],
                  data_arrays['data_initial12'], data_arrays['data_initial32'],
                  data_arrays['data_degrateAPC12'], data_arrays['data_degrateAPC32'],
                  data_arrays['data_velcade12'], data_arrays['data_velcade32']),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        optimized_burst = result.x[7]
        optimized_n_inner = result.x[8]
        nll = result.fun
        improvement = simple_nll - nll
        
        print(f"  Optimized: burst_size={optimized_burst:.4f}, n_inner={optimized_n_inner:.4f}")
        print(f"  NLL: {nll:.6f} (improvement: {improvement:+.6f})")
        
        results.append({
            'optimized_burst': optimized_burst,
            'optimized_n_inner': optimized_n_inner,
            'nll': nll,
            'improvement': improvement
        })
    
    best_result = min(results, key=lambda x: x['nll'])
    
    print(f"\n{'='*70}")
    print("COMBINED RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Baseline (simple):   NLL = {simple_nll:.6f}")
    print(f"Best (combined):     NLL = {best_result['nll']:.6f}")
    print(f"Improvement:         {best_result['improvement']:+.6f}")
    print(f"Best burst_size:     {best_result['optimized_burst']:.4f}")
    print(f"Best n_inner:        {best_result['optimized_n_inner']:.4f}")
    
    if best_result['improvement'] > 0.1:
        print(f"\n✅ IMPROVEMENT: Combined mechanism improves fit!")
    elif abs(best_result['improvement']) < 0.1:
        print(f"\n✓ EQUIVALENT: Combined converges to simple")
    else:
        print(f"\n❌ WORSE: Combined performs worse (possible optimization issue)")
    
    return best_result


def main():
    """
    Main test routine.
    """
    print("="*80)
    print("OPTIMIZATION IMPROVEMENT TEST")
    print("="*80)
    print("\nTest if complex mechanisms can improve upon simple model's fit")
    print("by optimizing only their additional parameters.")
    
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
    
    # Step 1: Optimize simple model
    simple_result = optimize_simple_model(data_arrays, max_iter=150, seed=42)
    
    # Step 2: Optimize fixed_burst (burst_size only)
    fb_result = optimize_fixed_burst_only(simple_result, data_arrays, num_runs=5)
    
    # Step 3: Optimize feedback_onion (n_inner only)
    fo_result = optimize_feedback_onion_only(simple_result, data_arrays, num_runs=5)
    
    # Step 4: Optimize combined (both parameters)
    combined_result = optimize_combined_only(simple_result, data_arrays, num_runs=5)
    
    # Final summary
    print("\n\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    simple_nll = simple_result['nll']
    
    print(f"\nBaseline (simple):             NLL = {simple_nll:.6f}")
    print(f"Fixed_burst (optimized):       NLL = {fb_result['nll']:.6f} ({fb_result['improvement']:+.6f})")
    print(f"Feedback_onion (optimized):    NLL = {fo_result['nll']:.6f} ({fo_result['improvement']:+.6f})")
    print(f"Combined (optimized):          NLL = {combined_result['nll']:.6f} ({combined_result['improvement']:+.6f})")
    
    print("\n" + "-"*80)
    print("INTERPRETATION:")
    print("-"*80)
    
    any_improvement = (fb_result['improvement'] > 0.1 or 
                       fo_result['improvement'] > 0.1 or 
                       combined_result['improvement'] > 0.1)
    
    if any_improvement:
        print("✅ At least one complex mechanism improves upon simple model!")
        print("   This suggests the data contains features that benefit from")
        print("   the additional parameters (burst dynamics or feedback).")
    else:
        print("✓ Complex mechanisms converge to simple model behavior.")
        print("  This suggests either:")
        print("  1. The data is well-explained by the simple constant-rate model")
        print("  2. The additional parameters don't significantly improve fit")
        print("  3. The optimized burst_size ≈ 1 and/or n_inner >> N")


if __name__ == "__main__":
    main()

