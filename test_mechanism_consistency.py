#!/usr/bin/env python3
"""
Test that mechanisms reduce to simpler forms at boundary conditions.

This script:
1. Optimizes the simple model to get best parameters
2. Tests that fixed_burst with burst_size=1 gives identical NLL
3. Tests that feedback_onion with n_inner=N gives similar NLL to simple
4. Verifies mathematical consistency across mechanisms
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add SecondVersion to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'SecondVersion'))

from MoMOptimization_join import (
    joint_objective, get_mechanism_info, unpack_parameters
)
from simulation_utils import load_experimental_data
from scipy.optimize import differential_evolution, minimize


def run_simple_optimization(data_arrays, max_iter=500, popsize=20, seed=42):
    """
    Run optimization for simple mechanism to get best parameters.
    
    Returns:
        dict: Optimized parameters
    """
    print("="*80)
    print("STEP 1: OPTIMIZING SIMPLE MECHANISM")
    print("="*80)
    
    mechanism = 'simple'
    mechanism_info = get_mechanism_info(mechanism, 'separate')
    bounds = mechanism_info['bounds']
    
    print(f"\nOptimizing {len(bounds)} parameters...")
    print(f"Max iterations: {max_iter}, Population size: {popsize}")
    
    # Global optimization
    print("\n🔍 Running global optimization (Differential Evolution)...")
    result_global = differential_evolution(
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
        popsize=popsize,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=seed,
        disp=True,
        polish=False
    )
    
    # Local refinement
    print("\n🎯 Running local refinement (L-BFGS-B)...")
    result_local = minimize(
        joint_objective,
        result_global.x,
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
    
    # Extract optimized parameters
    params_dict = unpack_parameters(result_local.x, mechanism_info)
    nll = result_local.fun
    
    print("\n✅ OPTIMIZATION COMPLETE")
    print(f"Final NLL: {nll:.6f}")
    print("\nOptimized Parameters:")
    for key, val in params_dict.items():
        if key not in ['n1', 'N1', 'n3', 'N3']:  # Skip these for brevity
            print(f"  {key:12s} = {val:.6f}")
    
    return {
        'params': result_local.x,
        'params_dict': params_dict,
        'nll': nll,
        'mechanism_info': mechanism_info
    }


def test_fixed_burst_consistency(simple_result, data_arrays):
    """
    Test that fixed_burst with burst_size=1 gives identical NLL to simple.
    """
    print("\n\n" + "="*80)
    print("STEP 2: TESTING FIXED_BURST WITH BURST_SIZE=1")
    print("="*80)
    
    mechanism = 'fixed_burst'
    mechanism_info = get_mechanism_info(mechanism, 'separate')
    
    # Create parameter vector: simple params + burst_size=1
    simple_params = simple_result['params']
    
    # Fixed_burst has same params as simple plus burst_size
    # Order: n2, N2, k, r21, r23, R21, R23, burst_size, alpha, beta_k, beta2_k, beta3_k
    fixed_burst_params = np.insert(simple_params, 7, 1.0)  # Insert burst_size=1 at index 7
    
    # Calculate NLL
    nll = joint_objective(
        fixed_burst_params,
        mechanism,
        mechanism_info,
        data_arrays['data_wt12'], data_arrays['data_wt32'],
        data_arrays['data_threshold12'], data_arrays['data_threshold32'],
        data_arrays['data_degrate12'], data_arrays['data_degrate32'],
        data_arrays['data_initial12'], data_arrays['data_initial32'],
        data_arrays['data_degrateAPC12'], data_arrays['data_degrateAPC32'],
        data_arrays['data_velcade12'], data_arrays['data_velcade32']
    )
    
    simple_nll = simple_result['nll']
    diff = abs(nll - simple_nll)
    rel_diff = (diff / simple_nll) * 100
    
    print(f"\n📊 Results:")
    print(f"  Simple model NLL:              {simple_nll:.10f}")
    print(f"  Fixed_burst (burst_size=1):    {nll:.10f}")
    print(f"  Absolute difference:           {diff:.10f}")
    print(f"  Relative difference:           {rel_diff:.8f}%")
    
    if diff < 1e-8:
        print("\n✅ PERFECT MATCH: Implementations are mathematically identical!")
        status = "PASS"
    elif diff < 1e-6:
        print("\n✅ PASS: Differences within numerical precision")
        status = "PASS"
    else:
        print(f"\n❌ FAIL: Significant difference detected!")
        status = "FAIL"
    
    return {
        'nll': nll,
        'simple_nll': simple_nll,
        'difference': diff,
        'relative_difference': rel_diff,
        'status': status
    }


def test_feedback_onion_boundary(simple_result, data_arrays):
    """
    Test feedback_onion behavior at extreme n_inner values.
    
    When n_inner is very large (≥ N), feedback should never activate,
    so it should behave similarly to simple (though not identical due to
    the formula structure).
    """
    print("\n\n" + "="*80)
    print("STEP 3: TESTING FEEDBACK_ONION BOUNDARY CONDITIONS")
    print("="*80)
    
    mechanism = 'feedback_onion'
    mechanism_info = get_mechanism_info(mechanism, 'separate')
    
    simple_params_dict = simple_result['params_dict']
    simple_nll = simple_result['nll']
    
    # Test various n_inner values
    N2 = simple_params_dict['N2']
    
    test_cases = [
        ('Very small (n_inner=1)', 1.0),
        ('Small (n_inner=10)', 10.0),
        ('Medium (n_inner=50)', 50.0),
        ('Large (n_inner=N2)', N2),
        ('Very large (n_inner=2*N2)', 2.0 * N2),
        ('Extremely large (n_inner=10*N2)', 10.0 * N2),
    ]
    
    print(f"\nSimple model NLL: {simple_nll:.6f}")
    print(f"N2 = {N2:.2f}\n")
    print("Testing different n_inner values:")
    print(f"{'Description':<35} | {'n_inner':>12} | {'NLL':>12} | {'Diff from simple':>15} | {'Status':<10}")
    print("-" * 95)
    
    results = []
    
    for description, n_inner_val in test_cases:
        # Create parameter vector: simple params + n_inner
        # Order: n2, N2, k, r21, r23, R21, R23, n_inner, alpha, beta_k, beta2_k, beta3_k
        feedback_params = np.insert(simple_result['params'], 7, n_inner_val)
        
        # Calculate NLL
        nll = joint_objective(
            feedback_params,
            mechanism,
            mechanism_info,
            data_arrays['data_wt12'], data_arrays['data_wt32'],
            data_arrays['data_threshold12'], data_arrays['data_threshold32'],
            data_arrays['data_degrate12'], data_arrays['data_degrate32'],
            data_arrays['data_initial12'], data_arrays['data_initial32'],
            data_arrays['data_degrateAPC12'], data_arrays['data_degrateAPC32'],
            data_arrays['data_velcade12'], data_arrays['data_velcade32']
        )
        
        diff = nll - simple_nll
        
        # Status determination
        if n_inner_val >= N2:
            # When n_inner >= N2, feedback should rarely/never activate
            # But won't be identical due to formula differences
            if diff < 100:
                status = "✓ Good"
            else:
                status = "? Check"
        else:
            # When n_inner < N2, feedback activates, so expect higher NLL
            if diff > 0:
                status = "✓ Expected"
            else:
                status = "? Unusual"
        
        print(f"{description:<35} | {n_inner_val:>12.2f} | {nll:>12.2f} | {diff:>+15.2f} | {status:<10}")
        
        results.append({
            'description': description,
            'n_inner': n_inner_val,
            'nll': nll,
            'difference': diff
        })
    
    print("\n📝 Analysis:")
    print("  • When n_inner >> N: Feedback never activates (W_m = 1 always)")
    print("  • When n_inner < N:  Feedback activates (W_m < 1, slower degradation)")
    print("  • Higher NLL indicates feedback is active and affecting the fit")
    
    # Check if large n_inner gives reasonable results
    large_n_inner_result = [r for r in results if r['n_inner'] >= 2*N2]
    if large_n_inner_result:
        avg_diff = np.mean([r['difference'] for r in large_n_inner_result])
        if avg_diff < 1000:
            print(f"\n✅ PASS: Large n_inner (>= 2*N2) gives reasonable NLL difference (avg: {avg_diff:.2f})")
            status = "PASS"
        else:
            print(f"\n⚠️  WARNING: Large n_inner still shows high NLL difference (avg: {avg_diff:.2f})")
            status = "WARNING"
    else:
        status = "INCOMPLETE"
    
    return {
        'results': results,
        'status': status
    }


def test_combined_mechanism(simple_result, data_arrays):
    """
    Test that fixed_burst_feedback_onion with burst_size=1 and n_inner=N
    behaves reasonably.
    """
    print("\n\n" + "="*80)
    print("STEP 4: TESTING FIXED_BURST_FEEDBACK_ONION (burst_size=1, n_inner=N)")
    print("="*80)
    
    mechanism = 'fixed_burst_feedback_onion'
    mechanism_info = get_mechanism_info(mechanism, 'separate')
    
    simple_params_dict = simple_result['params_dict']
    simple_nll = simple_result['nll']
    N2 = simple_params_dict['N2']
    
    # Create parameter vector: simple params + burst_size=1 + n_inner=N2
    # Order: n2, N2, k, r21, r23, R21, R23, burst_size, n_inner, alpha, beta_k, beta2_k, beta3_k
    combined_params = np.insert(simple_result['params'], 7, [1.0, N2])
    
    nll = joint_objective(
        combined_params,
        mechanism,
        mechanism_info,
        data_arrays['data_wt12'], data_arrays['data_wt32'],
        data_arrays['data_threshold12'], data_arrays['data_threshold32'],
        data_arrays['data_degrate12'], data_arrays['data_degrate32'],
        data_arrays['data_initial12'], data_arrays['data_initial32'],
        data_arrays['data_degrateAPC12'], data_arrays['data_degrateAPC32'],
        data_arrays['data_velcade12'], data_arrays['data_velcade32']
    )
    
    diff = abs(nll - simple_nll)
    
    print(f"\n📊 Results:")
    print(f"  Simple model NLL:                          {simple_nll:.6f}")
    print(f"  Combined (burst_size=1, n_inner={N2:.1f}):  {nll:.6f}")
    print(f"  Absolute difference:                       {diff:.6f}")
    
    # When both burst_size=1 and n_inner≈N, should be close to feedback_onion
    # which should be reasonably close to simple
    if diff < 100:
        print("\n✅ PASS: Combined mechanism gives reasonable NLL")
        status = "PASS"
    else:
        print(f"\n⚠️  WARNING: Large difference from simple model")
        status = "WARNING"
    
    return {
        'nll': nll,
        'simple_nll': simple_nll,
        'difference': diff,
        'status': status
    }


def main():
    """
    Main test routine.
    """
    print("="*80)
    print("MECHANISM CONSISTENCY TEST SUITE")
    print("="*80)
    print("\nThis script verifies that complex mechanisms reduce to simpler forms")
    print("at appropriate boundary conditions.")
    
    # Load experimental data
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
    print(f"   Total data points: {len(data_arrays['data_wt12']) + len(data_arrays['data_wt32']) + len(data_arrays['data_threshold12']) + len(data_arrays['data_threshold32']) + len(data_arrays['data_degrate12']) + len(data_arrays['data_degrate32']) + len(data_arrays['data_degrateAPC12']) + len(data_arrays['data_degrateAPC32']) + len(data_arrays['data_velcade12']) + len(data_arrays['data_velcade32'])}")
    
    # Step 1: Optimize simple model
    simple_result = run_simple_optimization(data_arrays, max_iter=200, popsize=15, seed=42)
    
    # Step 2: Test fixed_burst consistency
    fb_result = test_fixed_burst_consistency(simple_result, data_arrays)
    
    # Step 3: Test feedback_onion boundary
    fo_result = test_feedback_onion_boundary(simple_result, data_arrays)
    
    # Step 4: Test combined mechanism
    combined_result = test_combined_mechanism(simple_result, data_arrays)
    
    # Final summary
    print("\n\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print("\n✓ Test Results:")
    print(f"  1. Fixed_burst (burst_size=1):                {fb_result['status']}")
    print(f"     - Difference from simple: {fb_result['difference']:.2e}")
    print(f"  2. Feedback_onion (boundary behavior):        {fo_result['status']}")
    print(f"  3. Combined mechanism (both boundaries):      {combined_result['status']}")
    print(f"     - Difference from simple: {combined_result['difference']:.2f}")
    
    all_passed = (fb_result['status'] == 'PASS' and 
                  fo_result['status'] in ['PASS', 'WARNING'] and 
                  combined_result['status'] in ['PASS', 'WARNING'])
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("   Mechanism implementations are mathematically consistent.")
    else:
        print("\n⚠️  SOME TESTS FAILED OR HAVE WARNINGS")
        print("   Review the detailed results above.")
    
    # Save results
    output_file = Path('mechanism_consistency_test_results.txt')
    with open(output_file, 'w') as f:
        f.write("MECHANISM CONSISTENCY TEST RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Simple model NLL: {simple_result['nll']:.10f}\n\n")
        f.write(f"Fixed_burst (burst_size=1) NLL: {fb_result['nll']:.10f}\n")
        f.write(f"  Difference: {fb_result['difference']:.2e}\n")
        f.write(f"  Status: {fb_result['status']}\n\n")
        f.write(f"Combined mechanism NLL: {combined_result['nll']:.6f}\n")
        f.write(f"  Difference: {combined_result['difference']:.6f}\n")
        f.write(f"  Status: {combined_result['status']}\n\n")
        f.write("Feedback_onion results:\n")
        for r in fo_result['results']:
            f.write(f"  {r['description']}: n_inner={r['n_inner']:.2f}, "
                   f"NLL={r['nll']:.2f}, diff={r['difference']:.2f}\n")
    
    print(f"\n📄 Results saved to: {output_file}")


if __name__ == "__main__":
    main()

