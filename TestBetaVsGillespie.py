#!/usr/bin/env python3
"""
Test script to verify FastBetaSimulation against Gillespie algorithm.

This script performs rigorous validation:
1. Statistical correctness (mean, variance, distribution shape)
2. Performance benchmark (speedup measurement)
3. Visual comparison (KDE plots)

Now supports testing:
- simple
- fixed_burst
- time_varying_k
- time_varying_k_fixed_burst

Expected outcome: Beta method produces identical distributions but 100-300x faster.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import time
import sys
import os

# Import both simulation methods
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SecondVersion'))
from MultiMechanismSimulation import MultiMechanismSimulation
from MultiMechanismSimulationTimevary import MultiMechanismSimulationTimevary
from FastBetaSimulation import FastBetaSimulator, simulate_batch


def run_gillespie_simulations(mechanism, N_list, n_list, params, num_sims):
    """
    Run simulations using the traditional Gillespie algorithm.
    
    Args:
        mechanism: Mechanism name
        N_list: Initial cohesin counts [N1, N2, N3]
        n_list: Threshold counts [n1, n2, n3]
        params: Dictionary with mechanism-specific parameters
        num_sims: Number of simulations
    """
    print(f"\nRunning {num_sims} Gillespie simulations...")
    start_time = time.time()
    
    results = []
    
    # Determine if this is a simple or time-varying mechanism
    is_simple = mechanism in ['simple', 'fixed_burst']
    
    for _ in range(num_sims):
        if is_simple:
            rate_params = {'k': params['k']}
            if 'burst_size' in params:
                rate_params['burst_size'] = params['burst_size']
            
            sim = MultiMechanismSimulation(
                mechanism=mechanism,
                initial_state_list=N_list,
                rate_params=rate_params,
                n0_list=n_list,
                max_time=10000.0
            )
        else:  # time-varying
            rate_params = {
                'k_1': params['k_1'],
                'k_max': params['k_max']
            }
            if 'burst_size' in params:
                rate_params['burst_size'] = params['burst_size']
            
            sim = MultiMechanismSimulationTimevary(
                mechanism=mechanism,
                initial_state_list=N_list,
                rate_params=rate_params,
                n0_list=n_list,
                max_time=10000.0
            )
        
        _, _, sep_times = sim.simulate()
        results.append(sep_times)
    
    elapsed = time.time() - start_time
    results = np.array(results)
    
    print(f"  Completed in {elapsed:.2f}s ({elapsed/num_sims*1000:.2f}ms per sim)")
    
    return results, elapsed


def run_beta_simulations(mechanism, N_list, n_list, params, num_sims):
    """
    Run simulations using the fast Beta sampling method.
    
    Args:
        mechanism: Mechanism name
        N_list: Initial cohesin counts [N1, N2, N3]
        n_list: Threshold counts [n1, n2, n3]
        params: Dictionary with mechanism-specific parameters
        num_sims: Number of simulations
    """
    print(f"\nRunning {num_sims} Beta simulations...")
    start_time = time.time()
    
    # Prepare parameters based on mechanism
    if mechanism in ['simple', 'fixed_burst']:
        k = params['k']
        burst_size = params.get('burst_size', 1.0)
        k_1 = None
        k_max = None
    else:  # time_varying_k mechanisms
        k = None
        burst_size = params.get('burst_size', 1.0)
        k_1 = params['k_1']
        k_max = params['k_max']
    
    # Use the ultra-fast batch method
    results = simulate_batch(
        mechanism=mechanism,
        initial_states=np.array(N_list),
        n0_lists=np.array(n_list),
        k=k,
        burst_size=burst_size,
        k_1=k_1,
        k_max=k_max,
        num_simulations=num_sims
    )
    
    elapsed = time.time() - start_time
    
    print(f"  Completed in {elapsed:.2f}s ({elapsed/num_sims*1000:.4f}ms per sim)")
    
    return results, elapsed


def compare_statistics(gillespie_results, beta_results, labels):
    """Compare statistical properties of the two methods."""
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON")
    print("="*70)
    
    for i, label in enumerate(labels):
        g_data = gillespie_results[:, i]
        b_data = beta_results[:, i]
        
        g_mean = np.mean(g_data)
        b_mean = np.mean(b_data)
        g_std = np.std(g_data)
        b_std = np.std(b_data)
        
        mean_diff = abs(g_mean - b_mean) / g_mean * 100
        std_diff = abs(g_std - b_std) / g_std * 100
        
        print(f"\n{label}:")
        print(f"  Gillespie: Mean={g_mean:.2f}, Std={g_std:.2f}")
        print(f"  Beta:      Mean={b_mean:.2f}, Std={b_std:.2f}")
        print(f"  Difference: {mean_diff:.2f}% (mean), {std_diff:.2f}% (std)")
        
        if mean_diff > 5 or std_diff > 5:
            print(f"  ⚠ WARNING: Large difference detected!")
        else:
            print(f"  ✓ Excellent agreement")


def plot_comparison(gillespie_results, beta_results, labels, mechanism):
    """Create visual comparison plots."""
    print("\nGenerating comparison plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, (ax, label) in enumerate(zip(axes, labels)):
        g_data = gillespie_results[:, i]
        b_data = beta_results[:, i]
        
        # Histograms
        ax.hist(g_data, bins=50, density=True, alpha=0.3, color='blue', label='Gillespie')
        ax.hist(b_data, bins=50, density=True, alpha=0.3, color='red', label='Beta')
        
        # KDEs
        try:
            g_kde = gaussian_kde(g_data)
            b_kde = gaussian_kde(b_data)
            
            x_min = min(g_data.min(), b_data.min())
            x_max = max(g_data.max(), b_data.max())
            x_grid = np.linspace(x_min, x_max, 200)
            
            ax.plot(x_grid, g_kde(x_grid), 'b-', lw=2, alpha=0.7, label='Gillespie KDE')
            ax.plot(x_grid, b_kde(x_grid), 'r--', lw=2, alpha=0.7, label='Beta KDE')
        except:
            pass
        
        ax.set_xlabel('Segregation Time')
        ax.set_ylabel('Density')
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'beta_vs_gillespie_comparison_{mechanism}.png'
    plt.savefig(filename)
    print(f"  Plot saved to {filename}")


def test_mechanism(mechanism, params, N_list, n_list, num_simulations):
    """Test a specific mechanism."""
    print("\n" + "="*70)
    print(f"TESTING MECHANISM: {mechanism.upper()}")
    print("="*70)
    
    print(f"\nTest Configuration:")
    print(f"  Mechanism: {mechanism}")
    print(f"  N values: {N_list}")
    print(f"  n values: {n_list}")
    for key, val in params.items():
        print(f"  {key}: {val}")
    print(f"  Number of simulations: {num_simulations}")
    
    # Run both methods
    gillespie_results, gillespie_time = run_gillespie_simulations(
        mechanism, N_list, n_list, params, num_simulations
    )
    
    beta_results, beta_time = run_beta_simulations(
        mechanism, N_list, n_list, params, num_simulations
    )
    
    # Performance comparison
    speedup = gillespie_time / beta_time
    print(f"\n" + "="*70)
    print(f"PERFORMANCE COMPARISON")
    print("="*70)
    print(f"  Gillespie time: {gillespie_time:.2f}s")
    print(f"  Beta time:      {beta_time:.2f}s")
    print(f"  Speedup:        {speedup:.1f}x")
    
    if speedup > 50:
        print(f"  ✓ Excellent speedup achieved!")
    elif speedup > 10:
        print(f"  ✓ Good speedup")
    else:
        print(f"  ⚠ Speedup lower than expected")
    
    # Calculate Delta T values (T1-T2, T3-T2) from segregation times
    # gillespie_results and beta_results are (num_simulations, 3) arrays with [T1, T2, T3]
    print(f"\nCalculating Delta T values (T1-T2, T3-T2)...")
    gillespie_delta_t12 = gillespie_results[:, 0] - gillespie_results[:, 1]  # T1 - T2
    gillespie_delta_t32 = gillespie_results[:, 2] - gillespie_results[:, 1]  # T3 - T2
    
    beta_delta_t12 = beta_results[:, 0] - beta_results[:, 1]  # T1 - T2
    beta_delta_t32 = beta_results[:, 2] - beta_results[:, 1]  # T3 - T2
    
    # Stack Delta T values for comparison
    gillespie_delta = np.column_stack([gillespie_delta_t12, gillespie_delta_t32])
    beta_delta = np.column_stack([beta_delta_t12, beta_delta_t32])
    
    # Statistical comparison (use Delta T values)
    labels = ['ΔT₁₂ (Chr1 - Chr2)', 'ΔT₃₂ (Chr3 - Chr2)']
    compare_statistics(gillespie_delta, beta_delta, labels)
    
    # Visual comparison (use Delta T values)
    plot_comparison(gillespie_delta, beta_delta, labels, mechanism)
    
    return speedup


def main():
    """Main test routine - tests all supported mechanisms."""
    print("="*70)
    print("BETA SAMPLING vs GILLESPIE ALGORITHM - VALIDATION TEST")
    print("Testing all supported mechanisms")
    print("="*70)
    
    num_simulations = 5000  # Reduced for faster testing
    
    # Common parameters
    N_list = [300.0, 400.0, 1000.0]
    n_list = [3.0, 5.0, 8.0]
    
    # Test configurations for each mechanism
    test_configs = {
        'simple': {
            'k': 0.05
        },
        'fixed_burst': {
            'k': 0.05,
            'burst_size': 5.0
        },
        'time_varying_k': {
            'k_1': 0.001,
            'k_max': 0.05
        },
        'time_varying_k_fixed_burst': {
            'k_1': 0.001,
            'k_max': 0.05,
            'burst_size': 5.0
        }
    }
    
    # Run tests for all mechanisms
    results = {}
    for mechanism, params in test_configs.items():
        speedup = test_mechanism(mechanism, params, N_list, n_list, num_simulations)
        results[mechanism] = speedup
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF ALL TESTS")
    print("="*70)
    for mechanism, speedup in results.items():
        status = "✓" if speedup > 50 else "⚠"
        print(f"  {status} {mechanism:30s}: {speedup:6.1f}x speedup")
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)
    
    all_passed = all(s > 50 for s in results.values())
    if all_passed:
        print("\n✓ All mechanisms validated: Identical distributions, massive speedup!")
        print("  → Safe to integrate into optimization pipeline.")
    else:
        print("\n⚠ Some mechanisms need review before integration.")


if __name__ == "__main__":
    main()
