#!/usr/bin/env python3
"""
Test script to verify FastFeedbackSimulation against Gillespie algorithm.

Validates the "Sum of Waiting Times" method for feedback mechanisms.

Mechanisms tested:
1. feedback_onion (Constant k)
2. time_varying_k_feedback_onion (Time-varying k)

Expected outcome: Identical distributions, significant speedup (50-100x).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import time
import sys
import os

# Import simulations
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SecondVersion'))
from MultiMechanismSimulation import MultiMechanismSimulation
from MultiMechanismSimulationTimevary import MultiMechanismSimulationTimevary
from FastFeedbackSimulation import simulate_batch_feedback

def run_gillespie_simulations(mechanism, N_list, n_list, params, num_sims):
    """Run traditional Gillespie simulations."""
    print(f"\nRunning {num_sims} Gillespie simulations...")
    start_time = time.time()
    results = []
    
    is_simple = mechanism in ['feedback_onion', 'fixed_burst_feedback_onion']
    
    for _ in range(num_sims):
        if is_simple:
            # MultiMechanismSimulation expects 'simple' type rates
            rate_params = {
                'k': params['k'], 
                'n_inner': params['n_inner']
            }
            if 'burst_size' in params:
                rate_params['burst_size'] = params['burst_size']
                
            sim = MultiMechanismSimulation(
                mechanism=mechanism,
                initial_state_list=N_list,
                rate_params=rate_params,
                n0_list=n_list,
                max_time=10000.0
            )
        else:
            # Time varying
            rate_params = {
                'k_1': params['k_1'],
                'k_max': params['k_max'],
                'n_inner': params['n_inner']
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
    return np.array(results), elapsed

def run_feedback_simulations(mechanism, N_list, n_list, params, num_sims):
    """Run optimized Sum of Waiting Times simulations."""
    print(f"\nRunning {num_sims} Fast Feedback simulations...")
    start_time = time.time()
    
    # Extract params with defaults
    k = params.get('k')
    n_inner = params['n_inner']
    k_1 = params.get('k_1')
    k_max = params.get('k_max')
    burst_size = params.get('burst_size', 1.0)
        
    results = simulate_batch_feedback(
        mechanism=mechanism,
        initial_states=np.array(N_list),
        n0_lists=np.array(n_list),
        k=k,
        n_inner=n_inner,
        k_1=k_1,
        k_max=k_max,
        burst_size=burst_size,
        num_simulations=num_sims
    )
    
    elapsed = time.time() - start_time
    return results, elapsed

def compare_statistics(gillespie_results, beta_results, labels):
    """Compare statistical properties."""
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
        print(f"  Feedback:  Mean={b_mean:.2f}, Std={b_std:.2f}")
        print(f"  Difference: {mean_diff:.2f}% (mean), {std_diff:.2f}% (std)")
        
        if mean_diff > 5 or std_diff > 5:
            print(f"  ⚠ WARNING: Large difference detected!")
        else:
            print(f"  ✓ Excellent agreement")

def plot_comparison(gillespie_results, beta_results, labels, mechanism):
    """Create comparison plots."""
    print("\nGenerating comparison plots...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, (ax, label) in enumerate(zip(axes, labels)):
        g_data = gillespie_results[:, i]
        b_data = beta_results[:, i]
        
        ax.hist(g_data, bins=50, density=True, alpha=0.3, color='blue', label='Gillespie')
        ax.hist(b_data, bins=50, density=True, alpha=0.3, color='orange', label='Fast Feedback')
        
        try:
            g_kde = gaussian_kde(g_data)
            b_kde = gaussian_kde(b_data)
            x_min = min(g_data.min(), b_data.min())
            x_max = max(g_data.max(), b_data.max())
            x_grid = np.linspace(x_min, x_max, 200)
            ax.plot(x_grid, g_kde(x_grid), 'b-', lw=2, alpha=0.7, label='Gillespie KDE')
            ax.plot(x_grid, b_kde(x_grid), 'r--', lw=2, alpha=0.7, label='Feedback KDE')
        except:
            pass
            
        ax.set_title(label)
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(f'feedback_vs_gillespie_{mechanism}.png')
    print(f"  Plot saved to feedback_vs_gillespie_{mechanism}.png")

def test_mechanism(mechanism, params, N_list, n_list, num_simulations):
    """Test routine for a single mechanism."""
    print("\n" + "="*70)
    print(f"TESTING MECHANISM: {mechanism}")
    print("="*70)
    
    # Run
    g_res, g_time = run_gillespie_simulations(mechanism, N_list, n_list, params, num_simulations)
    f_res, f_time = run_feedback_simulations(mechanism, N_list, n_list, params, num_simulations)
    
    # Stats
    speedup = g_time / f_time
    print(f"\nSPEEDUP: {speedup:.1f}x")
    print(f"  Gillespie: {g_time:.2f}s")
    print(f"  Fast Feedback: {f_time:.2f}s")
    
    # Calculate Delta T values (T1-T2, T3-T2) from segregation times
    # g_res and f_res are (num_simulations, 3) arrays with [T1, T2, T3]
    print(f"\nCalculating Delta T values (T1-T2, T3-T2)...")
    g_delta_t12 = g_res[:, 0] - g_res[:, 1]  # T1 - T2
    g_delta_t32 = g_res[:, 2] - g_res[:, 1]  # T3 - T2
    
    f_delta_t12 = f_res[:, 0] - f_res[:, 1]  # T1 - T2
    f_delta_t32 = f_res[:, 2] - f_res[:, 1]  # T3 - T2
    
    # Stack Delta T values for comparison
    g_delta = np.column_stack([g_delta_t12, g_delta_t32])
    f_delta = np.column_stack([f_delta_t12, f_delta_t32])
    
    # Statistical comparison (use Delta T values)
    labels = ['ΔT₁₂ (Chr1 - Chr2)', 'ΔT₃₂ (Chr3 - Chr2)']
    compare_statistics(g_delta, f_delta, labels)
    plot_comparison(g_delta, f_delta, labels, mechanism)
    
    return speedup

def main():
    print("FAST FEEDBACK SIMULATION VALIDATION")
    
    num_sims = 5000
    N_list = [300.0, 400.0, 5000.0]
    n_list = [3.0, 5.0, 8.0]
    
    # Test 1: feedback_onion
    # params_onion = {'k': 0.05, 'n_inner': 50.0}
    # test_mechanism('feedback_onion', params_onion, N_list, n_list, num_sims)
    
    # # Test 2: time_varying_k_feedback_onion
    # params_tv_onion = {'k_1': 0.001, 'k_max': 0.05, 'n_inner': 50.0}
    # test_mechanism('time_varying_k_feedback_onion', params_tv_onion, N_list, n_list, num_sims)
    
    # Test 3: time_varying_k_combined
    # Test 4: fixed_burst_feedback_onion
    print("\n---------------------------------------------------")
    print("Test: fixed_burst_feedback_onion")
    params_fixed_onion = {'k': 0.05, 'n_inner': 50.0, 'burst_size': 5.0}
    test_mechanism('fixed_burst_feedback_onion', params_fixed_onion, N_list, n_list, num_sims)

if __name__ == "__main__":
    main()
