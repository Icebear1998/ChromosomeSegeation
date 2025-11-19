#!/usr/bin/env python3
"""
Test that time_varying_k with small tau (rapid rise to k_max) behaves like constant rate.

When tau → 0:
- k_1 = k_max/tau → ∞
- k(t) rapidly reaches k_max and stays constant
- Should behave similar to constant rate simulation

Note: We can't compare simulation-based to MoM-based NLL directly,
but we can compare the distributions and see if they're similar.
"""

import numpy as np
import matplotlib.pyplot as plt
from MultiMechanismSimulationTimevary import MultiMechanismSimulationTimevary
from pathlib import Path


def run_simulations_with_varying_tau(base_params, tau_values, num_sims=100):
    """
    Run simulations with different tau values to see convergence to constant rate.
    
    Args:
        base_params: Base parameters (N, n, k_max)
        tau_values: List of tau values to test
        num_sims: Number of simulations per tau
    
    Returns:
        dict: Results for each tau value
    """
    results = {}
    
    for tau in tau_values:
        print(f"\nTesting tau = {tau:.6f}")
        
        k_1 = base_params['k_max'] / tau
        print(f"  k_1 = k_max/tau = {base_params['k_max']:.4f}/{tau:.6f} = {k_1:.2e}")
        
        rate_params = {
            'k_1': k_1,
            'k_max': base_params['k_max']
        }
        
        delta_t12_list = []
        delta_t32_list = []
        
        for _ in range(num_sims):
            sim = MultiMechanismSimulationTimevary(
                mechanism='time_varying_k',
                initial_state_list=base_params['N_list'],
                rate_params=rate_params,
                n0_list=base_params['n_list'],
                max_time=base_params['max_time']
            )
            
            times, states, separate_times = sim.simulate()
            
            if separate_times[0] is not None and separate_times[1] is not None:
                delta_t12 = separate_times[0] - separate_times[1]
                delta_t12_list.append(delta_t12)
            
            if separate_times[2] is not None and separate_times[1] is not None:
                delta_t32 = separate_times[2] - separate_times[1]
                delta_t32_list.append(delta_t32)
        
        results[tau] = {
            'delta_t12': np.array(delta_t12_list),
            'delta_t32': np.array(delta_t32_list),
            'k_1': k_1
        }
        
        print(f"  Completed {len(delta_t12_list)} simulations for delta_t12")
        print(f"  Mean delta_t12: {np.mean(delta_t12_list):.2f}, Std: {np.std(delta_t12_list):.2f}")
        print(f"  Mean delta_t32: {np.mean(delta_t32_list):.2f}, Std: {np.std(delta_t32_list):.2f}")
    
    return results


def plot_convergence(results, base_params):
    """
    Plot how distributions converge as tau → 0.
    """
    output_dir = Path('time_varying_boundary_analysis')
    output_dir.mkdir(exist_ok=True)
    
    tau_values = sorted(results.keys(), reverse=True)  # Largest to smallest
    
    # Plot 1: Delta_t12 distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, tau in enumerate(tau_values[:6]):  # Plot up to 6
        ax = axes[idx]
        data = results[tau]['delta_t12']
        
        ax.hist(data, bins=20, density=True, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(data), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(data):.2f}')
        ax.set_xlabel('Delta_t12 (min)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'tau = {tau:.4f} min\nk_1 = {results[tau]["k_1"]:.2e}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'delta_t12_convergence.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_dir}/delta_t12_convergence.png")
    plt.close()
    
    # Plot 2: Mean and Std vs tau
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    tau_list = sorted(tau_values)
    means_12 = [np.mean(results[tau]['delta_t12']) for tau in tau_list]
    stds_12 = [np.std(results[tau]['delta_t12']) for tau in tau_list]
    means_32 = [np.mean(results[tau]['delta_t32']) for tau in tau_list]
    stds_32 = [np.std(results[tau]['delta_t32']) for tau in tau_list]
    
    # Mean convergence
    ax1.semilogx(tau_list, means_12, 'o-', linewidth=2, markersize=8, label='Delta_t12', color='blue')
    ax1.semilogx(tau_list, means_32, 's-', linewidth=2, markersize=8, label='Delta_t32', color='green')
    ax1.axhline(means_12[-1], color='blue', linestyle='--', alpha=0.5, label=f'Limit (tau→0): {means_12[-1]:.2f}')
    ax1.axhline(means_32[-1], color='green', linestyle='--', alpha=0.5)
    ax1.set_xlabel('tau (min)', fontsize=12)
    ax1.set_ylabel('Mean Separation Time (min)', fontsize=12)
    ax1.set_title('Mean Convergence as tau → 0', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Std convergence
    ax2.semilogx(tau_list, stds_12, 'o-', linewidth=2, markersize=8, label='Delta_t12', color='blue')
    ax2.semilogx(tau_list, stds_32, 's-', linewidth=2, markersize=8, label='Delta_t32', color='green')
    ax2.set_xlabel('tau (min)', fontsize=12)
    ax2.set_ylabel('Std of Separation Time (min)', fontsize=12)
    ax2.set_title('Std Convergence as tau → 0', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_statistics.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/convergence_statistics.png")
    plt.close()
    
    # Plot 3: Overlay distributions for comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(tau_values)))
    
    for idx, tau in enumerate(reversed(tau_values)):  # Smallest tau last (darkest)
        data_12 = results[tau]['delta_t12']
        data_32 = results[tau]['delta_t32']
        
        ax1.hist(data_12, bins=20, density=True, alpha=0.5, color=colors[idx], 
                label=f'tau={tau:.4f}', edgecolor='black', linewidth=0.5)
        ax2.hist(data_32, bins=20, density=True, alpha=0.5, color=colors[idx],
                label=f'tau={tau:.4f}', edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Delta_t12 (min)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Delta_t12: Convergence to Constant Rate\n(Darker = smaller tau)', 
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Delta_t32 (min)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Delta_t32: Convergence to Constant Rate\n(Darker = smaller tau)', 
                 fontsize=13, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'distribution_overlay.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/distribution_overlay.png")
    plt.close()


def main():
    """
    Test boundary condition: tau → 0 should give constant-rate behavior.
    """
    print("="*80)
    print("TIME_VARYING_K BOUNDARY CONDITION TEST")
    print("="*80)
    print("\nTesting: As tau → 0, k(t) rapidly reaches k_max and stays constant")
    print("Expected: Distributions should converge to a limit")
    
    # Base parameters (similar to simple model test)
    base_params = {
        'N_list': [100.0, 100.0, 100.0],
        'n_list': [2.0, 2.0, 5.0],
        'k_max': 0.05,  # Constant rate to reach
        'max_time': 1000.0
    }
    
    print(f"\nBase parameters:")
    print(f"  N = {base_params['N_list']}")
    print(f"  n = {base_params['n_list']}")
    print(f"  k_max = {base_params['k_max']:.4f}")
    
    # Test different tau values (from large to small)
    # As tau decreases, k_1 increases, and k(t) reaches k_max faster
    tau_values = [100.0, 50.0, 20.0, 10.0, 5.0, 1.0, 0.1, 0.01]
    
    print(f"\nTesting {len(tau_values)} tau values: {tau_values}")
    print("Note: Smaller tau → larger k_1 → faster rise to k_max")
    
    # Run simulations
    print("\n" + "-"*80)
    print("Running simulations...")
    print("-"*80)
    results = run_simulations_with_varying_tau(base_params, tau_values, num_sims=200)
    
    # Analyze convergence
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS")
    print("="*80)
    
    tau_sorted = sorted(tau_values)
    print("\nMean Delta_t12 as tau decreases:")
    for tau in tau_sorted:
        mean_12 = np.mean(results[tau]['delta_t12'])
        print(f"  tau = {tau:8.4f}: mean = {mean_12:8.2f} min")
    
    # Check convergence
    means = [np.mean(results[tau]['delta_t12']) for tau in tau_sorted]
    convergence_diff = abs(means[-1] - means[-2])
    convergence_pct = (convergence_diff / means[-1]) * 100
    
    print(f"\nConvergence check (last two tau values):")
    print(f"  tau = {tau_sorted[-2]:.4f}: mean = {means[-2]:.2f}")
    print(f"  tau = {tau_sorted[-1]:.4f}: mean = {means[-1]:.2f}")
    print(f"  Difference: {convergence_diff:.2f} min ({convergence_pct:.2f}%)")
    
    if convergence_pct < 5.0:
        print(f"\n✅ CONVERGED: Difference < 5% indicates tau → 0 limit reached")
    else:
        print(f"\n⚠️  NOT FULLY CONVERGED: Consider testing even smaller tau values")
    
    # Generate plots
    print("\n" + "-"*80)
    print("Generating plots...")
    print("-"*80)
    plot_convergence(results, base_params)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nAs tau → 0:")
    print(f"  • k_1 = k_max/tau → ∞")
    print(f"  • k(t) = min(k_1*t, k_max) reaches k_max almost immediately")
    print(f"  • System behaves like constant rate k = k_max")
    print(f"\nLimiting behavior (tau = {tau_sorted[-1]:.4f}):")
    print(f"  • Mean Delta_t12: {means[-1]:.2f} min")
    print(f"  • Mean Delta_t32: {np.mean(results[tau_sorted[-1]]['delta_t32']):.2f} min")
    print(f"\nThis represents the behavior of time_varying_k when rate")
    print(f"becomes constant (k = k_max = {base_params['k_max']:.4f}) from t ≈ 0")
    
    print("\n📊 All plots saved to: time_varying_boundary_analysis/")
    print("\n✅ TEST COMPLETE")


if __name__ == "__main__":
    main()

