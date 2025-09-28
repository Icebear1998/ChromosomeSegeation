#!/usr/bin/env python3
"""
Test script to compare three burst mechanisms: fixed_burst, random_normal_burst, and geometric_burst
using optimized parameters from fixed_burst_feedback_onion_join.txt
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import os
from collections import defaultdict
import time
import warnings

# Add the SecondVersion directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'SecondVersion'))

from MultiMechanismSimulation import MultiMechanismSimulation

def load_parameters(filename):
    """Load parameters from the optimized parameter file."""
    params = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        params[key] = float(value)
                    except ValueError:
                        params[key] = value
    return params

def run_burst_comparison():
    """Compare the three burst mechanisms using optimized parameters."""
    
    print("=" * 80)
    print("Burst Mechanisms Comparison Test")
    print("=" * 80)
    
    # Load optimized parameters
    param_file = 'SecondVersion/optimized_parameters_fixed_burst_feedback_onion_join.txt'
    if not os.path.exists(param_file):
        print(f"❌ Parameter file not found: {param_file}")
        return False
    
    params = load_parameters(param_file)
    print(f"✓ Loaded parameters from {param_file}")
    
    # Extract simulation parameters
    initial_state_list = [params['N1'], params['N2'], params['N3']]
    n0_list = [params['n1'], params['n2'], params['n3']]
    k = params['k']
    burst_size = params['burst_size']
    max_time = 1000.0  # Generous max time
    
    print(f"\nSimulation Parameters:")
    print(f"  Initial states: [{initial_state_list[0]:.1f}, {initial_state_list[1]:.1f}, {initial_state_list[2]:.1f}]")
    print(f"  Thresholds: [{n0_list[0]:.1f}, {n0_list[1]:.1f}, {n0_list[2]:.1f}]")
    print(f"  Base rate k: {k:.6f}")
    print(f"  Burst size: {burst_size:.2f}")
    
    # Define mechanisms to test
    mechanisms = {
        'fixed_burst': {
            'rate_params': {'k': k, 'burst_size': burst_size},
            'description': 'Fixed burst size'
        },
        'random_normal_burst': {
            'rate_params': {'k': k, 'burst_size': burst_size, 'var_burst_size': (burst_size * 0.3)**2},
            'description': 'Normal distribution (CV=0.3)'
        },
        'geometric_burst': {
            'rate_params': {'k': k, 'burst_size': burst_size},
            'description': 'Geometric distribution'
        }
    }
    
    # Run simulations
    num_runs = 1000  # Number of simulation runs per mechanism
    results = {}
    
    print(f"\nRunning {num_runs} simulations for each mechanism...")
    print("-" * 60)
    
    for mechanism_name, mechanism_info in mechanisms.items():
        print(f"\nTesting {mechanism_name} ({mechanism_info['description']})...")
        
        separation_times = []
        final_times = []
        num_events = []
        run_times = []
        
        for run in range(num_runs):
            start_time = time.time()
            
            try:
                sim = MultiMechanismSimulation(
                    mechanism=mechanism_name,
                    initial_state_list=initial_state_list.copy(),
                    rate_params=mechanism_info['rate_params'].copy(),
                    n0_list=n0_list.copy(),
                    max_time=max_time
                )
                
                times, states, separate_times = sim.simulate()
                
                separation_times.append(separate_times)
                final_times.append(times[-1])
                num_events.append(len(times))
                run_times.append(time.time() - start_time)
                
            except Exception as e:
                print(f"    ❌ Run {run+1} failed: {e}")
                continue
        
        if separation_times:
            # Calculate statistics
            sep_times_array = np.array(separation_times)
            time_diffs_12 = sep_times_array[:, 0] - sep_times_array[:, 1]  # T1 - T2
            time_diffs_32 = sep_times_array[:, 2] - sep_times_array[:, 1]  # T3 - T2
            
            results[mechanism_name] = {
                'separation_times': separation_times,
                'final_times': final_times,
                'num_events': num_events,
                'run_times': run_times,
                'time_diffs_12': time_diffs_12,
                'time_diffs_32': time_diffs_32,
                'description': mechanism_info['description']
            }
            
            print(f"  ✓ Completed {len(separation_times)} successful runs")
            print(f"    Mean final time: {np.mean(final_times):.2f} ± {np.std(final_times):.2f}")
            print(f"    Mean events: {np.mean(num_events):.1f} ± {np.std(num_events):.1f}")
            print(f"    Mean T1-T2: {np.mean(time_diffs_12):.2f} ± {np.std(time_diffs_12):.2f}")
            print(f"    Mean T3-T2: {np.mean(time_diffs_32):.2f} ± {np.std(time_diffs_32):.2f}")
            print(f"    Avg runtime: {np.mean(run_times)*1000:.1f} ms per simulation")
        else:
            print(f"  ❌ All runs failed for {mechanism_name}")
    
    return results

def create_comparison_plots(results):
    """Create comparison plots for the different mechanisms in a 2x2 layout."""
    
    if len(results) < 2:
        print("❌ Need at least 2 successful mechanisms to create plots")
        return
    
    print(f"\nCreating comparison plots...")
    
    # Set up the 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Burst Mechanisms Comparison', fontsize=16, fontweight='bold')
    
    mechanisms = list(results.keys())
    colors = ['blue', 'red', 'green', 'orange', 'purple'][:len(mechanisms)]
    
    # Plot 1 (Top Left): Time differences T1-T2 histogram
    ax = axes[0, 0]
    for i, (mech, data) in enumerate(results.items()):
        ax.hist(data['time_diffs_12'], bins=20, alpha=0.4, color=colors[i], 
               label=f"{mech} (μ={np.mean(data['time_diffs_12']):.2f})", density=True)
    ax.set_xlabel('T1 - T2 (time units)')
    ax.set_ylabel('Density')
    ax.set_title('Time Difference: Chromosome 1 - 2')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2 (Top Right): Time differences T3-T2 histogram
    ax = axes[0, 1]
    for i, (mech, data) in enumerate(results.items()):
        ax.hist(data['time_diffs_32'], bins=20, alpha=0.4, color=colors[i],
               label=f"{mech} (μ={np.mean(data['time_diffs_32']):.2f})", density=True)
    ax.set_xlabel('T3 - T2 (time units)')
    ax.set_ylabel('Density')
    ax.set_title('Time Difference: Chromosome 3 - 2')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3 (Bottom Left): T1-T2 boxplot
    ax = axes[1, 0]
    time_diffs_12_data = [data['time_diffs_12'] for data in results.values()]
    mechanism_names = list(results.keys())
    # Suppress matplotlib deprecation warning for labels parameter
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=matplotlib.MatplotlibDeprecationWarning)
        bp = ax.boxplot(time_diffs_12_data, labels=mechanism_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    ax.set_ylabel('T1 - T2 (time units)')
    ax.set_title('Time Difference: Chromosome 1 - 2 (Boxplot)')
    ax.grid(True, alpha=0.3)
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 4 (Bottom Right): T3-T2 boxplot
    ax = axes[1, 1]
    time_diffs_32_data = [data['time_diffs_32'] for data in results.values()]
    # Suppress matplotlib deprecation warning for labels parameter
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=matplotlib.MatplotlibDeprecationWarning)
        bp = ax.boxplot(time_diffs_32_data, labels=mechanism_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    ax.set_ylabel('T3 - T2 (time units)')
    ax.set_title('Time Difference: Chromosome 3 - 2 (Boxplot)')
    ax.grid(True, alpha=0.3)
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = 'burst_mechanisms_comparison.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved as {plot_filename}")
    
    # Show the plot
    plt.show()

def print_detailed_comparison(results):
    """Print detailed statistical comparison."""
    
    print("\n" + "=" * 80)
    print("DETAILED STATISTICAL COMPARISON")
    print("=" * 80)
    
    mechanisms = list(results.keys())
    
    # Compare time differences
    print(f"\n📊 Time Difference Statistics (T1-T2):")
    print("-" * 50)
    for mech, data in results.items():
        td12 = data['time_diffs_12']
        print(f"{mech:20s}: μ={np.mean(td12):6.2f}, σ={np.std(td12):6.2f}, "
              f"min={np.min(td12):6.2f}, max={np.max(td12):6.2f}")
    
    print(f"\n📊 Time Difference Statistics (T3-T2):")
    print("-" * 50)
    for mech, data in results.items():
        td32 = data['time_diffs_32']
        print(f"{mech:20s}: μ={np.mean(td32):6.2f}, σ={np.std(td32):6.2f}, "
              f"min={np.min(td32):6.2f}, max={np.max(td32):6.2f}")
    
    # Compare efficiency
    print(f"\n⚡ Computational Efficiency:")
    print("-" * 50)
    for mech, data in results.items():
        events = data['num_events']
        runtime = np.array(data['run_times']) * 1000  # Convert to ms
        print(f"{mech:20s}: {np.mean(events):6.0f} events, {np.mean(runtime):6.1f} ms runtime")
    
    # Statistical tests (if scipy is available)
    try:
        from scipy import stats
        print(f"\n🔬 Statistical Tests (p-values for difference in T1-T2 means):")
        print("-" * 60)
        
        for i, mech1 in enumerate(mechanisms):
            for j, mech2 in enumerate(mechanisms[i+1:], i+1):
                data1 = results[mech1]['time_diffs_12']
                data2 = results[mech2]['time_diffs_12']
                
                # Welch's t-test (unequal variances)
                t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                
                print(f"{mech1:15s} vs {mech2:15s}: p = {p_value:.4f} {significance}")
    
    except ImportError:
        print(f"\n📝 Note: Install scipy for statistical significance tests")

def main():
    """Main function to run the burst mechanism comparison."""
    
    print("Starting burst mechanisms comparison test...")
    
    # Run the comparison
    results = run_burst_comparison()
    
    if not results:
        print("❌ No successful results to analyze")
        return False
    
    # Print detailed comparison
    print_detailed_comparison(results)
    
    # Create plots
    try:
        create_comparison_plots(results)
    except Exception as e:
        print(f"⚠️  Could not create plots: {e}")
        print("   (matplotlib might not be available)")
    
    print(f"\n🎉 Burst mechanisms comparison completed successfully!")
    print(f"   Tested {len(results)} mechanisms with detailed statistical analysis")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
