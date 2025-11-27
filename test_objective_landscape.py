#!/usr/bin/env python3
"""
Test script to analyze the continuity and noise of the simulation-based objective function.

Performs two tests:
1. Landscape Scan: Varies a parameter (e.g., k_max) and plots NLL to check for smoothness.
2. Noise Quantification: Evaluates NLL multiple times at the same point to measure stochastic noise.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import warnings
from tqdm import tqdm

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SecondVersion'))
from simulation_utils import load_experimental_data
from SimulationOptimization_join_GPUbase import joint_objective

# Suppress warnings
warnings.filterwarnings('ignore')

def get_base_params(mechanism):
    """Return a reasonable set of base parameters for testing."""
    # Using parameters close to what we've seen in previous runs
    if mechanism == 'time_varying_k':
        return [
            2.33,   # n2
            50.33,  # N2
            0.051,  # k_max
            30.0,   # tau
            0.46,   # r21
            2.48,   # r23
            0.51,   # R21
            4.55,   # R23
            0.54,   # alpha
            0.45,   # beta_k
            1.0,    # beta_tau
            1.0     # beta_tau2
        ]
    return None

def test_landscape_scan(mechanism, datasets, param_idx, param_name, range_min, range_max, num_points=20, num_sims=500):
    """
    Scan a parameter over a range and plot the NLL landscape.
    """
    print(f"\nRunning Landscape Scan for {param_name}...")
    print(f"Range: [{range_min}, {range_max}], Points: {num_points}, Sims: {num_sims}")
    
    base_params = get_base_params(mechanism)
    param_values = np.linspace(range_min, range_max, num_points)
    nll_values = []
    
    for val in tqdm(param_values):
        current_params = base_params.copy()
        current_params[param_idx] = val
        
        nll = joint_objective(current_params, mechanism, datasets, num_simulations=num_sims)
        nll_values.append(nll)
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, nll_values, 'o-', linewidth=2)
    plt.title(f'NLL Landscape: Varying {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('Negative Log-Likelihood')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'landscape_scan_{param_name}.png')
    print(f"Plot saved to landscape_scan_{param_name}.png")
    
    # Check for smoothness (simple metric: sum of absolute second derivatives)
    nll_array = np.array(nll_values)
    roughness = np.sum(np.abs(np.diff(nll_array, 2)))
    print(f"Roughness metric (lower is smoother): {roughness:.4f}")
    
    return param_values, nll_values

def test_noise_quantification(mechanism, datasets, num_repeats=20, num_sims=500):
    """
    Evaluate NLL multiple times at the same point to quantify noise.
    """
    print(f"\nRunning Noise Quantification...")
    print(f"Repeats: {num_repeats}, Sims: {num_sims}")
    
    base_params = get_base_params(mechanism)
    nll_values = []
    
    for _ in tqdm(range(num_repeats)):
        nll = joint_objective(base_params, mechanism, datasets, num_simulations=num_sims)
        nll_values.append(nll)
    
    nll_array = np.array(nll_values)
    mean_nll = np.mean(nll_array)
    std_nll = np.std(nll_array)
    cv_nll = (std_nll / mean_nll) * 100
    
    print(f"Mean NLL: {mean_nll:.4f}")
    print(f"Std Dev:  {std_nll:.4f}")
    print(f"CV:       {cv_nll:.4f}%")
    print(f"Range:    [{np.min(nll_array):.4f}, {np.max(nll_array):.4f}]")
    
    plt.figure(figsize=(10, 6))
    plt.hist(nll_values, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(mean_nll, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_nll:.2f}')
    plt.title(f'NLL Noise Distribution ({num_sims} sims)')
    plt.xlabel('Negative Log-Likelihood')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('nll_noise_distribution.png')
    print("Plot saved to nll_noise_distribution.png")
    
    return nll_values

def main():
    mechanism = 'time_varying_k'
    datasets = load_experimental_data()
    
    if not datasets:
        print("Error: No datasets loaded")
        return
    
    # 1. Landscape Scan for k_max (index 2)
    # Base is 0.051, scan around it
    test_landscape_scan(mechanism, datasets, 2, 'k_max', 0.04, 0.06, num_points=20, num_sims=1000)
    
    # 2. Landscape Scan for tau (index 3)
    # Base is 30.0, scan around it
    test_landscape_scan(mechanism, datasets, 3, 'tau', 20.0, 40.0, num_points=20, num_sims=1000)
    
    # 3. Noise Quantification
    test_noise_quantification(mechanism, datasets, num_repeats=30, num_sims=1000)

if __name__ == "__main__":
    main()
