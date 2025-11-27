#!/usr/bin/env python3
import numpy as np
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from SimulationOptimization_join_GPUbase import run_optimization, MECHANISM_PARAM_NAMES, unpack_mechanism_params, joint_objective
from simulation_utils import apply_mutant_params, run_simulation_for_dataset

# Try to import GPU simulation function
try:
    from simulation_utils_gpu import run_simulation_for_dataset_gpu
    GPU_AVAILABLE = True
    print("GPU acceleration available and enabled.")
except ImportError:
    run_simulation_for_dataset_gpu = None
    GPU_AVAILABLE = False
    print("GPU acceleration NOT available. Falling back to CPU.")

def generate_synthetic_data(mechanism, true_params, num_simulations=1000):
    print(f"Generating synthetic data for mechanism: {mechanism}")
    datasets = {}
    strains = ['wildtype', 'threshold', 'degrade', 'degradeAPC', 'velcade']
    
    # Unpack true params to get base_params and mutant effects
    param_names = MECHANISM_PARAM_NAMES[mechanism]
    params_vector = [true_params[name] for name in param_names]
    
    base_params, alpha, beta_k, beta_tau, beta_tau2 = unpack_mechanism_params(params_vector, mechanism)
    
    for strain in strains:
        print(f"  Simulating {strain}...", end='', flush=True)
        # Apply mutant params
        params, n0_list = apply_mutant_params(base_params, strain, alpha, beta_k, beta_tau, beta_tau2)
        
        # Run simulation
        if GPU_AVAILABLE:
            delta_t12, delta_t32 = run_simulation_for_dataset_gpu(
                mechanism, params, n0_list, num_simulations=num_simulations, max_time=2000.0
            )
        else:
            delta_t12, delta_t32 = run_simulation_for_dataset(
                mechanism, params, n0_list, num_simulations=num_simulations
            )
        
        datasets[strain] = {
            'delta_t12': delta_t12,
            'delta_t32': delta_t32
        }
        print(f" Done. ({len(delta_t12)} samples)")
        
    return datasets

def main():
    mechanism = 'time_varying_k'
    
    # Define True Parameters
    # Using values that are physically reasonable for this system
    true_params = {
        'n2': 10.0,
        'N2': 100.0,
        'k_max': 0.05,
        'tau': 30.0,
        'r21': 1.0, 
        'r23': 1.0, 
        'R21': 1.0, 
        'R23': 1.0, 
        'alpha': 0.5, 
        'beta_k': 0.1, 
        'beta_tau': 2.0, 
        'beta_tau2': 3.0
    }
    
    print("True Parameters:")
    for k, v in true_params.items():
        print(f"  {k}: {v}")
    
    # Generate Data
    synthetic_datasets = generate_synthetic_data(mechanism, true_params, num_simulations=1000)
    
    # Calculate NLL of True Parameters
    param_names = MECHANISM_PARAM_NAMES[mechanism]
    true_params_vector = [true_params[name] for name in param_names]
    
    print("\nCalculating NLL of True Parameters on Synthetic Data...")
    true_nll = joint_objective(true_params_vector, mechanism, synthetic_datasets, num_simulations=1000)
    print(f"True Parameters NLL: {true_nll:.4f}")
    
    # Run Optimization
    print("\nRunning Optimization to recover parameters...")
    print("This may take a few minutes...")
    
    # We use a smaller number of iterations for the test
    results = run_optimization(
        mechanism, 
        synthetic_datasets, 
        max_iterations=5, 
        num_simulations=200,
        popsize=3,
        workers=1
    )
    
    # Report Results
    print("\n" + "="*60)
    print(f"{'Parameter':<15} {'True Value':<15} {'Recovered':<15} {'Error %':<15}")
    print("-" * 60)
    
    recovered = results['params']
    total_error = 0
    for name in param_names:
        true_val = true_params[name]
        rec_val = recovered[name]
        error = abs(rec_val - true_val) / true_val * 100
        total_error += error
        print(f"{name:<15} {true_val:<15.4f} {rec_val:<15.4f} {error:<15.2f}")
    print("-" * 60)
    print(f"Average Error: {total_error / len(param_names):.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()
