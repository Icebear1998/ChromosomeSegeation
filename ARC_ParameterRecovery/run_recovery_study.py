#!/usr/bin/env python3
"""
Parameter Recovery Study - Main Orchestration Script for ARC

This script manages the entire parameter recovery experiment:
1. Establishes ground truth parameters from real data optimization
2. Generates synthetic data using ground truth
3. Launches N independent optimization runs in parallel
4. Saves results to CSV for analysis

Usage:
    python run_recovery_study.py --mechanism time_varying_k_combined --num_runs 50
"""

import argparse
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

# Import the necessary functions from existing scripts
from SimulationOptimization_join import run_optimization
from simulation_utils import (
    load_experimental_data, 
    run_simulation_for_dataset, 
    apply_mutant_params,
    get_parameter_names,
    calculate_likelihood
)

def get_ground_truth_params(mechanism, datasets, max_iterations=100, num_simulations=300):
    """
    Runs a single optimization against the REAL data to get a credible
    "ground truth" parameter set.
    
    Args:
        mechanism (str): Mechanism name
        datasets (dict): Real experimental datasets
        max_iterations (int): Iterations for ground truth optimization
        num_simulations (int): Simulations per evaluation
        
    Returns:
        dict: Ground truth parameters
    """
    print("=" * 60)
    print("STEP 1: Finding Ground Truth Parameters")
    print("=" * 60)
    print(f"Mechanism: {mechanism}")
    print(f"Max iterations: {max_iterations}")
    print(f"Simulations per evaluation: {num_simulations}")
    print()
    
    # Use existing optimization function to find ground truth
    results = run_optimization(
        mechanism=mechanism,
        datasets=datasets,
        max_iterations=max_iterations,
        num_simulations=num_simulations,
        selected_strains=None,  # Use all strains for ground truth
        use_parallel=True
    )
    
    if not results or not results.get('success'):
        raise RuntimeError("Failed to find ground truth parameters.")
    
    print("✓ Ground truth parameters found successfully!")
    print(f"  Final NLL: {results['nll']:.4f}")
    print(f"  Converged: {results.get('converged', 'Unknown')}")
    
    # Print ground truth parameters
    print("\nGround Truth Parameters:")
    for param, value in results['params'].items():
        print(f"  {param}: {value:.6f}")
    
    return results['params']

def generate_synthetic_data(mechanism, ground_truth_params, num_simulations=1000):
    """
    Generates high-quality, noise-free synthetic datasets for all strains.
    
    Args:
        mechanism (str): Mechanism name
        ground_truth_params (dict): Ground truth parameter values
        num_simulations (int): Number of simulations for synthetic data
        
    Returns:
        dict: Synthetic datasets for all strains
    """
    print("=" * 60)
    print("STEP 2: Generating Synthetic Data")
    print("=" * 60)
    print(f"Synthetic data size: {num_simulations} simulations per strain")
    print()
    
    # Convert ground truth to base parameters
    if mechanism == 'time_varying_k_combined':
        n2 = ground_truth_params['n2']
        N2 = ground_truth_params['N2']
        k_max = ground_truth_params['k_max']
        tau = ground_truth_params['tau']
        r21 = ground_truth_params['r21']
        r23 = ground_truth_params['r23']
        R21 = ground_truth_params['R21']
        R23 = ground_truth_params['R23']
        burst_size = ground_truth_params['burst_size']
        n_inner = ground_truth_params['n_inner']
        alpha = ground_truth_params['alpha']
        beta_k = ground_truth_params['beta_k']
        beta_tau = ground_truth_params['beta_tau']
        
        base_params = {
            'n1': max(r21 * n2, 1), 'n2': n2, 'n3': max(r23 * n2, 1),
            'N1': max(R21 * N2, 1), 'N2': N2, 'N3': max(R23 * N2, 1),
            'k_1': k_max / tau, 'k_max': k_max, 'tau': tau,
            'burst_size': burst_size, 'n_inner': n_inner
        }
    else:
        raise ValueError(f"Unsupported mechanism: {mechanism}")
    
    # Generate synthetic data for all strains
    synthetic_datasets = {}
    strains = ['wildtype', 'threshold', 'degrate', 'degrateAPC']
    
    for strain in strains:
        print(f"  Generating {strain} data...")
        
        # Apply mutant-specific modifications
        params, n0_list = apply_mutant_params(
            base_params, strain, alpha, beta_k, beta_tau
        )
        
        # Run high-quality simulation
        sim_delta_t12, sim_delta_t32 = run_simulation_for_dataset(
            mechanism, params, n0_list, num_simulations
        )
        
        if sim_delta_t12 is None or sim_delta_t32 is None:
            raise RuntimeError(f"Failed to generate synthetic data for {strain}")
        
        synthetic_datasets[strain] = {
            'delta_t12': np.array(sim_delta_t12),
            'delta_t32': np.array(sim_delta_t32)
        }
        
        print(f"    Generated {len(sim_delta_t12)} T1-T2 and {len(sim_delta_t32)} T3-T2 points")
    
    print("✓ Synthetic data generation complete!")
    return synthetic_datasets

def run_single_recovery(run_id, mechanism, synthetic_datasets, ground_truth_params, 
                       max_iterations=100, num_simulations=200):
    """
    Function to be run in parallel. Executes one full optimization run
    against the synthetic data from a random starting point.
    
    Args:
        run_id (int): Recovery run identifier
        mechanism (str): Mechanism name
        synthetic_datasets (dict): Target synthetic datasets
        ground_truth_params (dict): Ground truth parameters for reference
        max_iterations (int): Max iterations for recovery optimization
        num_simulations (int): Simulations per recovery evaluation
        
    Returns:
        dict: Recovery results or None if failed
    """
    print(f"Starting recovery run #{run_id+1}...")
    start_time = time.time()
    
    try:
        # Run optimization against synthetic data
        # Note: We use all strains in synthetic data for recovery
        results = run_optimization(
            mechanism=mechanism,
            datasets=synthetic_datasets,
            max_iterations=max_iterations,
            num_simulations=num_simulations,
            selected_strains=None,  # Use all synthetic strains
            use_parallel=False  # Single worker per recovery to avoid nested parallelization
        )
        
        elapsed_time = time.time() - start_time
        
        if results and results.get('success'):
            print(f"Recovery run #{run_id+1} completed in {elapsed_time:.1f}s. NLL: {results['nll']:.4f}")
            
            # Prepare result dictionary
            result_data = {
                'run_id': run_id,
                'converged': results.get('converged', False),
                'final_nll': results['nll'],
                'elapsed_time': elapsed_time
            }
            
            # Add recovered parameters
            for param, value in results['params'].items():
                result_data[param] = value
            
            # Add ground truth values for comparison
            for param, value in ground_truth_params.items():
                result_data[f'{param}_truth'] = value
            
            return result_data
        else:
            print(f"Recovery run #{run_id+1} failed after {elapsed_time:.1f}s")
            return {
                'run_id': run_id,
                'converged': False,
                'final_nll': 1e6,
                'elapsed_time': elapsed_time,
                **{param: np.nan for param in ground_truth_params.keys()},
                **{f'{param}_truth': value for param, value in ground_truth_params.items()}
            }
            
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"Recovery run #{run_id+1} failed with error: {e}")
        return {
            'run_id': run_id,
            'converged': False,
            'final_nll': 1e6,
            'elapsed_time': elapsed_time,
            **{param: np.nan for param in ground_truth_params.keys()},
            **{f'{param}_truth': value for param, value in ground_truth_params.items()}
        }

def main(args):
    """
    Main function to orchestrate the parameter recovery study.
    """
    print("=" * 80)
    print("PARAMETER RECOVERY STUDY")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mechanism: {args.mechanism}")
    print(f"Recovery runs: {args.num_runs}")
    print(f"Output file: {args.output_file}")
    print(f"Available CPUs: {cpu_count()}")
    print()
    
    total_start_time = time.time()
    
    try:
        # Load experimental data
        print("Loading experimental data...")
        datasets = load_experimental_data()
        print(f"✓ Loaded {len(datasets)} experimental datasets")
        for name, data in datasets.items():
            print(f"  {name}: {len(data['delta_t12'])} T1-T2, {len(data['delta_t32'])} T3-T2 points")
        print()
        
        # Step 1: Get ground truth parameters by fitting to real data
        ground_truth_params = get_ground_truth_params(
            args.mechanism, 
            datasets,
            max_iterations=args.gt_iterations,
            num_simulations=args.gt_simulations
        )
        
        # Step 2: Generate synthetic dataset
        synthetic_datasets = generate_synthetic_data(
            args.mechanism, 
            ground_truth_params,
            num_simulations=args.synthetic_size
        )
        
        # Save synthetic data for reference
        synthetic_filename = f"synthetic_data_{args.mechanism}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        synthetic_df_list = []
        for strain, data in synthetic_datasets.items():
            strain_df = pd.DataFrame({
                'strain': strain,
                'delta_t12': data['delta_t12'],
                'delta_t32': data['delta_t32']
            })
            synthetic_df_list.append(strain_df)
        
        synthetic_df = pd.concat(synthetic_df_list, ignore_index=True)
        synthetic_df.to_csv(synthetic_filename, index=False)
        print(f"✓ Synthetic data saved to: {synthetic_filename}")
        print()
        
        # Step 3: Run N recovery optimizations in parallel
        print("=" * 60)
        print("STEP 3: Running Parallel Recovery Optimizations")
        print("=" * 60)
        print(f"Number of recovery runs: {args.num_runs}")
        print(f"Max iterations per run: {args.max_iterations}")
        print(f"Simulations per evaluation: {args.num_simulations}")
        
        num_workers = min(args.num_runs, cpu_count())
        print(f"Using {num_workers} parallel processes")
        print("This will take several hours...")
        print()
        
        # Prepare the partial function for parallel processing
        task_func = partial(
            run_single_recovery,
            mechanism=args.mechanism,
            synthetic_datasets=synthetic_datasets,
            ground_truth_params=ground_truth_params,
            max_iterations=args.max_iterations,
            num_simulations=args.num_simulations
        )
        
        # Run recovery optimizations in parallel
        recovery_start_time = time.time()
        
        with Pool(processes=num_workers) as pool:
            results = pool.map(task_func, range(args.num_runs))
        
        recovery_time = time.time() - recovery_start_time
        
        # Step 4: Collect results and save to CSV
        print("=" * 60)
        print("STEP 4: Processing and Saving Results")
        print("=" * 60)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv(args.output_file, index=False)
        
        # Generate summary
        total_time = time.time() - total_start_time
        successful_runs = results_df['converged'].sum()
        
        print(f"✓ PARAMETER RECOVERY STUDY COMPLETED!")
        print(f"✓ Total time: {total_time:.1f} seconds ({total_time/3600:.2f} hours)")
        print(f"✓ Recovery time: {recovery_time:.1f} seconds ({recovery_time/3600:.2f} hours)")
        print(f"✓ Results saved to: {args.output_file}")
        print(f"✓ Successful recoveries: {successful_runs}/{args.num_runs} ({100*successful_runs/args.num_runs:.1f}%)")
        
        if successful_runs > 0:
            successful_df = results_df[results_df['converged']]
            best_nll = successful_df['final_nll'].min()
            mean_nll = successful_df['final_nll'].mean()
            std_nll = successful_df['final_nll'].std()
            print(f"✓ Best recovered NLL: {best_nll:.4f}")
            print(f"✓ Mean recovered NLL: {mean_nll:.4f} ± {std_nll:.4f}")
        
        # Create summary file
        summary_filename = f"recovery_summary_{args.mechanism}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_filename, 'w') as f:
            f.write("PARAMETER RECOVERY STUDY SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Mechanism: {args.mechanism}\n")
            f.write(f"Recovery runs: {args.num_runs}\n")
            f.write(f"Synthetic data size: {args.synthetic_size}\n")
            f.write(f"Max iterations: {args.max_iterations}\n")
            f.write(f"Simulations per evaluation: {args.num_simulations}\n")
            f.write(f"Parallel processes: {num_workers}\n")
            f.write(f"Total runtime: {total_time:.1f} seconds ({total_time/3600:.2f} hours)\n")
            f.write(f"Recovery runtime: {recovery_time:.1f} seconds ({recovery_time/3600:.2f} hours)\n")
            f.write(f"Successful recoveries: {successful_runs}/{args.num_runs} ({100*successful_runs/args.num_runs:.1f}%)\n")
            
            if successful_runs > 0:
                f.write(f"Best NLL: {best_nll:.4f}\n")
                f.write(f"Mean NLL: {mean_nll:.4f} ± {std_nll:.4f}\n")
            
            f.write(f"\nGround Truth Parameters:\n")
            for param, value in ground_truth_params.items():
                f.write(f"  {param}: {value:.6f}\n")
            
            f.write(f"\nOutput Files:\n")
            f.write(f"  Results: {args.output_file}\n")
            f.write(f"  Synthetic data: {synthetic_filename}\n")
            f.write(f"  Summary: {summary_filename}\n")
        
        print(f"✓ Summary saved to: {summary_filename}")
        print()
        print("Next steps:")
        print("1. Download the results CSV file to your local machine")
        print("2. Run analyze_parameter_recovery.py for detailed analysis")
        print("3. Review parameter identifiability and model sloppiness")
        
    except Exception as e:
        print(f"✗ Parameter recovery study failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error information
        error_filename = f"recovery_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(error_filename, 'w') as f:
            f.write(f"Parameter recovery study error\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write(f"Arguments: {args}\n")
            f.write(f"Error: {str(e)}\n")
            f.write(f"Traceback:\n")
            traceback.print_exc(file=f)
        
        print(f"Error details saved to: {error_filename}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a parameter recovery study in parallel.")
    
    # Required arguments
    parser.add_argument('--mechanism', type=str, required=True, 
                       help="Mechanism name to test (e.g., time_varying_k_combined)")
    
    # Recovery study configuration
    parser.add_argument('--num_runs', type=int, default=50, 
                       help="Number of recovery optimizations to run (default: 50)")
    parser.add_argument('--output_file', type=str, default="recovery_results.csv", 
                       help="Path to save the output CSV file (default: recovery_results.csv)")
    
    # Ground truth optimization settings
    parser.add_argument('--gt_iterations', type=int, default=100,
                       help="Max iterations for ground truth optimization (default: 100)")
    parser.add_argument('--gt_simulations', type=int, default=300,
                       help="Simulations per evaluation for ground truth (default: 300)")
    
    # Synthetic data settings
    parser.add_argument('--synthetic_size', type=int, default=1000,
                       help="Number of synthetic data points per strain (default: 1000)")
    
    # Recovery optimization settings
    parser.add_argument('--max_iterations', type=int, default=100,
                       help="Max iterations per recovery run (default: 100)")
    parser.add_argument('--num_simulations', type=int, default=200,
                       help="Simulations per evaluation for recovery (default: 200)")
    
    args = parser.parse_args()
    main(args)
