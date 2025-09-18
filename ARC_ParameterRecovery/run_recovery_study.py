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

def get_ground_truth_params(mechanism, study_dir="."):
    """
    Returns the ground truth parameters for the recovery study.
    These values are copied from simulation_optimized_parameters_time_varying_k_combined_finetune.txt
    and rounded to reasonable precision.
    
    Args:
        mechanism (str): Mechanism name
        study_dir (str): Directory to save the ground truth parameters file
        
    Returns:
        dict: Ground truth parameters (rounded)
    """
    print("=" * 60)
    print("STEP 1: Setting Ground Truth Parameters")
    print("=" * 60)
    print(f"Mechanism: {mechanism}")
    print("Using pre-defined ground truth parameters from fine-tuned optimization")
    print()
    
    if mechanism == 'time_varying_k_combined':
        # Ground truth parameters from simulation_optimized_parameters_time_varying_k_combined_finetune.txt
        # Original values (rounded to reasonable precision for recovery study):
        ground_truth_params = {
            'n2': 5,           # Original: 5.121055
            'N2': 180,           # Original: 168.842023  
            'k_max': 0.015,   # Original: 0.013139
            'tau': 120,         # Original: 47.262590
            'r21': 2.415,        # Original: 2.415095
            'r23': 1.458,        # Original: 1.457673
            'R21': 1.293,        # Original: 1.292956
            'R23': 2.475,        # Original: 2.474872
            'burst_size': 16,  # Original: 14.276450
            'n_inner': 20,     # Original: 20.503246
            'alpha': 0.654,      # Original: 0.653572
            'beta_k': 0.720,     # Original: 0.720142
            'beta_tau': 2.5      # Original: 2.9
        }
    else:
        raise ValueError(f"Ground truth parameters not defined for mechanism: {mechanism}")
    
    print("✓ Ground truth parameters set successfully!")
    print(f"  Loaded {len(ground_truth_params)} parameters")
    
    # Print ground truth parameters
    print("\nGround Truth Parameters (rounded):")
    for param, value in ground_truth_params.items():
        print(f"  {param}: {value}")
    
        # Save rounded parameters for reference in the study directory
    rounded_file = os.path.join(study_dir, "ground_truth_parameters_rounded.txt")
    with open(rounded_file, 'w') as f:
        f.write("Ground Truth Parameters for Parameter Recovery Study\n")
        f.write("=" * 60 + "\n")
        f.write(f"Source: simulation_optimized_parameters_time_varying_k_combined_finetune.txt (manually copied)\n")
        f.write(f"Mechanism: {mechanism}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\nRounded Parameters:\n")
        for param, value in ground_truth_params.items():
            f.write(f"{param} = {value}\n")
    
    print(f"✓ Rounded parameters saved to: {rounded_file}")
    
    return ground_truth_params

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
                       max_iterations=100, num_simulations=200, output_file=None):
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
        output_file (str): Path to save individual results incrementally
        
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
            
        else:
            print(f"Recovery run #{run_id+1} failed after {elapsed_time:.1f}s")
            result_data = {
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
        result_data = {
            'run_id': run_id,
            'converged': False,
            'final_nll': 1e6,
            'elapsed_time': elapsed_time,
            **{param: np.nan for param in ground_truth_params.keys()},
            **{f'{param}_truth': value for param, value in ground_truth_params.items()}
        }
    
    # Save individual result immediately
    if output_file:
        save_individual_result(result_data, output_file)
    
    return result_data

def save_individual_result(result_data, output_file):
    """
    Save individual recovery result to CSV file incrementally.
    Creates file with header if it doesn't exist, otherwise appends.
    
    Args:
        result_data (dict): Single recovery result
        output_file (str): Path to CSV file
    """
    import fcntl  # For file locking on Unix systems
    
    try:
        # Convert to DataFrame
        result_df = pd.DataFrame([result_data])
        
        # Check if file exists
        file_exists = os.path.exists(output_file)
        
        # Use file locking to prevent race conditions in parallel writing
        with open(output_file, 'a' if file_exists else 'w') as f:
            # Lock the file
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            
            # Write header if new file
            if not file_exists:
                result_df.to_csv(f, index=False, header=True)
            else:
                result_df.to_csv(f, index=False, header=False)
            
            # Flush to ensure data is written
            f.flush()
            os.fsync(f.fileno())
            
        # Count how many results we have so far
        if file_exists:
            try:
                existing_df = pd.read_csv(output_file)
                total_completed = len(existing_df)
                print(f"✓ Saved result for run #{result_data['run_id']+1} to {output_file} (Progress: {total_completed} runs completed)")
                
                # Create checkpoint backup every 10 runs
                if total_completed % 10 == 0:
                    # Extract just the timestamp from the filename for checkpoint naming
                    base_name = os.path.basename(output_file)
                    if 'recovery_results_' in base_name:
                        timestamp_part = base_name.replace('recovery_results_', '').replace('.csv', '')
                        checkpoint_file = os.path.join(os.path.dirname(output_file), f'recovery_results_{timestamp_part}_checkpoint_{total_completed}.csv')
                    else:
                        checkpoint_file = output_file.replace('.csv', f'_checkpoint_{total_completed}.csv')
                    existing_df.to_csv(checkpoint_file, index=False)
                    print(f"✓ Created checkpoint backup: {checkpoint_file}")
                    
            except:
                print(f"✓ Saved result for run #{result_data['run_id']+1} to {output_file}")
        else:
            print(f"✓ Saved result for run #{result_data['run_id']+1} to {output_file} (Progress: 1 runs completed)")
        
    except Exception as e:
        print(f"Warning: Failed to save individual result for run #{result_data['run_id']+1}: {e}")
        # Don't fail the entire run if saving fails

def main(args):
    """
    Main function to orchestrate the parameter recovery study.
    """
    # Create organized output directory structure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create main Output folder
    output_dir = "Output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find the next available Recovery folder number
    recovery_num = 1
    while True:
        study_dir = os.path.join(output_dir, f"Recovery{recovery_num}")
        if not os.path.exists(study_dir):
            break
        recovery_num += 1
    
    # Create the recovery study directory
    os.makedirs(study_dir, exist_ok=True)
    
    # Update output file path to be in the recovery study directory
    args.output_file = os.path.join(study_dir, f"recovery_results_{timestamp}.csv")
    
    print("=" * 80)
    print("PARAMETER RECOVERY STUDY")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mechanism: {args.mechanism}")
    print(f"Recovery runs: {args.num_runs}")
    print(f"Output directory: {output_dir}")
    print(f"Study directory: {study_dir}")
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
        
        # Step 1: Set ground truth parameters from fine-tuned optimization results
        ground_truth_params = get_ground_truth_params(
            args.mechanism,
            study_dir
        )
        
        # Step 2: Generate synthetic dataset
        synthetic_datasets = generate_synthetic_data(
            args.mechanism, 
            ground_truth_params,
            num_simulations=args.synthetic_size
        )
        
        # Save synthetic data for reference in the study directory
        synthetic_filename = os.path.join(study_dir, f"synthetic_data_{args.mechanism}_{timestamp}.csv")
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
            num_simulations=args.num_simulations,
            output_file=args.output_file
        )
        
        # Initialize the output file with proper header
        print(f"Initializing output file: {args.output_file}")
        if os.path.exists(args.output_file):
            print(f"Warning: Output file {args.output_file} already exists. Results will be appended.")
        
        # Run recovery optimizations in parallel
        recovery_start_time = time.time()
        
        with Pool(processes=num_workers) as pool:
            results = pool.map(task_func, range(args.num_runs))
        
        recovery_time = time.time() - recovery_start_time
        
        # Step 4: Verify and finalize results
        print("=" * 60)
        print("STEP 4: Finalizing Results")
        print("=" * 60)
        
        # Read the incrementally saved results
        if os.path.exists(args.output_file):
            results_df = pd.read_csv(args.output_file)
            print(f"✓ Read {len(results_df)} results from incremental saves")
        else:
            # Fallback: create DataFrame from returned results
            results_df = pd.DataFrame(results)
            results_df.to_csv(args.output_file, index=False)
            print(f"✓ Saved {len(results_df)} results to {args.output_file}")
        
        # Verify we have all expected results
        expected_runs = set(range(args.num_runs))
        actual_runs = set(results_df['run_id'].values) if 'run_id' in results_df.columns else set()
        missing_runs = expected_runs - actual_runs
        
        if missing_runs:
            print(f"Warning: Missing results for runs: {sorted(missing_runs)}")
        else:
            print(f"✓ All {args.num_runs} recovery runs completed and saved")
        
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
        
        # Create summary file in the study directory
        summary_filename = os.path.join(study_dir, f"recovery_summary_{args.mechanism}_{timestamp}.txt")
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
    
    # Ground truth parameters are now hard-coded in the get_ground_truth_params function
    
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
