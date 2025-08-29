#!/usr/bin/env python3
"""
Parameter Recovery Study for MoM-based Optimization (Local Version)

This script performs a parameter recovery study using Method of Moments (MoM)
instead of direct simulation. It's designed to run locally and complement
the simulation-based ARC study.

The study:
1. Uses optimized parameters from MoM optimization as "ground truth"
2. Generates synthetic data using MoM theoretical PDFs
3. Runs multiple recovery attempts from different starting points
4. Assesses parameter identifiability and model sloppiness
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from multiprocessing import Pool, cpu_count
import time
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import MoM optimization functions
from MoMOptimization_join import (
    joint_objective, 
    get_mechanism_info, 
    unpack_parameters,
    get_rounded_parameters
)
from MoMCalculations import compute_pdf_for_mechanism

class MoMParameterRecoveryStudy:
    """
    Manages parameter recovery study using Method of Moments approach.
    """
    
    def __init__(self, mechanism='fixed_burst_feedback_onion', n_recovery_runs=30, 
                 synthetic_data_size=500, gamma_mode='separate'):
        """
        Initialize the MoM parameter recovery study.
        
        Args:
            mechanism (str): Mechanism to study
            n_recovery_runs (int): Number of recovery optimization runs
            synthetic_data_size (int): Size of synthetic datasets
            gamma_mode (str): 'unified' or 'separate' gamma mode
        """
        self.mechanism = mechanism
        self.n_recovery_runs = n_recovery_runs
        self.synthetic_data_size = synthetic_data_size
        self.gamma_mode = gamma_mode
        
        # Get mechanism information
        self.mechanism_info = get_mechanism_info(mechanism, gamma_mode)
        
        print(f"MoM Parameter Recovery Study Setup:")
        print(f"  Mechanism: {mechanism}")
        print(f"  Gamma mode: {gamma_mode}")
        print(f"  Parameters: {len(self.mechanism_info['params'])} ({', '.join(self.mechanism_info['params'])})")
        print(f"  Recovery runs: {n_recovery_runs}")
        print(f"  Synthetic data size: {synthetic_data_size}")
    
    def load_ground_truth_parameters(self, filename=None):
        """
        Load ground truth parameters from MoM optimization results file.
        
        Args:
            filename (str, optional): Parameter file to load. If None, tries to find automatically.
            
        Returns:
            dict: Ground truth parameters
        """
        if filename is None:
            # Try to find the most recent parameter file for this mechanism
            # Look for files that contain the mechanism name
            pattern = f"optimized_parameters_{self.mechanism}"
            files = [f for f in os.listdir('.') if pattern in f and f.endswith('.txt')]
            
            # If no exact match, try broader search
            if not files:
                broader_pattern = f"optimized_parameters_"
                files = [f for f in os.listdir('.') if f.startswith(broader_pattern) and f.endswith('.txt')]
                # Filter for mechanism name in filename
                files = [f for f in files if self.mechanism in f]
            
            if not files:
                raise FileNotFoundError(f"No parameter files found for mechanism {self.mechanism}")
            
            # Use the most recent file
            filename = max(files, key=os.path.getmtime)
            print(f"Using parameter file: {filename}")
        
        # Parse the parameter file
        ground_truth = {}
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
                
            # Parse parameters from the file
            for line in lines:
                line = line.strip()
                if ':' in line and not line.startswith('#'):
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Skip NLL values and comments
                    if '_nll' in key or key in ['total_nll', 'wt_nll']:
                        continue
                    if value == 'not_fitted' or '#' in value:
                        continue
                    
                    try:
                        ground_truth[key] = float(value.split()[0])  # Take first number if there are comments
                    except (ValueError, IndexError):
                        continue
            
            # Calculate derived parameters if missing
            # The ratio parameters r21, r23, R21, R23 are derived from the basic parameters
            if 'r21' not in ground_truth and 'n1' in ground_truth and 'n2' in ground_truth:
                ground_truth['r21'] = ground_truth['n1'] / ground_truth['n2']
            if 'r23' not in ground_truth and 'n3' in ground_truth and 'n2' in ground_truth:
                ground_truth['r23'] = ground_truth['n3'] / ground_truth['n2']
            if 'R21' not in ground_truth and 'N1' in ground_truth and 'N2' in ground_truth:
                ground_truth['R21'] = ground_truth['N1'] / ground_truth['N2']
            if 'R23' not in ground_truth and 'N3' in ground_truth and 'N2' in ground_truth:
                ground_truth['R23'] = ground_truth['N3'] / ground_truth['N2']
            
            # Verify we have the required parameters for the mechanism
            required_params = self.mechanism_info['params']
            missing_params = set(required_params) - set(ground_truth.keys())
            if missing_params:
                print(f"Warning: Missing parameters in file: {missing_params}")
                print(f"Available parameters: {list(ground_truth.keys())}")
            
            print(f"✓ Loaded {len(ground_truth)} ground truth parameters")
            return ground_truth
            
        except Exception as e:
            raise ValueError(f"Error loading parameters from {filename}: {e}")
    
    def generate_synthetic_data(self, ground_truth_params):
        """
        Generate synthetic data using MoM theoretical PDFs.
        
        Args:
            ground_truth_params (dict): Ground truth parameter values
            
        Returns:
            dict: Synthetic datasets for all strains
        """
        print(f"\nGenerating synthetic data using MoM theoretical PDFs...")
        print(f"Synthetic data size: {self.synthetic_data_size} points per strain/chromosome pair")
        
        # Convert ground truth dict to parameter vector for MoM functions
        param_vector = []
        for param_name in self.mechanism_info['params']:
            if param_name in ground_truth_params:
                param_vector.append(ground_truth_params[param_name])
            else:
                raise ValueError(f"Missing parameter {param_name} in ground truth")
        
        param_vector = np.array(param_vector)
        param_dict = unpack_parameters(param_vector, self.mechanism_info)
        
        # Extract mechanism-specific parameters
        mech_params = {}
        if self.mechanism == 'fixed_burst':
            mech_params['burst_size'] = param_dict['burst_size']
        elif self.mechanism == 'time_varying_k':
            mech_params['k_1'] = param_dict['k_1']
        elif self.mechanism == 'feedback':
            mech_params['feedbackSteepness'] = param_dict['feedbackSteepness']
            mech_params['feedbackThreshold'] = param_dict['feedbackThreshold']
        elif self.mechanism == 'feedback_linear':
            mech_params['w1'] = param_dict['w1']
            mech_params['w2'] = param_dict['w2']
            mech_params['w3'] = param_dict['w3']
        elif self.mechanism == 'feedback_onion':
            mech_params['n_inner'] = param_dict['n_inner']
        elif self.mechanism == 'feedback_zipper':
            mech_params['z1'] = param_dict['z1']
            mech_params['z2'] = param_dict['z2']
            mech_params['z3'] = param_dict['z3']
        elif self.mechanism == 'fixed_burst_feedback_linear':
            mech_params['burst_size'] = param_dict['burst_size']
            mech_params['w1'] = param_dict['w1']
            mech_params['w2'] = param_dict['w2']
            mech_params['w3'] = param_dict['w3']
        elif self.mechanism == 'fixed_burst_feedback_onion':
            mech_params['burst_size'] = param_dict['burst_size']
            mech_params['n_inner'] = param_dict['n_inner']
        
        # Generate synthetic data ranges (similar to experimental data ranges)
        # Create reasonable time ranges for each strain
        time_ranges = {
            'wildtype': {'12': np.linspace(-80, 80, self.synthetic_data_size),
                        '32': np.linspace(-80, 80, self.synthetic_data_size)},
            'threshold': {'12': np.linspace(-80, 80, self.synthetic_data_size),
                         '32': np.linspace(-80, 80, self.synthetic_data_size)},
            'degrate': {'12': np.linspace(-80, 80, self.synthetic_data_size),
                       '32': np.linspace(-80, 80, self.synthetic_data_size)},
            'degrateAPC': {'12': np.linspace(-80, 80, self.synthetic_data_size),
                          '32': np.linspace(-80, 80, self.synthetic_data_size)}
        }
        
        synthetic_datasets = {}
        
        # Generate data for each strain
        strains = ['wildtype', 'threshold', 'degrate', 'degrateAPC']
        
        for strain in strains:
            print(f"  Generating {strain} data...")
            
            strain_data = {}
            
            # Apply strain-specific parameter modifications
            if strain == 'wildtype':
                n1, n2, n3 = param_dict['n1'], param_dict['n2'], param_dict['n3']
                N1, N2, N3 = param_dict['N1'], param_dict['N2'], param_dict['N3']
                k = param_dict['k']
                
            elif strain == 'threshold':
                n1 = max(param_dict['n1'] * param_dict['alpha'], 1)
                n2 = max(param_dict['n2'] * param_dict['alpha'], 1)
                n3 = max(param_dict['n3'] * param_dict['alpha'], 1)
                N1, N2, N3 = param_dict['N1'], param_dict['N2'], param_dict['N3']
                k = param_dict['k']
                
            elif strain == 'degrate':
                n1, n2, n3 = param_dict['n1'], param_dict['n2'], param_dict['n3']
                N1, N2, N3 = param_dict['N1'], param_dict['N2'], param_dict['N3']
                k = max(param_dict['beta_k'] * param_dict['k'], 0.001)
                
            elif strain == 'degrateAPC':
                n1, n2, n3 = param_dict['n1'], param_dict['n2'], param_dict['n3']
                N1, N2, N3 = param_dict['N1'], param_dict['N2'], param_dict['N3']
                k = max(param_dict['beta2_k'] * param_dict['k'], 0.001)
            
            # Generate PDF and sample from it for both chromosome pairs
            for pair in ['12', '32']:
                time_points = time_ranges[strain][pair]
                
                if pair == '12':
                    pdf = compute_pdf_for_mechanism(
                        self.mechanism, time_points, n1, N1, n2, N2, k, mech_params, pair12=True
                    )
                else:  # pair == '32'
                    pdf = compute_pdf_for_mechanism(
                        self.mechanism, time_points, n3, N3, n2, N2, k, mech_params, pair12=False
                    )
                
                # Normalize PDF and sample from it
                if np.any(pdf > 0) and not np.any(np.isnan(pdf)):
                    pdf_normalized = pdf / np.sum(pdf)
                    
                    # Sample from the theoretical distribution
                    synthetic_times = np.random.choice(
                        time_points, 
                        size=self.synthetic_data_size, 
                        p=pdf_normalized, 
                        replace=True
                    )
                    
                    strain_data[f'delta_t{pair}'] = synthetic_times
                else:
                    # Fallback: generate random data in reasonable range
                    print(f"    Warning: Invalid PDF for {strain} {pair}, using fallback data")
                    strain_data[f'delta_t{pair}'] = np.random.normal(0, 20, self.synthetic_data_size)
            
            synthetic_datasets[strain] = strain_data
            print(f"    Generated {len(strain_data['delta_t12'])} T1-T2 and {len(strain_data['delta_t32'])} T3-T2 points")
        
        print("✓ Synthetic data generation complete!")
        return synthetic_datasets
    
    def recovery_objective(self, params_vector, synthetic_datasets, ground_truth_params):
        """
        Objective function for parameter recovery optimization.
        
        Args:
            params_vector (array): Parameter vector to test
            synthetic_datasets (dict): Target synthetic datasets
            ground_truth_params (dict): Ground truth parameters for reference
            
        Returns:
            float: Negative log-likelihood
        """
        try:
            # Convert synthetic datasets to the format expected by joint_objective
            data_wt12 = synthetic_datasets['wildtype']['delta_t12']
            data_wt32 = synthetic_datasets['wildtype']['delta_t32']
            data_threshold12 = synthetic_datasets['threshold']['delta_t12']
            data_threshold32 = synthetic_datasets['threshold']['delta_t32']
            data_degrate12 = synthetic_datasets['degrate']['delta_t12']
            data_degrate32 = synthetic_datasets['degrate']['delta_t32']
            data_degrateAPC12 = synthetic_datasets['degrateAPC']['delta_t12']
            data_degrateAPC32 = synthetic_datasets['degrateAPC']['delta_t32']
            
            # Use empty arrays for initial strain (excluded from fitting)
            data_initial12 = np.array([])
            data_initial32 = np.array([])
            
            # Call the joint objective function from MoM optimization
            nll = joint_objective(
                params_vector, self.mechanism, self.mechanism_info,
                data_wt12, data_wt32, data_threshold12, data_threshold32,
                data_degrate12, data_degrate32, data_initial12, data_initial32,
                data_degrateAPC12, data_degrateAPC32
            )
            
            return nll
            
        except Exception as e:
            return 1e6

def run_single_mom_recovery(args):
    """
    Run a single parameter recovery optimization for MoM.
    This function is designed to be called by multiprocessing.
    
    Args:
        args (tuple): (run_id, mechanism, mechanism_info, param_bounds, synthetic_datasets, ground_truth_params, max_iterations)
        
    Returns:
        dict: Recovery results for this run
    """
    run_id, mechanism, mechanism_info, param_bounds, synthetic_datasets, ground_truth_params, max_iterations = args
    
    print(f"Starting MoM recovery run {run_id + 1}")
    start_time = time.time()
    
    try:
        # Create study instance for this process
        study = MoMParameterRecoveryStudy(mechanism=mechanism)
        
        # Run optimization with random initial guess
        result = differential_evolution(
            study.recovery_objective,
            param_bounds,
            args=(synthetic_datasets, ground_truth_params),
            maxiter=max_iterations,
            popsize=15,
            seed=None,  # Use different random seed for each run
            disp=False,  # Suppress output for parallel runs
            workers=1    # Each process uses single worker
        )
        
        # Extract results
        param_dict = unpack_parameters(result.x, mechanism_info)
        final_nll = result.fun
        converged = result.success and final_nll < 1e5  # Consider converged if reasonable NLL
        
        elapsed_time = time.time() - start_time
        print(f"Completed MoM recovery run {run_id + 1} in {elapsed_time:.1f}s (NLL: {final_nll:.2f})")
        
        # Return results
        results = {
            'run_id': run_id,
            'converged': converged,
            'final_nll': final_nll,
            'elapsed_time': elapsed_time
        }
        
        # Add recovered parameters
        for param_name in mechanism_info['params']:
            results[param_name] = param_dict[param_name]
        
        # Add ground truth values for comparison
        for param_name in mechanism_info['params']:
            if param_name in ground_truth_params:
                results[f'{param_name}_truth'] = ground_truth_params[param_name]
        
        return results
        
    except Exception as e:
        print(f"MoM recovery run {run_id + 1} failed: {e}")
        # Return failed result
        results = {
            'run_id': run_id,
            'converged': False,
            'final_nll': 1e6,
            'elapsed_time': time.time() - start_time
        }
        # Add NaN for all parameters
        for param_name in mechanism_info['params']:
            results[param_name] = np.nan
            if param_name in ground_truth_params:
                results[f'{param_name}_truth'] = ground_truth_params[param_name]
        
        return results

def main():
    """
    Main function to run the complete MoM parameter recovery study.
    """
    print("=" * 80)
    print("MoM PARAMETER RECOVERY STUDY")
    print("=" * 80)
    
    # Configuration
    mechanism = 'fixed_burst_feedback_onion'  # Change this to test different mechanisms
    gamma_mode = 'separate'                   # 'unified' or 'separate'
    n_recovery_runs = 25                      # Number of recovery attempts
    synthetic_data_size = 300                 # Size of synthetic dataset per strain
    max_iterations = 100                      # Max iterations per recovery run
    n_processes = min(cpu_count(), n_recovery_runs)  # Use available CPU cores
    
    # Initialize study
    study = MoMParameterRecoveryStudy(
        mechanism=mechanism,
        n_recovery_runs=n_recovery_runs,
        synthetic_data_size=synthetic_data_size,
        gamma_mode=gamma_mode
    )
    
    try:
        # Step 1: Load ground truth parameters
        print(f"\n{'='*60}")
        print("STEP 1: Loading Ground Truth Parameters")
        print(f"{'='*60}")
        
        ground_truth = study.load_ground_truth_parameters()
        
        print("Ground truth parameters:")
        for param, value in ground_truth.items():
            print(f"  {param}: {value:.6f}")
        
        # Step 2: Generate synthetic data
        print(f"\n{'='*60}")
        print("STEP 2: Generating Synthetic Data")
        print(f"{'='*60}")
        
        synthetic_datasets = study.generate_synthetic_data(ground_truth)
        
        # Save synthetic data for reference
        synthetic_filename = f"mom_synthetic_data_{mechanism}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
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
        
        # Step 3: Run parameter recovery in parallel
        print(f"\n{'='*60}")
        print("STEP 3: Running MoM Parameter Recovery")
        print(f"{'='*60}")
        
        print(f"Starting {n_recovery_runs} recovery runs using {n_processes} processes...")
        
        # Prepare arguments for parallel processing
        args_list = []
        for run_id in range(n_recovery_runs):
            args = (
                run_id, 
                mechanism, 
                study.mechanism_info,
                study.mechanism_info['bounds'], 
                synthetic_datasets, 
                ground_truth,
                max_iterations
            )
            args_list.append(args)
        
        # Run recovery optimizations in parallel
        start_time = time.time()
        
        with Pool(processes=n_processes) as pool:
            recovery_results = pool.map(run_single_mom_recovery, args_list)
        
        total_time = time.time() - start_time
        
        # Step 4: Save results
        print(f"\n{'='*60}")
        print("STEP 4: Saving Results")
        print(f"{'='*60}")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(recovery_results)
        
        # Save to CSV
        output_filename = f"mom_parameter_recovery_{mechanism}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_filename, index=False)
        
        # Print summary
        print(f"✓ MoM recovery study completed in {total_time:.1f} seconds")
        print(f"✓ Results saved to: {output_filename}")
        
        successful_runs = results_df['converged'].sum()
        print(f"✓ Successful recoveries: {successful_runs}/{n_recovery_runs} ({100*successful_runs/n_recovery_runs:.1f}%)")
        
        if successful_runs > 0:
            best_nll = results_df[results_df['converged']]['final_nll'].min()
            mean_nll = results_df[results_df['converged']]['final_nll'].mean()
            print(f"✓ Best recovered NLL: {best_nll:.2f}")
            print(f"✓ Mean recovered NLL: {mean_nll:.2f}")
            
            # Show parameter recovery statistics
            print(f"\nParameter Recovery Summary:")
            param_names = study.mechanism_info['params']
            for param in param_names:
                if param in results_df.columns:
                    successful_values = results_df[results_df['converged']][param]
                    truth_col = f'{param}_truth'
                    
                    if truth_col in results_df.columns and len(successful_values) > 0:
                        truth_value = results_df[truth_col].iloc[0]
                        mean_recovered = successful_values.mean()
                        std_recovered = successful_values.std()
                        relative_error = abs(mean_recovered - truth_value) / abs(truth_value) * 100
                        
                        print(f"  {param:12s}: truth={truth_value:8.4f}, "
                              f"recovered={mean_recovered:8.4f}±{std_recovered:6.4f} "
                              f"(error: {relative_error:5.1f}%)")
        
        print(f"\n{'='*80}")
        print("MoM PARAMETER RECOVERY STUDY COMPLETE")
        print(f"{'='*80}")
        
        print(f"\nNext steps:")
        print(f"1. Run analyze_parameter_recovery_mom.py to visualize results")
        print(f"2. Compare with simulation-based recovery results from ARC")
        print(f"3. Assess parameter identifiability differences between MoM and simulation")
        
    except Exception as e:
        print(f"Error in MoM parameter recovery study: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
