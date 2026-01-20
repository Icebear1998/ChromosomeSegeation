import numpy as np
import pandas as pd
import ast
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from simulation_utils import parse_parameters, calculate_k1_from_params, load_experimental_data
from FastBetaSimulation import simulate_batch
from FastFeedbackSimulation import simulate_batch_feedback

def run_simulation_and_get_times(mechanism, params, num_simulations=1000):
    """
    Run simulation and return T1, T2, T3 arrays.
    Adapted from simulation_utils.run_simulation_for_dataset logic.
    """
    n0_list = [params['n1'], params['n2'], params['n3']]
    initial_states = np.array([params['N1'], params['N2'], params['N3']])
    n0_array = np.array(n0_list)

    # Dispatch based on mechanism
    use_beta_method = mechanism in ['simple', 'fixed_burst', 'time_varying_k', 'time_varying_k_fixed_burst', 'time_varying_k_burst_onion']
    use_feedback_method = mechanism in ['feedback_onion', 'fixed_burst_feedback_onion', 'time_varying_k_feedback_onion', 'time_varying_k_combined']

    results = None

    if use_beta_method:
        if mechanism in ['simple', 'fixed_burst']:
            k = params['k']
            burst_size = params.get('burst_size', 1.0)
            k_1 = None
            k_max = None
        else:  # time_varying_k mechanisms
            k = None
            burst_size = params.get('burst_size', 1.0)
            k_1 = calculate_k1_from_params(params)
            k_max = params['k_max']
        
        results = simulate_batch(
            mechanism=mechanism,
            initial_states=initial_states,
            n0_lists=n0_array,
            k=k,
            burst_size=burst_size,
            k_1=k_1,
            k_max=k_max,
            num_simulations=num_simulations
        )

    elif use_feedback_method:
        # Default optional params
        k = None
        n_inner = params['n_inner']
        k_1 = None
        k_max = None
        burst_size = params.get('burst_size', 1.0)
        
        if mechanism == 'feedback_onion':
            k = params['k']
        elif mechanism == 'fixed_burst_feedback_onion':
            k = params['k']
        elif mechanism in ['time_varying_k_feedback_onion', 'time_varying_k_combined']:
            k_1 = calculate_k1_from_params(params)
            k_max = params['k_max']
            
        results = simulate_batch_feedback(
            mechanism=mechanism,
            initial_states=initial_states,
            n0_lists=n0_array,
            k=k,
            n_inner=n_inner,
            k_1=k_1,
            k_max=k_max,
            burst_size=burst_size,
            num_simulations=num_simulations
        )
    else:
        raise ValueError(f"Unknown or unsupported mechanism for fast simulation: {mechanism}")

    # Extract T1, T2, T3
    t1_array = results[:, 0]
    t2_array = results[:, 1]
    t3_array = results[:, 2]
    
    return t1_array, t2_array, t3_array

def process_file(csv_file, mechanism_name):
    print(f"--- Processing {mechanism_name} from {csv_file} ---")
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return

    # Load experimental data for comparison (Wildtype)
    exp_data = load_experimental_data()
    wt_t12 = exp_data.get('wildtype', {}).get('delta_t12', np.array([]))
    wt_t32 = exp_data.get('wildtype', {}).get('delta_t32', np.array([]))
    
    wt_t12_mean, wt_t12_std = (np.mean(wt_t12), np.std(wt_t12)) if len(wt_t12) > 0 else (0, 0)
    wt_t32_mean, wt_t32_std = (np.mean(wt_t32), np.std(wt_t32)) if len(wt_t32) > 0 else (0, 0)
    
    print(f"EXPERIMENTAL DATA (Wildtype):")
    print(f"  Delta(T1, T2): {wt_t12_mean:.2f} ± {wt_t12_std:.2f}")
    print(f"  Delta(T3, T2): {wt_t32_mean:.2f} ± {wt_t32_std:.2f}")
    print("-" * 40)

    results = {
        't1_means': [], 't1_stds': [],
        't2_means': [], 't2_stds': [],
        't3_means': [], 't3_stds': [],
        'dt12_means': [], 'dt12_stds': [],
        'dt32_means': [], 'dt32_stds': []
    }

    for idx, row in df.iterrows():
        try:
            fold = row.get('fold', idx)
            param_str = row['params']
            
            # Parse params list string
            if isinstance(param_str, str):
                param_list = ast.literal_eval(param_str)
            else:
                param_list = param_str # Assume it's already a list if not string (unlikely for CSV)

            # Convert to dictionary with derived params
            params = parse_parameters(param_list, mechanism_name)
            
            # Run simulation
            t1, t2, t3 = run_simulation_and_get_times(mechanism_name, params, num_simulations=1000)
            
            # Calculate Deltas
            dt12 = t1 - t2
            dt32 = t3 - t2
            
            # Calculate stats
            t1_mean, t1_std = np.mean(t1), np.std(t1)
            t2_mean, t2_std = np.mean(t2), np.std(t2)
            t3_mean, t3_std = np.mean(t3), np.std(t3)
            
            dt12_mean, dt12_std = np.mean(dt12), np.std(dt12)
            dt32_mean, dt32_std = np.mean(dt32), np.std(dt32)

            # Store stats
            results['t1_means'].append(t1_mean)
            results['t1_stds'].append(t1_std)
            results['t2_means'].append(t2_mean)
            results['t2_stds'].append(t2_std)
            results['t3_means'].append(t3_mean)
            results['t3_stds'].append(t3_std)
            results['dt12_means'].append(dt12_mean)
            results['dt12_stds'].append(dt12_std)
            results['dt32_means'].append(dt32_mean)
            results['dt32_stds'].append(dt32_std)
            
            # Report individual fold
            print(f"Fold {fold}:")
            print(f"  Sim Delta(T1, T2): {dt12_mean:.2f} ± {dt12_std:.2f}")
            print(f"  Sim Delta(T3, T2): {dt32_mean:.2f} ± {dt32_std:.2f}")

        except Exception as e:
            print(f"Error processing fold {idx} in {csv_file}: {e}")
    
    # Calculate and print averages across folds
    print("-" * 30)
    print(f"AVERAGE ACROSS ALL FOLDS ({mechanism_name}):")
    print(f"  Sim Delta(T1, T2): {np.mean(results['dt12_means']):.2f} ± {np.mean(results['dt12_stds']):.2f}")
    print(f"  Sim Delta(T3, T2): {np.mean(results['dt32_means']):.2f} ± {np.mean(results['dt32_stds']):.2f}")
    print(f"  (Exp Delta(T1, T2): {wt_t12_mean:.2f} ± {wt_t12_std:.2f})")
    print(f"  (Exp Delta(T3, T2): {wt_t32_mean:.2f} ± {wt_t32_std:.2f})")
    print("=" * 40 + "\n")

def main():
    files_map = {
        'cv_results_time_varying_k_277424.csv': 'time_varying_k',
        'cv_results_time_varying_k_combined_277424.csv': 'time_varying_k_combined',
        'cv_results_time_varying_k_feedback_onion_277424.csv': 'time_varying_k_feedback_onion',
        'cv_results_time_varying_k_fixed_burst_277424.csv': 'time_varying_k_fixed_burst'
    }
    
    base_dir = '/Users/kienphan/WorkingSpace/ResearchProjs/ChromosomeSegeation'
    
    for filename, mechanism in files_map.items():
        full_path = os.path.join(base_dir, filename)
        if os.path.exists(full_path):
            process_file(full_path, mechanism)
        else:
            print(f"File not found: {full_path}")

if __name__ == "__main__":
    main()
