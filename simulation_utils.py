#!/usr/bin/env python3
"""
Shared utilities for simulation-based optimization of chromosome segregation timing models.
Contains common functions used by both joint and independent optimization strategies.
"""

import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import wasserstein_distance
import warnings
import sys
import os

warnings.filterwarnings('ignore')



def load_experimental_data():
    """
    Load experimental data from Excel file.
    
    Returns:
        dict: Dictionary with dataset names as keys and timing difference arrays as values
    """
    try:
        file_path = "Data/All_strains_SCStimes.xlsx"
        
        # Try importing pandas first
        try:
            import pandas as pd
            df = pd.read_excel(file_path, sheet_name='Sheet1')
            
            datasets = {}
            dataset_mapping = {
                'wildtype': ('wildtype12', 'wildtype32'),
                'threshold': ('threshold12', 'threshold32'),
                'degrade': ('degRade12', 'degRade32'),
                'degradeAPC': ('degRadeAPC12', 'degRadeAPC32'),
                'velcade': ('degRadeVel12', 'degRadeVel32')
            }
            
            for dataset_name, (col_12, col_32) in dataset_mapping.items():
                if col_12 in df.columns and col_32 in df.columns:
                    datasets[dataset_name] = {
                        'delta_t12': df[col_12].dropna().values,
                        'delta_t32': df[col_32].dropna().values
                    }
            return datasets
            
        except ImportError:
            # Fallback to openpyxl if pandas is not available
            import openpyxl
            wb = openpyxl.load_workbook(file_path, data_only=True)
            sheet = wb['Sheet1']
            
            # Get headers from first row
            headers = [cell.value for cell in sheet[1]]
            
            data_cols = {}
            for col_idx, header in enumerate(headers):
                if header:
                    # Read column data (skipping header)
                    col_values = []
                    for row in sheet.iter_rows(min_row=2, min_col=col_idx+1, max_col=col_idx+1, values_only=True):
                        val = row[0]
                        if val is not None:
                            col_values.append(val)
                    data_cols[header] = np.array(col_values)
            
            datasets = {}
            dataset_mapping = {
                'wildtype': ('wildtype12', 'wildtype32'),
                'threshold': ('threshold12', 'threshold32'),
                'degrade': ('degRade12', 'degRade32'),
                'degradeAPC': ('degRadeAPC12', 'degRadeAPC32'),
                'velcade': ('degRadeVel12', 'degRadeVel32')
            }
            
            for dataset_name, (col_12, col_32) in dataset_mapping.items():
                if col_12 in data_cols and col_32 in data_cols:
                    datasets[dataset_name] = {
                        'delta_t12': data_cols[col_12],
                        'delta_t32': data_cols[col_32]
                    }
            return datasets

    except Exception as e:
        print(f"Error loading experimental data: {e}")
        return {}


def apply_mutant_params(base_params, mutant_type, alpha, beta_k=None, beta_k1=None, beta_k2=None, beta_k3=None, beta_tau=None, beta_tau2=None):
    """
    Apply mutant-specific parameter modifications for both simple and time-varying mechanisms.
    
    Args:
        base_params (dict): Base wildtype parameters
        mutant_type (str): Type of mutant ('wildtype', 'threshold', 'degrade', 'degradeAPC', 'velcade')
        alpha (float): Multiplier for threshold counts (threshold mutant)
        beta_k (float): Multiplier for k_max (time-varying) - degradation rate
        beta_k1 (float): Multiplier for k (simple) - separase mutant (degrade)
        beta_k2 (float): Multiplier for k (simple) - APC mutant (degradeAPC)
        beta_k3 (float): Multiplier for k (simple) - velcade mutant (velcade)
        beta_tau (float): Multiplier for tau (APC mutant) - tau becomes 2-3 times larger (time-varying only)
        beta_tau2 (float): Multiplier for tau (velcade mutant) - similar to APC but separate parameter (time-varying only)
    
    Returns:
        tuple: (modified_params, modified_n0_list)
    """
    params = base_params.copy()
    
    # Base parameters - always present
    n1, n2, n3 = params['n1'], params['n2'], params['n3']
    N1, N2, N3 = params['N1'], params['N2'], params['N3']
    
    # Determine if this is a simple mechanism or time-varying
    is_simple = 'k' in params  # Simple mechanisms have 'k', time-varying have 'k_max' and 'tau'
    # Note: 'beta_k' is now exclusively for time-varying k_max modification.
    # Simple mechanisms use beta_k1, beta_k2, beta_k3.

    if mutant_type == 'threshold':
        # Threshold mutant: reduce threshold counts (small n)
        # FIX: Enforce n >= 1 to match MoM logic
        n1 = max(alpha * n1, 1.0)
        n2 = max(alpha * n2, 1.0)
        n3 = max(alpha * n3, 1.0)
    
    elif mutant_type == 'degrade':
        # Degradation mutant: reduce degradation rate
        if is_simple:
            # Simple mechanisms: modify 'k' with beta_k1
            if beta_k1 is None:
                raise ValueError("beta_k1 required for simple mechanism 'degrade' mutant")
            params['k'] = beta_k1 * params['k']
        else:
            # Time-varying mechanisms: modify 'k_max' with beta_k
            if beta_k is None:
                 raise ValueError("beta_k required for time-varying mechanism 'degrade' mutant")
            params['k_max'] = beta_k * params['k_max']
    
    elif mutant_type == 'degradeAPC':
        if is_simple:
            # Simple mechanisms: modify 'k' with beta_k2
            if beta_k2 is None:
                raise ValueError("beta_k2 required for simple mechanism 'degradeAPC' mutant")
            params['k'] = beta_k2 * params['k']
        else:
            if 'tau' in params:
                params['tau'] = beta_tau * params['tau']
            elif 'k_1' in params and 'k_max' in params:
                current_tau = params['k_max'] / params['k_1']
                params['k_1'] = params['k_max'] / (beta_tau * current_tau)
    
    elif mutant_type == 'velcade':
        if is_simple:
             # Simple mechanisms: modify 'k' with beta_k3
            if beta_k3 is None:
                raise ValueError("beta_k3 required for simple mechanism 'velcade' mutant")
            params['k'] = beta_k3 * params['k']
        else:
            if 'tau' in params:
                params['tau'] = beta_tau2 * params['tau']
            elif 'k_1' in params and 'k_max' in params:
                current_tau = params['k_max'] / params['k_1']
                params['k_1'] = params['k_max'] / (beta_tau2 * current_tau)
    
    # Update the modified parameters
    params['n1'], params['n2'], params['n3'] = n1, n2, n3
    params['N1'], params['N2'], params['N3'] = N1, N2, N3
    
    # Ensure k_1 is calculated and available for simulations (time-varying mechanisms only)
    if 'tau' in params and 'k_max' in params and 'k_1' not in params:
        params['k_1'] = calculate_k1_from_params(params)
    
    # Create n0_list (thresholds for simulation)
    n0_list = [n1, n2, n3]
    
    return params, n0_list


def calculate_k1_from_params(params):
    """
    Calculate k_1 from k_max and tau parameters.
    
    Args:
        params (dict): Parameters containing k_max and tau
    
    Returns:
        float: Calculated k_1 value
    """
    if 'k_max' in params and 'tau' in params:
        return params['k_max'] / params['tau']
    elif 'k_1' in params:
        return params['k_1']
    else:
        raise ValueError("Cannot calculate k_1: missing k_max/tau or k_1 parameters")


def run_simulation_for_dataset(mechanism, params, n0_list, num_simulations=500):
    """
    Run simulations for a given mechanism and parameters.
    Handles both simple mechanisms (simple, fixed_burst, feedback_onion, fixed_burst_feedback_onion)
    and time-varying mechanisms (time_varying_k, time_varying_k_fixed_burst, etc.)
    
    OPTIMIZATION: Uses O(1) Beta sampling for 'simple', 'fixed_burst', 'time_varying_k', 
    and 'time_varying_k_fixed_burst' mechanisms, falling back to Gillespie for complex 
    mechanisms (feedback, time-varying feedback).
    
    Args:
        mechanism (str): Mechanism name
        params (dict): Parameter dictionary
        n0_list (list): List of threshold values [n01, n02, n03]
        num_simulations (int): Number of simulations to run
    
    Returns:
        tuple: (delta_t12_array, delta_t32_array) as numpy arrays, or (None, None) on failure
    """
    # Check if we can use the fast Beta method
    use_beta_method = mechanism in ['simple', 'fixed_burst', 'time_varying_k', 'time_varying_k_fixed_burst']
    
    if use_beta_method:
        # Import the fast Beta simulation method
        try:
            from FastBetaSimulation import simulate_batch
            
            # Prepare parameters for Beta method
            initial_states = np.array([params['N1'], params['N2'], params['N3']])
            n0_array = np.array(n0_list)
            
            # Prepare mechanism-specific parameters
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
            
            # Use the ultra-fast vectorized method
            # RUN INDEPENDENT SIMULATIONS FOR T12 AND T32
            # We need 2 * num_simulations total samples
            total_sims = 2 * num_simulations
            
            results = simulate_batch(
                mechanism=mechanism,
                initial_states=initial_states,
                n0_lists=n0_array,
                k=k,
                burst_size=burst_size,
                k_1=k_1,
                k_max=k_max,
                num_simulations=total_sims
            )
            
            # Split results for independent sampling
            # First half for T12, Second half for T32
            # Extract segregation times (columns are chromosomes 1, 2, 3)
            
            # Dataset A (for T12)
            t1_A = results[:num_simulations, 0]
            t2_A = results[:num_simulations, 1]
            # Dataset A T3 is irrelevant
            
            # Dataset B (for T32)
            # Dataset B T1 is irrelevant
            t2_B = results[num_simulations:, 1]
            t3_B = results[num_simulations:, 2]
            
            # Calculate time differences independently
            delta_t12_array = t1_A - t2_A
            delta_t32_array = t3_B - t2_B
            
            return delta_t12_array, delta_t32_array
            
        except ImportError as e:
            print(f"!!! FastBetaSimulation import failed: {e}")
            raise ImportError(f"FastBetaSimulation module not found. Gillespie fallback is disabled. Error: {e}")

    # Check if we can use the fast Feedback method (Vectorized Sum of Exponentials)
    use_feedback_method = mechanism in ['feedback_onion', 'fixed_burst_feedback_onion', 
                                        'time_varying_k_feedback_onion', 'time_varying_k_combined']
    
    if use_feedback_method:
        try:
            from FastFeedbackSimulation import simulate_batch_feedback
            
            initial_states = np.array([params['N1'], params['N2'], params['N3']])
            n0_array = np.array(n0_list)
            
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
                # burst_size already extracted above
            elif mechanism in ['time_varying_k_feedback_onion', 'time_varying_k_combined']:
                k_1 = calculate_k1_from_params(params)
                k_max = params['k_max']
            
            # RUN INDEPENDENT SIMULATIONS FOR T12 AND T32
            total_sims = 2 * num_simulations
                
            results = simulate_batch_feedback(
                mechanism=mechanism,
                initial_states=initial_states,
                n0_lists=n0_array,
                k=k,
                n_inner=n_inner,
                k_1=k_1,
                k_max=k_max,
                burst_size=burst_size,
                num_simulations=total_sims
            )
            
            # Split results for independent sampling
            # Dataset A (for T12)
            t1_A = results[:num_simulations, 0]
            t2_A = results[:num_simulations, 1]
            
            # Dataset B (for T32)
            t2_B = results[num_simulations:, 1]
            t3_B = results[num_simulations:, 2]
            
            delta_t12_array = t1_A - t2_A
            delta_t32_array = t3_B - t2_B
            
            return delta_t12_array, delta_t32_array
            
        except ImportError as e:
            print(f"\n!!!!!!!!!!!!!!!!! CRITICAL ERROR !!!!!!!!!!!!!!!!!")
            print(f"FastFeedbackSimulation import failed: {e}")
            print(f"Falling back to SLOW Gillespie simulation.")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            warnings.warn("FastFeedbackSimulation not found")
            raise ImportError(f"FastFeedbackSimulation module not found. Gillespie fallback is disabled. Error: {e}")

    
    # If neither method was used, raise an error
    raise ValueError(f"No fast simulation method available for mechanism '{mechanism}'. "
                   f"Standard Gillespie simulation has been disabled.")



# ======================== KDE BANDWIDTH CONFIGURATION ========================
# Global configuration for KDE bandwidth selection
# Options:
#   method='scott': Use Scott's rule (h = std * n^(-1/5)) - adaptive to sample size
#   method='fixed': Use a fixed bandwidth value
KDE_BANDWIDTH_CONFIG = {
    'method': 'scott',      # 'scott' or 'fixed'
    'fixed_value': 10.0     # Used when method='fixed'
}

def set_kde_bandwidth(method='scott', fixed_value=10.0):
    """
    Set the KDE bandwidth configuration globally.
    
    Args:
        method: 'scott' for Scott's rule (adaptive) or 'fixed' for fixed bandwidth
        fixed_value: Bandwidth value when method='fixed'
    
    Example:
        set_kde_bandwidth(method='fixed', fixed_value=15.0)
        set_kde_bandwidth(method='scott')
    """
    global KDE_BANDWIDTH_CONFIG
    if method not in ['scott', 'fixed']:
        raise ValueError("method must be 'scott' or 'fixed'")
    KDE_BANDWIDTH_CONFIG['method'] = method
    KDE_BANDWIDTH_CONFIG['fixed_value'] = fixed_value
    print(f"KDE bandwidth set to: method='{method}'" + 
          (f", fixed_value={fixed_value}" if method == 'fixed' else " (adaptive)"))

def get_kde_bandwidth():
    """Get current KDE bandwidth configuration."""
    return KDE_BANDWIDTH_CONFIG.copy()

# =============================================================================


def calculate_likelihood(exp_data, sim_data):
    """
    Calculate negative log-likelihood using KDE.
    
    Bandwidth selection is controlled by global KDE_BANDWIDTH_CONFIG:
    - 'scott': Scott's rule (h = std * n^(-1/5)) - adaptive to sample size
    - 'fixed': Use fixed bandwidth value
    
    Use set_kde_bandwidth() to change the configuration.
    
    Args:
        exp_data: Dictionary or array of experimental data
        sim_data: Dictionary or array of simulation data
        
    Returns:
        float: Negative log-likelihood
    """
    try:
        # Handle dictionary input
        if isinstance(exp_data, dict) and isinstance(sim_data, dict):
            nll_total = 0
            
            for key in ['delta_t12', 'delta_t32']:
                if key in exp_data and key in sim_data:
                    exp_values = np.asarray(exp_data[key]).flatten()
                    sim_values = np.asarray(sim_data[key]).flatten()
                    
                    # Remove non-finite values
                    exp_values = exp_values[np.isfinite(exp_values)]
                    sim_values = sim_values[np.isfinite(sim_values)]
                    
                    if len(exp_values) == 0 or len(sim_values) < 10:
                        return 1e6
                    
                    # Select bandwidth based on configuration
                    if KDE_BANDWIDTH_CONFIG['method'] == 'scott':
                        bandwidth = np.std(sim_values) * (len(sim_values) ** (-1/5))
                        bandwidth = max(bandwidth, 0.1)  # Minimum to avoid numerical issues
                    else:
                        bandwidth = KDE_BANDWIDTH_CONFIG['fixed_value']
                    
                    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
                    kde.fit(sim_values.reshape(-1, 1))
                    
                    # Calculate likelihood
                    log_densities = kde.score_samples(exp_values.reshape(-1, 1))
                    
                    # Robust KDE: Mix with uniform background to handle outliers
                    # This prevents infinite NLL for data points far from simulation range
                    # P(x) = (1 - epsilon) * KDE(x) + epsilon * P_background
                    epsilon = 1e-3
                    background_density = 1e-6
                    
                    densities = np.exp(log_densities)
                    robust_densities = (1 - epsilon) * densities + epsilon * background_density
                    log_densities = np.log(robust_densities)
                    
                    nll = -np.sum(log_densities)
                    nll_total += nll
            
            return nll_total
        
        # Handle array input
        else:
            exp_values = np.asarray(exp_data).flatten()
            sim_values = np.asarray(sim_data).flatten()
            
            exp_values = exp_values[np.isfinite(exp_values)]
            sim_values = sim_values[np.isfinite(sim_values)]
            
            if len(exp_values) == 0 or len(sim_values) < 10:
                return 1e6
            
            # Select bandwidth based on configuration
            if KDE_BANDWIDTH_CONFIG['method'] == 'scott':
                bandwidth = np.std(sim_values) * (len(sim_values) ** (-1/5))
                bandwidth = max(bandwidth, 0.1)
            else:
                bandwidth = KDE_BANDWIDTH_CONFIG['fixed_value']
            
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
            kde.fit(sim_values.reshape(-1, 1))
            
            log_densities = kde.score_samples(exp_values.reshape(-1, 1))
            
            # Robust KDE
            epsilon = 1e-3
            background_density = 1e-6
            
            densities = np.exp(log_densities)
            robust_densities = (1 - epsilon) * densities + epsilon * background_density
            log_densities = np.log(robust_densities)
            
            return -np.sum(log_densities)
    
    except Exception:
        return 1e6




def calculate_emd(exp_data, sim_data):
    """
    Calculate Earth Mover's Distance (Wasserstein Distance) between experimental and simulation data.
    
    This metric does not require bandwidth tuning and converges more stably than NLL.
    
    Args:
        exp_data: Dictionary or array of experimental data
        sim_data: Dictionary or array of simulation data
        
    Returns:
        float: Sum of Wasserstein distances across all datasets (in minutes)
    """
    try:
        # Handle dictionary input
        if isinstance(exp_data, dict) and isinstance(sim_data, dict):
            total_emd = 0
            
            for key in ['delta_t12', 'delta_t32']:
                if key in exp_data and key in sim_data:
                    exp_values = np.asarray(exp_data[key]).flatten()
                    sim_values = np.asarray(sim_data[key]).flatten()
                    
                    # Remove non-finite values
                    exp_values = exp_values[np.isfinite(exp_values)]
                    sim_values = sim_values[np.isfinite(sim_values)]
                    
                    if len(exp_values) == 0 or len(sim_values) == 0:
                        return 1e6  # Penalty for empty data
                    
                    emd = wasserstein_distance(exp_values, sim_values)
                    total_emd += emd
            
            return total_emd
        
        # Handle array input
        else:
            exp_values = np.asarray(exp_data).flatten()
            sim_values = np.asarray(sim_data).flatten()
            
            exp_values = exp_values[np.isfinite(exp_values)]
            sim_values = sim_values[np.isfinite(sim_values)]
            
            if len(exp_values) == 0 or len(sim_values) == 0:
                return 1e6
            
            return wasserstein_distance(exp_values, sim_values)
    
    except Exception:
        return 1e6


def calculate_wasserstein_p_value(sample1, sample2, num_permutations=1000, seed=None):
    """
    Calculate the p-value for the Wasserstein distance between two samples
    using a permutation test.
    
    The null hypothesis is that the two samples are drawn from the same distribution.
    A low p-value indicates that the distributions are significantly different.
    
    Args:
        sample1 (array-like): First sample (e.g., experimental data)
        sample2 (array-like): Second sample (e.g., simulation data)
        num_permutations (int): Number of permutations to perform (default: 1000)
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        tuple: (p_value, observed_distance)
    """
    rng = np.random.default_rng(seed)
    
    # Ensure inputs are 1D numpy arrays and remove NaNs
    u = np.asarray(sample1).flatten()
    v = np.asarray(sample2).flatten()
    u = u[np.isfinite(u)]
    v = v[np.isfinite(v)]
    
    if len(u) == 0 or len(v) == 0:
        return np.nan, np.nan
        
    # 1. Calculate observed Wasserstein distance
    obs_dist = wasserstein_distance(u, v)
    
    if obs_dist == 0:
        return 1.0, 0.0
        
    # 2. Pool the data
    pooled = np.concatenate([u, v])
    n = len(u)
    m = len(v)
    total_samples = n + m
    
    # 3. Permutation test
    count_geq = 0
    
    # Pre-allocate for performance (optional, but good for many perms)
    # Just loop is fine for 1000-10000
    
    for _ in range(num_permutations):
        # Shuffle the pooled data
        permuted = rng.permutation(pooled)
        
        # Split into two new samples of original sizes
        perm_u = permuted[:n]
        perm_v = permuted[n:]
        
        # Calculate statistic for permuted data
        perm_dist = wasserstein_distance(perm_u, perm_v)
        
        if perm_dist >= obs_dist:
            count_geq += 1
            
    # Calculate p-value
    # Add 1 to numerator and denominator to avoid p=0 (standard practice for permutation tests)
    p_value = (count_geq + 1) / (num_permutations + 1)
    
    return p_value, obs_dist


def get_parameter_bounds(mechanism):
    """
    Get parameter bounds for optimization.
    
    Args:
        mechanism (str): Mechanism name
    
    Returns:
        list: List of (min, max) tuples for each parameter
    """
    is_simple = mechanism in ['simple', 'fixed_burst', 'feedback_onion', 'fixed_burst_feedback_onion']
    
    # Base bounds: n2, N2, rate param(s), r21, r23, R21, R23
    bounds = [
        (0.0, 50.0),      # n2
        (50.0, 1000.0),   # N2
        (0.001, 0.1),      # k_max
        (2.0, 240.0),       # tau
        (0.25, 4.0),      # r21
        (0.25, 4.0),      # r23
        (0.4, 2.0),        # R21
        (0.5, 5.0),       # R23
    ]
    
    # Add mechanism-specific optional parameters
    if mechanism in ['fixed_burst', 'fixed_burst_feedback_onion', 
                      'time_varying_k_fixed_burst', 'time_varying_k_combined']:
        bounds.append((1.0, 50.0))  # burst_size
    if mechanism in ['feedback_onion', 'fixed_burst_feedback_onion',
                      'time_varying_k_feedback_onion', 'time_varying_k_combined']:
        bounds.append((1.0, 100.0))  # n_inner
    
    # Add mutant parameter bounds
    bounds.append((0.1, 0.7))   # alpha
    
    if is_simple:
        # Simple mechanisms now have 3 distinct beta_k parameters
        bounds.extend([
            (0.1, 1.0),   # beta_k1
            (0.1, 1.0),   # beta_k2
            (0.1, 1.0),   # beta_k3
        ])
    else:
        # Time-varying mechanisms retain original beta_k + beta_tau parameters
        bounds.extend([
            (0.1, 1.0),   # beta_k (for k_max)
            (2.0, 10.0),  # beta_tau
            (2.0, 20.0),  # beta_tau2
        ])
    
    return bounds


def get_parameter_names(mechanism):
    """
    Get parameter names for a given mechanism (time-varying mechanisms only).
    
    Args:
        mechanism (str): Mechanism name (must be time-varying)
    
    Returns:
        list: List of parameter names
    """
    time_varying_params = {
        'simple': ['n2', 'N2', 'k', 'r21', 'r23', 'R21', 'R23', 'alpha', 'beta_k1', 'beta_k2', 'beta_k3'],
        'fixed_burst': ['n2', 'N2', 'k', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'alpha', 'beta_k1', 'beta_k2', 'beta_k3'],
        'feedback_onion': ['n2', 'N2', 'k', 'r21', 'r23', 'R21', 'R23', 'n_inner', 'alpha', 'beta_k1', 'beta_k2', 'beta_k3'],
        'fixed_burst_feedback_onion': ['n2', 'N2', 'k', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'n_inner', 'alpha', 'beta_k1', 'beta_k2', 'beta_k3'],
        'time_varying_k': ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'alpha', 'beta_k', 'beta_tau', 'beta_tau2'],
        'time_varying_k_fixed_burst': ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'alpha', 'beta_k', 'beta_tau', 'beta_tau2'],
        'time_varying_k_feedback_onion': ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'n_inner', 'alpha', 'beta_k', 'beta_tau', 'beta_tau2'],
        'time_varying_k_combined': ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'n_inner', 'alpha', 'beta_k', 'beta_tau', 'beta_tau2'],
    }
    
    if mechanism not in time_varying_params:
        raise ValueError(f"Unknown mechanism: {mechanism}")
    
    return time_varying_params[mechanism]


def parse_parameters(params_vector, mechanism):
    """
    Parse parameter vector into a dictionary (time-varying mechanisms only).
    
    Args:
        params_vector (array): Parameter values
        mechanism (str): Mechanism name (must be time-varying)
    
    Returns:
        dict: Parameter dictionary with derived parameters
    """
    param_names = get_parameter_names(mechanism)
    param_dict = dict(zip(param_names, params_vector))
    
    # Calculate derived parameters
    # Calculate derived parameters
    param_dict['n1'] = max(param_dict['r21'] * param_dict['n2'], 1)
    param_dict['n3'] = max(param_dict['r23'] * param_dict['n2'], 1)
    param_dict['N1'] = max(param_dict['R21'] * param_dict['N2'], 1)
    param_dict['N3'] = max(param_dict['R23'] * param_dict['N2'], 1)
    
    if 'k_max' in param_dict and 'tau' in param_dict:
        param_dict['k_1'] = calculate_k1_from_params(param_dict)
    
    return param_dict


def save_optimization_results(mechanism, results, filename=None, selected_strains=None):
    """
    Save optimization results to a text file.
    
    Args:
        mechanism (str): Mechanism name
        results (dict): Optimization results
        filename (str): Output filename (optional)
        selected_strains (list): List of selected strain names (optional)
    """
    if filename is None:
        filename = f"simulation_optimized_parameters_{mechanism}.txt"
    
    try:
        with open(filename, 'w') as f:
            f.write("Simulation-based Optimization Results\n")
            f.write(f"Mechanism: {mechanism}\n")
            f.write(f"Negative Log-Likelihood: {results['fun']:.6f}\n")
            f.write(f"Converged: {results['success']}\n")
            f.write(f"Status: {results.get('message', 'Unknown')}\n")
            
            if selected_strains:
                f.write(f"Datasets: {', '.join(selected_strains)}\n")
            else:
                f.write("Datasets: wildtype, threshold, degrade, degradeAPC\n")
            f.write("\n")
            
            # Parse parameters
            param_dict = parse_parameters(results['x'], mechanism)
            
            # Write ratio-based parameters (what was optimized)
            f.write("Optimized Parameters (ratio-based):\n")
            param_names = get_parameter_names(mechanism)
            for name in param_names:
                if name in param_dict:
                    f.write(f"{name} = {param_dict[name]:.6f}\n")
            f.write("\n")
            
            # Write derived parameters
            f.write("Derived Parameters:\n")
            f.write(f"n1 = {param_dict['n1']:.6f}\n")
            f.write(f"n3 = {param_dict['n3']:.6f}\n")
            f.write(f"N1 = {param_dict['N1']:.6f}\n")
            f.write(f"N3 = {param_dict['N3']:.6f}\n")
            if 'k_1' in param_dict:
                f.write(f"k_1 = {param_dict['k_1']:.6f}\n")
            f.write("\n")
        
    
    except Exception:
        pass


def load_parameters(filename):
    """
    Load parameters from optimization results file.
    
    Args:
        filename (str): Path to parameter file
    
    Returns:
        dict: Dictionary of parameters, or None if error occurred
    """
    params = {}
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if ':' in line:
                parts = line.split(':', 1)
                try:
                    key = parts[0].strip()
                    val = float(parts[1].strip())
                    params[key] = val
                except ValueError:
                    pass
            elif '=' in line:
                parts = line.split('=', 1)
                try:
                    key = parts[0].strip()
                    val = float(parts[1].strip())
                    params[key] = val
                except ValueError:
                    pass
        
        # Calculate derived parameters if missing
        if 'r21' in params and 'n1' not in params:
            params['n1'] = max(params['r21'] * params['n2'], 1)
        if 'r23' in params and 'n3' not in params:
             params['n3'] = max(params['r23'] * params['n2'], 1)
        if 'R21' in params and 'N1' not in params:
             params['N1'] = max(params['R21'] * params['N2'], 1)
        if 'R23' in params and 'N3' not in params:
             params['N3'] = max(params['R23'] * params['N2'], 1)
             
        return params
    except Exception as e:
        print(f"Error loading parameters: {e}")
        return None
