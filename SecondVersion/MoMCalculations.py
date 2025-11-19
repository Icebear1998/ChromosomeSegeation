import numpy as np
from scipy.stats import norm
import math


def compute_pdf_for_mechanism(mechanism, data, n_i, N_i, n_j, N_j, k, mech_params, pair12 = True):
    """
    Compute PDF for any mechanism with appropriate parameters.
    """
    if mechanism == 'simple':
        return compute_pdf_mom(mechanism, data, n_i, N_i, n_j, N_j, k)
    elif mechanism == 'fixed_burst':
        return compute_pdf_mom(mechanism, data, n_i, N_i, n_j, N_j, k,
                               burst_size=mech_params['burst_size'])
    elif mechanism == 'time_varying_k':
        return compute_pdf_mom(mechanism, data, n_i, N_i, n_j, N_j, k,
                               k_1=mech_params['k_1'])
    elif mechanism == 'feedback':
        return compute_pdf_mom(mechanism, data, n_i, N_i, n_j, N_j, k,
                               feedbackSteepness=mech_params['feedbackSteepness'],
                               feedbackThreshold=mech_params['feedbackThreshold'])
    elif mechanism == 'feedback_linear':
        if pair12:
            return compute_pdf_mom(mechanism, data, n_i, N_i, n_j, N_j, k,
                               w1=mech_params['w1'], w2=mech_params['w2'])
        else:
            return compute_pdf_mom(mechanism, data, n_i, N_i, n_j, N_j, k,
                               w1=mech_params['w3'], w2=mech_params['w2'])
    elif mechanism == 'feedback_onion':
        return compute_pdf_mom(mechanism, data, n_i, N_i, n_j, N_j, k,
                               n_inner=mech_params['n_inner'])
    elif mechanism == 'feedback_zipper':
        if pair12:
            return compute_pdf_mom(mechanism, data, n_i, N_i, n_j, N_j, k,
                               z1=mech_params['z1'], z2=mech_params['z2'])
        else:
            return compute_pdf_mom(mechanism, data, n_i, N_i, n_j, N_j, k,
                               z1=mech_params['z3'], z2=mech_params['z2'])
    elif mechanism == 'fixed_burst_feedback_linear':
        if pair12:
            return compute_pdf_mom(mechanism, data, n_i, N_i, n_j, N_j, k,
                               burst_size=mech_params['burst_size'],
                               w1=mech_params['w1'], w2=mech_params['w2'])
        else:
            return compute_pdf_mom(mechanism, data, n_i, N_i, n_j, N_j, k,
                               burst_size=mech_params['burst_size'],
                               w1=mech_params['w3'], w2=mech_params['w2'])
    elif mechanism == 'fixed_burst_feedback_onion':
        return compute_pdf_mom(mechanism, data, n_i, N_i, n_j, N_j, k,
                               burst_size=mech_params['burst_size'],
                               n_inner=mech_params['n_inner'])
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")
    

def compute_moments_mom(mechanism, n_i, N_i, n_j, N_j, k, burst_size=None, k_1=None, feedbackSteepness=None, feedbackThreshold=None, w1=None, w2=None, n_inner=None, z1=None, z2=None):
    """
    Compute Method of Moments mean and variance for f_X = T_i - T_j.

    Args:
        mechanism (str): 'simple', 'fixed_burst', 'time_varying_k', 'feedback', 'feedback_linear', 'feedback_onion', 'fixed_burst_feedback_linear', 'fixed_burst_feedback_onion'.
        n_i, n_j (float): Threshold cohesin counts for chromosomes i and j.
        N_i, N_j (float): Initial cohesin counts for chromosomes i and j.
        k (float): Degradation rates (k_i, k_j for simple; lambda_i, lambda_j for fixed_burst).
        burst_size (float, optional): Burst size b for fixed_burst mechanism.
        w1, w2 (float, optional): Feedback parameters for feedback_linear mechanism.
        n_inner (float, optional): Inner parameter for feedback_onion mechanism.

    Returns:
        tuple: (mean_X, var_X) for f_X = T_i - T_j.
    """
    # Input validation for all mechanisms
    if n_i < 0 or n_j < 0 or N_i <= 0 or N_j <= 0 or k <= 0:
        return np.inf, np.inf

    # Ensure inputs are valid
    if n_i >= N_i or n_j >= N_j:
        raise ValueError(
            "Final cohesin counts must be less than initial counts.")

    if mechanism == 'simple':
        # Simple Model: Harmonic sums
        # Use proper rounding for threshold values: n=1.9 rounds to 2, n=1.2 rounds to 1
        final_state_i = max(1, round(n_i))
        final_state_j = max(1, round(n_j))
        
        sum1_Ti = sum(1/m for m in range(final_state_i + 1, int(N_i) + 1))
        sum1_Tj = sum(1/m for m in range(final_state_j + 1, int(N_j) + 1))
        sum2_Ti = sum(1/(m**2) for m in range(final_state_i + 1, int(N_i) + 1))
        sum2_Tj = sum(1/(m**2) for m in range(final_state_j + 1, int(N_j) + 1))

        mean_Ti = sum1_Ti / k
        mean_Tj = sum1_Tj / k
        var_Ti = sum2_Ti / (k**2)
        var_Tj = sum2_Tj / (k**2)
    
    # elif mechanism == 'fixed_burst':
    #     if burst_size is None or burst_size <= 0 or math.isnan(burst_size):
    #         print(f"Invalid burst size: {burst_size} (None: {burst_size is None}, <=0: {burst_size <= 0 if burst_size is not None else 'N/A'}, NaN: {math.isnan(burst_size) if burst_size is not None else 'N/A'})")
    #         return np.inf, np.inf
    #     if N_i is None or N_j is None or math.isnan(N_i) or math.isnan(N_j):
    #         print("Invalid N_i or N_j")
    #         return np.inf, np.inf
    #     if n_i is None or n_j is None or math.isnan(n_i) or math.isnan(n_j):
    #         print("Invalid n_i or n_j")
    #         return np.inf, np.inf
        
    #     # print("N_i, N_j, n_i, n_j, burst_size", N_i, N_j, n_i, n_j, burst_size)
    #     # Ensure number of bursts is non-negative and valid
    #     num_bursts_i = max(0, int(np.ceil((N_i - n_i) / burst_size)))
    #     num_bursts_j = max(0, int(np.ceil((N_j - n_j) / burst_size)))

    #     if num_bursts_i == 0 or num_bursts_j == 0:
    #         return 0.0, 0.0

    #     # Compute moments with safe division
    #     mean_Ti = sum(1 / (k * max(1e-10, (N_i - m * burst_size)))
    #                   for m in range(num_bursts_i))
    #     mean_Tj = sum(1 / (k * max(1e-10, (N_j - m * burst_size)))
    #                   for m in range(num_bursts_j))
    #     var_Ti = sum(1 / (k * max(1e-10, (N_i - m * burst_size)))
    #                  ** 2 for m in range(num_bursts_i))
    #     var_Tj = sum(1 / (k * max(1e-10, (N_j - m * burst_size)))
    #                  ** 2 for m in range(num_bursts_j))

    elif mechanism == 'fixed_burst':
        # Validate burst_size
        if burst_size is None or burst_size <= 0 or math.isnan(burst_size):
            return np.inf, np.inf
        
        # Fixed burst with FRACTIONAL BURST ADJUSTMENT for continuous burst_size.
        # Instead of integer number of bursts, use fractional events.
        # The last burst can be partial (e.g., 0.5 of a burst).
        
        # Round n and N (cohesins are discrete)
        n_i_rounded = max(1, round(n_i))
        N_i_rounded = max(n_i_rounded + 1, round(N_i))
        n_j_rounded = max(1, round(n_j))
        N_j_rounded = max(n_j_rounded + 1, round(N_j))
        
        # Number of cohesins to degrade
        delta_i = N_i_rounded - n_i_rounded
        delta_j = N_j_rounded - n_j_rounded
        
        # Fractional number of bursts (keep as float!)
        num_bursts_i = delta_i / burst_size
        num_bursts_j = delta_j / burst_size
        
        # Full bursts (integer part)
        full_bursts_i = int(np.floor(num_bursts_i))
        full_bursts_j = int(np.floor(num_bursts_j))
        
        # Fractional part of last burst
        frac_burst_i = num_bursts_i - full_bursts_i
        frac_burst_j = num_bursts_j - full_bursts_j
        
        # Sum over full bursts
        mean_Ti = sum(1 / (k * max(1e-10, float(N_i_rounded - m * burst_size)))
                      for m in range(full_bursts_i))
        mean_Tj = sum(1 / (k * max(1e-10, float(N_j_rounded - m * burst_size)))
                      for m in range(full_bursts_j))
        
        # Add fractional contribution from last partial burst
        if frac_burst_i > 1e-6:  # Only add if there's a meaningful fraction
            remaining_i = N_i_rounded - full_bursts_i * burst_size
            if remaining_i > 1e-10:
                mean_Ti += frac_burst_i / (k * remaining_i)
        
        if frac_burst_j > 1e-6:
            remaining_j = N_j_rounded - full_bursts_j * burst_size
            if remaining_j > 1e-10:
                mean_Tj += frac_burst_j / (k * remaining_j)
        
        # Same for variance
        var_Ti = sum((1 / (k * max(1e-10, float(N_i_rounded - m * burst_size))))**2
                     for m in range(full_bursts_i))
        var_Tj = sum((1 / (k * max(1e-10, float(N_j_rounded - m * burst_size))))**2
                     for m in range(full_bursts_j))
        
        # Add fractional variance contribution
        if frac_burst_i > 1e-6:
            remaining_i = N_i_rounded - full_bursts_i * burst_size
            if remaining_i > 1e-10:
                var_Ti += frac_burst_i * (1 / (k * remaining_i))**2
        
        if frac_burst_j > 1e-6:
            remaining_j = N_j_rounded - full_bursts_j * burst_size
            if remaining_j > 1e-10:
                var_Tj += frac_burst_j * (1 / (k * remaining_j))**2

    # """
    # Fixed burst with continuous burst_size via mixture interpretation.
    
    # burst_size = 1.5 means: 50% bursts of size 1, 50% bursts of size 2
    # burst_size = 2.3 means: 70% bursts of size 2, 30% bursts of size 3
    # """
    # # Decompose into integer components
    # burst_lower = int(np.floor(burst_size))
    # burst_upper = int(np.ceil(burst_size))
    
    # # Handle edge case where burst_size is already integer
    # if burst_lower == burst_upper:
    #     return mom_fixed_burst_integer(n_i, N_i, n_j, N_j, k, burst_lower)
    
    #     # Weight for mixture: how much of upper vs lower
    #     weight_upper = burst_size - burst_lower
    #     weight_lower = 1.0 - weight_upper
        
    #     # Compute moments for both integer burst sizes
    #     mean_Ti_lower, var_Ti_lower, mean_Tj_lower, var_Tj_lower = \
    #         mom_fixed_burst_integer(n_i, N_i, n_j, N_j, k, burst_lower)
    #     mean_Ti_upper, var_Ti_upper, mean_Tj_upper, var_Tj_upper = \
    #         mom_fixed_burst_integer(n_i, N_i, n_j, N_j, k, burst_upper)
        
    #     # Mix the moments
    #     mean_Ti = weight_lower * mean_Ti_lower + weight_upper * mean_Ti_upper
    #     mean_Tj = weight_lower * mean_Tj_lower + weight_upper * mean_Tj_upper
        
    #     # For variance, need to account for variance of mixture:
    #     # Var(mixture) = E[Var] + Var[E]
    #     var_Ti = (weight_lower * (var_Ti_lower + mean_Ti_lower**2) + 
    #             weight_upper * (var_Ti_upper + mean_Ti_upper**2) - mean_Ti**2)
    #     var_Tj = (weight_lower * (var_Tj_lower + mean_Tj_lower**2) + 
    #             weight_upper * (var_Tj_upper + mean_Tj_upper**2) - mean_Tj**2)    


    elif mechanism == 'feedback_onion':
        if n_inner is None:
            raise ValueError(
                "Parameter 'n_inner' must be provided for the feedback onion mechanism.")

        def mom_feedback_onion(N, n, k, n_inner):
            mean_T = 0.0
            var_T = 0.0
            for m in range(int(n) + 1, int(N) + 1):
                if m > n_inner:
                    W_m = (m / max(n_inner, 1e-10)) ** (-1/3)
                else:
                    W_m = 1.0
                E_tau_m = 1 / (W_m * m)
                mean_T += E_tau_m
                var_T += E_tau_m**2
            mean_T /= k
            var_T /= k**2
            return mean_T, var_T

        mean_Ti, var_Ti = mom_feedback_onion(
            N_i, n_i, k, n_inner)
        mean_Tj, var_Tj = mom_feedback_onion(
            N_j, n_j, k, n_inner)


    elif mechanism == 'fixed_burst_feedback_onion':
        if burst_size is None or burst_size <= 0 or math.isnan(burst_size):
            return np.inf, np.inf
        if n_inner is None:
            raise ValueError(
                "Parameter 'n_inner' must be provided for the fixed burst feedback onion mechanism.")

        # For fixed_burst_feedback_onion:
        # - Propensity at state m: k * W_m * m
        # - Each event removes burst_size cohesins
        # - We need to sum over BURST EVENTS, not individual cohesins
        
        final_state_i = max(1, int(round(n_i)))
        final_state_j = max(1, int(round(n_j)))
        
        mean_Ti = 0.0
        mean_Tj = 0.0
        var_Ti = 0.0
        var_Tj = 0.0
        
        # Sum over burst events for chromosome i
        # Start at N_i and remove burst_size cohesins each step
        current_state = int(N_i)
        while current_state > final_state_i:
            # Calculate feedback weight at current state
            if current_state > n_inner:
                W_m = (current_state / max(n_inner, 1e-10)) ** (-1/3)
            else:
                W_m = 1.0
            
            # Expected time for one burst event at this state
            # Propensity = k * W_m * current_state
            # Expected time = 1 / propensity
            E_tau_event = 1.0 / (W_m * k * max(current_state, 1e-10))
            mean_Ti += E_tau_event
            var_Ti += E_tau_event ** 2
            
            # Move to next state after burst
            current_state -= burst_size
        
        # Sum over burst events for chromosome j
        current_state = int(N_j)
        while current_state > final_state_j:
            # Calculate feedback weight at current state
            if current_state > n_inner:
                W_m = (current_state / max(n_inner, 1e-10)) ** (-1/3)
            else:
                W_m = 1.0
            
            # Expected time for one burst event at this state
            E_tau_event = 1.0 / (W_m * k * max(current_state, 1e-10))
            mean_Tj += E_tau_event
            var_Tj += E_tau_event ** 2
            
            # Move to next state after burst
            current_state -= burst_size

    else:
        raise ValueError(
            "Mechanism must be 'simple', 'fixed_burst', 'time_varying_k', 'feedback', 'feedback_linear', 'feedback_onion', 'feedback_zipper', 'fixed_burst_feedback_linear', or 'fixed_burst_feedback_onion'.")

    mean_X = mean_Ti - mean_Tj
    var_X = var_Ti + var_Tj
    return mean_X, var_X


def compute_pdf_mom(mechanism, x_grid, n_i, N_i, n_j, N_j, k, burst_size=None, k_1=None, feedbackSteepness=None, feedbackThreshold=None, w1=None, w2=None, n_inner=None, z1=None, z2=None):
    mean_X, var_X = compute_moments_mom(mechanism, n_i, N_i, n_j, N_j, k, burst_size=burst_size,
                                        k_1=k_1, feedbackSteepness=feedbackSteepness, feedbackThreshold=feedbackThreshold, w1=w1, w2=w2, n_inner=n_inner, z1=z1, z2=z2)
    if np.isinf(mean_X) or np.isinf(var_X):
        # Small positive value to avoid log(0)
        return np.full_like(x_grid, 1e-10)
    else:
        return norm.pdf(x_grid, loc=mean_X, scale=np.sqrt(var_X))
