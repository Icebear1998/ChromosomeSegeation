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
        final_state_i = max(1, int(round(n_i)))
        final_state_j = max(1, int(round(n_j)))
        
        sum1_Ti = sum(1/m for m in range(final_state_i + 1, int(N_i) + 1))
        sum1_Tj = sum(1/m for m in range(final_state_j + 1, int(N_j) + 1))
        sum2_Ti = sum(1/(m**2) for m in range(final_state_i + 1, int(N_i) + 1))
        sum2_Tj = sum(1/(m**2) for m in range(final_state_j + 1, int(N_j) + 1))

        mean_Ti = sum1_Ti / k
        mean_Tj = sum1_Tj / k
        var_Ti = sum2_Ti / (k**2)
        var_Tj = sum2_Tj / (k**2)

    elif mechanism == 'fixed_burst':
        if burst_size is None or burst_size <= 0 or math.isnan(burst_size):
            print(f"Invalid burst size: {burst_size} (None: {burst_size is None}, <=0: {burst_size <= 0 if burst_size is not None else 'N/A'}, NaN: {math.isnan(burst_size) if burst_size is not None else 'N/A'})")
            return np.inf, np.inf
        if N_i is None or N_j is None or math.isnan(N_i) or math.isnan(N_j):
            print("Invalid N_i or N_j")
            return np.inf, np.inf
        if n_i is None or n_j is None or math.isnan(n_i) or math.isnan(n_j):
            print("Invalid n_i or n_j")
            return np.inf, np.inf
        
        # FIXED: Use consistent rounding with simple model
        # Sum over cohesin counts m (same as simple), with burst-size weighting
        final_state_i = max(1, int(round(n_i)))
        final_state_j = max(1, int(round(n_j)))
        
        mean_Ti = 0.0
        mean_Tj = 0.0
        var_Ti = 0.0
        var_Tj = 0.0
        
        # Sum over cohesin counts m from n_i to N_i (same range as simple model)
        for m in range(final_state_i + 1, int(N_i) + 1):
            # At cohesin count m, rate is k*m, but removal happens in bursts
            # Expected time for burst: burst_size / (k*m)
            # When burst_size=1: this becomes 1/(k*m), identical to simple
            E_tau_m = burst_size / (k * m)
            mean_Ti += E_tau_m / burst_size
            var_Ti += (E_tau_m / burst_size) ** 2
        
        for m in range(final_state_j + 1, int(N_j) + 1):
            E_tau_m = burst_size / (k * m)
            mean_Tj += E_tau_m / burst_size
            var_Tj += (E_tau_m / burst_size) ** 2


    elif mechanism == 'feedback_onion':
        if n_inner is None:
            raise ValueError(
                "Parameter 'n_inner' must be provided for the feedback onion mechanism.")

        def mom_feedback_onion(N, n, k, n_inner):
            """
            FIXED: W_m now depends on current cohesin count m, not initial N.
            When m > n_inner: degradation is slower (W_m < 1)
            When m <= n_inner: normal degradation (W_m = 1)
            This creates true state-dependent feedback.
            """
            # FIXED: Use same rounding as simple model for consistency
            final_state = max(1, int(round(n)))
            
            mean_T = 0.0
            var_T = 0.0
            for m in range(final_state + 1, int(N) + 1):
                # Feedback depends on CURRENT cohesin count m, not initial N
                if m > n_inner:
                    W_m = (m / n_inner) ** (-1/3)  # Slower degradation when m > n_inner
                else:
                    W_m = 1.0  # Normal degradation when m <= n_inner
                E_tau_m = 1/(W_m * k * m)  # Include k here for correct units
                mean_T += E_tau_m
                var_T += E_tau_m**2
            return mean_T, var_T

        mean_Ti, var_Ti = mom_feedback_onion(
            N_i, n_i, k, n_inner)
        mean_Tj, var_Tj = mom_feedback_onion(
            N_j, n_j, k, n_inner)


    elif mechanism == 'fixed_burst_feedback_onion':
        if burst_size is None or burst_size <= 0 or math.isnan(burst_size):
            print(f"Invalid burst size: {burst_size}")
            return np.inf, np.inf
        if n_inner is None:
            raise ValueError(
                "Parameter 'n_inner' must be provided for the fixed burst feedback onion mechanism.")

        # FIXED: Use consistent approach - sum over cohesin counts with state-dependent feedback
        final_state_i = max(1, int(round(n_i)))
        final_state_j = max(1, int(round(n_j)))
        
        mean_Ti = 0.0
        mean_Tj = 0.0
        var_Ti = 0.0
        var_Tj = 0.0
        
        # Sum over cohesin counts m (same range as simple/feedback_onion)
        for m in range(final_state_i + 1, int(N_i) + 1):
            # Feedback depends on CURRENT cohesin count m
            if m > n_inner:
                W_m = (m / n_inner) ** (-1/3)  # Slower degradation when m > n_inner
            else:
                W_m = 1.0  # Normal degradation when m <= n_inner
            # Time for burst at count m with feedback: burst_size / (W_m * k * m)
            # When burst_size=1: becomes 1/(W_m*k*m), identical to feedback_onion
            E_tau_m = burst_size / (W_m * k * m)
            mean_Ti += E_tau_m / burst_size
            var_Ti += (E_tau_m / burst_size) ** 2
        
        for m in range(final_state_j + 1, int(N_j) + 1):
            if m > n_inner:
                W_m = (m / n_inner) ** (-1/3)
            else:
                W_m = 1.0
            E_tau_m = burst_size / (W_m * k * m)
            mean_Tj += E_tau_m / burst_size
            var_Tj += (E_tau_m / burst_size) ** 2

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
