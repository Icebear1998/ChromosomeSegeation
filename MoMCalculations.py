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
        if pair12:
            return compute_pdf_mom(mechanism, data, n_i, N_i, n_j, N_j, k,
                               n_inner1=mech_params['n_inner1'], n_inner2=mech_params['n_inner2'])
        else:
            return compute_pdf_mom(mechanism, data, n_i, N_i, n_j, N_j, k,
                               n_inner1=mech_params['n_inner3'], n_inner2=mech_params['n_inner2'])
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
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")
    

def compute_moments_mom(mechanism, n_i, N_i, n_j, N_j, k, burst_size=None, k_1=None, feedbackSteepness=None, feedbackThreshold=None, w1=None, w2=None, n_inner1=None, n_inner2=None, z1=None, z2=None):
    """
    Compute Method of Moments mean and variance for f_X = T_i - T_j.

    Args:
        mechanism (str): 'simple', 'fixed_burst', 'time_varying_k', 'feedback', 'feedback_linear', 'feedback_onion', 'fixed_burst_feedback_linear'.
        n_i, n_j (float): Threshold cohesin counts for chromosomes i and j.
        N_i, N_j (float): Initial cohesin counts for chromosomes i and j.
        k (float): Degradation rates (k_i, k_j for simple; lambda_i, lambda_j for fixed_burst).
        burst_size (float, optional): Burst size b for fixed_burst mechanism.
        w1, w2 (float, optional): Feedback parameters for feedback_linear mechanism.
        n_inner1, n_inner2 (float, optional): Inner parameters for feedback_onion mechanism.

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
        sum1_Ti = sum(1/m for m in range(int(n_i) + 1, int(N_i) + 1))
        sum1_Tj = sum(1/m for m in range(int(n_j) + 1, int(N_j) + 1))
        sum2_Ti = sum(1/(m**2) for m in range(int(n_i) + 1, int(N_i) + 1))
        sum2_Tj = sum(1/(m**2) for m in range(int(n_j) + 1, int(N_j) + 1))

        mean_Ti = sum1_Ti / k
        mean_Tj = sum1_Tj / k
        var_Ti = sum2_Ti / (k**2)
        var_Tj = sum2_Tj / (k**2)

    elif mechanism == 'fixed_burst':
        if burst_size is None or burst_size <= 0 or math.isnan(burst_size):
            print("Invalid burst size 123")
            return np.inf, np.inf
        if N_i is None or N_j is None or math.isnan(N_i) or math.isnan(N_j):
            print("Invalid N_i or N_j")
            return np.inf, np.inf
        if n_i is None or n_j is None or math.isnan(n_i) or math.isnan(n_j):
            print("Invalid n_i or n_j")
            return np.inf, np.inf
        # print("N_i, N_j, n_i, n_j, burst_size", N_i, N_j, n_i, n_j, burst_size)
        # Ensure number of bursts is non-negative and valid
        num_bursts_i = max(0, int(np.ceil((N_i - n_i) / burst_size)))
        num_bursts_j = max(0, int(np.ceil((N_j - n_j) / burst_size)))

        if num_bursts_i == 0 or num_bursts_j == 0:
            return 0.0, 0.0

        # Compute moments with safe division
        mean_Ti = sum(1 / (k * max(1e-10, (N_i - m * burst_size)))
                      for m in range(num_bursts_i))
        mean_Tj = sum(1 / (k * max(1e-10, (N_j - m * burst_size)))
                      for m in range(num_bursts_j))
        var_Ti = sum(1 / (k * max(1e-10, (N_i - m * burst_size)))
                     ** 2 for m in range(num_bursts_i))
        var_Tj = sum(1 / (k * max(1e-10, (N_j - m * burst_size)))
                     ** 2 for m in range(num_bursts_j))

    elif mechanism == 'time_varying_k':
        if k_1 is None:
            raise ValueError(
                "k_1 must be provided for time_varying mechanism.")

        def compute_tm(m, N, k, k1):
            if k1 == 0:
                return (1 / k) * np.log(N / m)
            discriminant = k**2 + 2 * k1 * np.log(N / m)
            return (-k + np.sqrt(discriminant)) / k1

        def kt(t, k, k1):
            return k + k1 * t

        def mom_time_varying_k(N, n, k, k1):
            expected_time = 0
            variance = 0
            for m in range(int(n + 1), int(N + 1)):
                t_m = compute_tm(m, N, k, k1)
                rate = kt(t_m, k, k1) * m
                tau_mean = 1 / rate
                expected_time += tau_mean
                variance += tau_mean**2
            return expected_time, variance

        mean_Ti, var_Ti = mom_time_varying_k(N_i, n_i, k, k_1)
        mean_Tj, var_Tj = mom_time_varying_k(N_j, n_j, k, k_1)

    elif mechanism == 'feedback_linear':
        if w1 is None or w2 is None:
            print(w1, w2)
            raise ValueError(
                "Parameters 'w1' and 'w2' must be provided for the feedback linear mechanism.")

        def mom_feedback_linear(N, n, k, w=100):
            mean_T = 0.0
            var_T = 0.0
            for m in range(int(n) + 1, int(N) + 1):
                W_m = 1 - w*m
                E_tau_m = 1/(W_m*m)
                mean_T += E_tau_m
                var_T += E_tau_m**2
            mean_T /= k
            var_T /= k**2
            return mean_T, var_T

        mean_Ti, var_Ti = mom_feedback_linear(
            N_i, n_i, k, w1)
        mean_Tj, var_Tj = mom_feedback_linear(
            N_j, n_j, k, w2)

    elif mechanism == 'feedback_onion':
        if n_inner1 is None or n_inner2 is None:
            raise ValueError(
                "Parameters 'n_inner1' and 'n_inner2' must be provided for the feedback onion mechanism.")

        def mom_feedback_onion(N, n, k, n_inner):
            mean_T = 0.0
            var_T = 0.0
            for m in range(int(n) + 1, int(N) + 1):
                if N > n_inner:
                    W_m = (N / n_inner) ** (-1/3)
                else:
                    W_m = 1.0
                E_tau_m = 1/(W_m*m)
                mean_T += E_tau_m
                var_T += E_tau_m**2
            mean_T /= k
            var_T /= k**2
            return mean_T, var_T

        mean_Ti, var_Ti = mom_feedback_onion(
            N_i, n_i, k, n_inner1)
        mean_Tj, var_Tj = mom_feedback_onion(
            N_j, n_j, k, n_inner2)

    elif mechanism == 'feedback_zipper':
        if z1 is None or z2 is None:
            raise ValueError(
                "Parameters 'z1' and 'z2' must be provided for the feedback zipper mechanism.")

        def mom_feedback_zipper(N, n, k, z):
            mean_T = 0.0
            var_T = 0.0
            for m in range(int(n) + 1, int(N) + 1):
                W_m = z / N
                E_tau_m = 1/(W_m*m)
                mean_T += E_tau_m
                var_T += E_tau_m**2
            mean_T /= k
            var_T /= k**2
            return mean_T, var_T

        mean_Ti, var_Ti = mom_feedback_zipper(
            N_i, n_i, k, z1)
        mean_Tj, var_Tj = mom_feedback_zipper(
            N_j, n_j, k, z2)

    elif mechanism == 'feedback':
        if feedbackSteepness is None or feedbackThreshold is None:
            raise ValueError(
                "Parameters 'a' and 'm_threshold' must be provided for the feedback mechanism.")

        def mom_feedback(N, n, k, feedbackSteepness, feedbackThreshold):
            mean_T = 0.0
            var_T = 0.0
            for m in range(int(n) + 1, int(N) + 1):
                W_m = 1 / (1 + np.exp(feedbackSteepness *
                           (m - feedbackThreshold)))
                E_tau_m = 1/(W_m*m)
                mean_T += E_tau_m
                var_T += E_tau_m**2
            mean_T /= k
            var_T /= k**2
            return mean_T, var_T

        mean_Ti, var_Ti = mom_feedback(
            N_i, n_i, k, feedbackSteepness, feedbackThreshold)
        mean_Tj, var_Tj = mom_feedback(
            N_j, n_j, k, feedbackSteepness, feedbackThreshold)

    elif mechanism == 'fixed_burst_feedback_linear':
        if burst_size is None or burst_size <= 0 or math.isnan(burst_size):
            print("Invalid burst size")
            return np.inf, np.inf
        if w1 is None or w2 is None:
            print(w1, w2)
            raise ValueError(
                "Parameters 'w1' and 'w2' must be provided for the feedback linear mechanism.")

        # Ensure number of bursts is non-negative and valid
        if N_i is None or N_j is None or math.isnan(N_i) or math.isnan(N_j) or burst_size is None:
            print("N_i, N_j, n_i, n_j, burst_size", N_i, N_j, n_i, n_j, burst_size)
        num_bursts_i = max(0, int(np.ceil((N_i - n_i) / burst_size)))
        num_bursts_j = max(0, int(np.ceil((N_j - n_j) / burst_size)))

        if num_bursts_i == 0 or num_bursts_j == 0:
            return 0.0, 0.0

        # Compute moments with both burst size and feedback effects
        mean_Ti = 0.0
        mean_Tj = 0.0
        var_Ti = 0.0
        var_Tj = 0.0

        # For chromosome i
        for m in range(num_bursts_i):
            current_N = N_i - m * burst_size
            W_m = 1 - w1 * current_N
            if W_m <= 0:  # Ensure positive rate
                return np.inf, np.inf
            E_tau_m = 1 / (W_m * current_N)
            mean_Ti += E_tau_m
            var_Ti += E_tau_m**2

        # For chromosome j
        for m in range(num_bursts_j):
            current_N = N_j - m * burst_size
            W_m = 1 - w2 * current_N
            if W_m <= 0:  # Ensure positive rate
                return np.inf, np.inf
            E_tau_m = 1 / (W_m * current_N)
            mean_Tj += E_tau_m
            var_Tj += E_tau_m**2

        # Scale by degradation rate
        mean_Ti /= k
        mean_Tj /= k
        var_Ti /= k**2
        var_Tj /= k**2

    else:
        raise ValueError(
            "Mechanism must be 'simple', 'fixed_burst', 'time_varying_k', 'feedback', 'feedback_linear', 'feedback_onion', 'feedback_zipper', or 'fixed_burst_feedback_linear'.")

    mean_X = mean_Ti - mean_Tj
    var_X = var_Ti + var_Tj
    return mean_X, var_X


def compute_pdf_mom(mechanism, x_grid, n_i, N_i, n_j, N_j, k, burst_size=None, k_1=None, feedbackSteepness=None, feedbackThreshold=None, w1=None, w2=None, n_inner1=None, n_inner2=None, z1=None, z2=None):
    mean_X, var_X = compute_moments_mom(mechanism, n_i, N_i, n_j, N_j, k, burst_size=burst_size,
                                        k_1=k_1, feedbackSteepness=feedbackSteepness, feedbackThreshold=feedbackThreshold, w1=w1, w2=w2, n_inner1=n_inner1, n_inner2=n_inner2, z1=z1, z2=z2)
    if np.isinf(mean_X) or np.isinf(var_X):
        # Small positive value to avoid log(0)
        return np.full_like(x_grid, 1e-10)
    else:
        return norm.pdf(x_grid, loc=mean_X, scale=np.sqrt(var_X))
