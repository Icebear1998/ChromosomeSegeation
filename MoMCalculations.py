import numpy as np
from scipy.stats import norm
import math


def compute_moments_mom(mechanism, n_i, N_i, n_j, N_j, k, burst_size=None, k_1=None, feedbackSteepness=None, feedbackThreshold=None, w1=None, w2=None):
    """
    Compute Method of Moments mean and variance for f_X = T_i - T_j.

    Args:
        mechanism (str): 'simple', 'fixed_burst', 'time_varying_k', 'feedback'.
        n_i, n_j (float): Threshold cohesin counts for chromosomes i and j.
        N_i, N_j (float): Initial cohesin counts for chromosomes i and j.
        k (float): Degradation rates (k_i, k_j for simple; lambda_i, lambda_j for fixed_burst).
        burst_size (float, optional): Burst size b for fixed_burst mechanism.

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

    else:
        raise ValueError(
            "Mechanism must be 'simple', 'fixed_burst', 'time_varying_k', or 'feedback'.")

    mean_X = mean_Ti - mean_Tj
    var_X = var_Ti + var_Tj
    return mean_X, var_X


def compute_pdf_mom(mechanism, x_grid, n_i, N_i, n_j, N_j, k, burst_size=None, k_1=None, feedbackSteepness=None, feedbackThreshold=None, w1=None, w2=None):
    mean_X, var_X = compute_moments_mom(mechanism, n_i, N_i, n_j, N_j, k, burst_size=burst_size,
                                        k_1=k_1, feedbackSteepness=feedbackSteepness, feedbackThreshold=feedbackThreshold, w1=w1, w2=w2)
    if np.isinf(mean_X) or np.isinf(var_X):
        # Small positive value to avoid log(0)
        return np.full_like(x_grid, 1e-10)
    else:
        return norm.pdf(x_grid, loc=mean_X, scale=np.sqrt(var_X))
