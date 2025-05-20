import numpy as np
from scipy.stats import norm
import math

def compute_moments_mom(mechanism, n_i, N_i, n_j, N_j, k, burst_size=None, mean_burst_size=None, var_burst_size=None, k_0=None, k_1=None):
    """
    Compute Method of Moments mean and variance for f_X = T_i - T_j.
    
    Args:
        mechanism (str): 'simple', 'fixed_burst', or 'random_normal_burst'.
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
    
    elif mechanism == 'k_change':
        
        # Compute deterministic time T_i^* and T_j^*
        def solve_time(N, n, k0, k1):
            if k1 == 0:
                return (np.log(N / n) / k0) if k0 > 0 else np.inf
            discriminant = k0**2 + 2 * k1 * np.log(N / n)
            if discriminant < 0:
                return np.inf
            return (-k0 + np.sqrt(discriminant)) / k1 if k1 > 0 else np.inf

        T_i_star = solve_time(N_i, n_i, k_0, k_1)
        T_j_star = solve_time(N_j, n_j, k_0, k_1)

        # Mean approximation using deterministic times
        mean_Ti = T_i_star
        mean_Tj = T_j_star

        # Variance approximation using effective rate at midpoint
        if T_i_star > 0 and k_0 > 0:
            k_mid_i = k_0 + k_1 * (T_i_star / 2)
            var_Ti = sum(1 / ((k_mid_i * m)**2) for m in range(int(n_i) + 1, int(N_i) + 1))
        else:
            var_Ti = np.inf

        if T_j_star > 0 and k_0 > 0:
            k_mid_j = k_0 + k_1 * (T_j_star / 2)
            var_Tj = sum(1 / ((k_mid_j * m)**2) for m in range(int(n_j) + 1, int(N_j) + 1))
        else:
            var_Tj = np.inf

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
        #print("N_i, N_j, n_i, n_j, burst_size", N_i, N_j, n_i, n_j, burst_size)
        # Ensure number of bursts is non-negative and valid
        num_bursts_i = max(0, int(np.ceil((N_i - n_i) / burst_size)))
        num_bursts_j = max(0, int(np.ceil((N_j - n_j) / burst_size)))

        if num_bursts_i == 0 or num_bursts_j == 0:
            return 0.0, 0.0

        # Compute moments with safe division
        mean_Ti = sum(1 / (k * max(1e-10, (N_i - m * burst_size))) for m in range(num_bursts_i))
        mean_Tj = sum(1 / (k * max(1e-10, (N_j - m * burst_size))) for m in range(num_bursts_j))
        var_Ti = sum(1 / (k * max(1e-10, (N_i - m * burst_size)))**2 for m in range(num_bursts_i))
        var_Tj = sum(1 / (k * max(1e-10, (N_j - m * burst_size)))**2 for m in range(num_bursts_j))

    elif mechanism == 'random_normal_burst':
        if mean_burst_size is None or mean_burst_size <= 0:
            return np.inf, np.inf

        num_bursts_i = max(0, (N_i - n_i) / mean_burst_size) - 1
        num_bursts_j = max(0, (N_j - n_j) / mean_burst_size) - 1

        if num_bursts_i < 1 or num_bursts_j < 1:
            return 0.0, 0.0

        mean_Ti = sum(1 / (k * max(1e-10, (N_i - m * mean_burst_size))) for m in range(int(np.floor(num_bursts_i))))
        mean_Tj = sum(1 / (k * max(1e-10, (N_j - m * mean_burst_size))) for m in range(int(np.floor(num_bursts_j))))
        var_Ti = sum(1 / (k * max(1e-10, (N_i - m * mean_burst_size)))**2 for m in range(int(np.floor(num_bursts_i))))
        var_Tj = sum(1 / (k * max(1e-10, (N_j - m * mean_burst_size)))**2 for m in range(int(np.floor(num_bursts_j))))

    else:
        raise ValueError("Mechanism must be 'simple', 'fixed_burst', or 'random_normal_burst'.")

    mean_X = mean_Ti - mean_Tj
    var_X = var_Ti + var_Tj
    return mean_X, var_X

def compute_pdf_mom(mechanism, x_grid, n_i, N_i, n_j, N_j, k, burst_size=None, mean_burst_size=None, var_burst_size=None):
    mean_X, var_X = compute_moments_mom(mechanism, n_i, N_i, n_j, N_j, k, burst_size, mean_burst_size, var_burst_size)
    if np.isinf(mean_X) or np.isinf(var_X):
        return np.full_like(x_grid, 1e-10)  # Small positive value to avoid log(0)
    if mechanism == 'simple':
        return norm.pdf(x_grid, loc=mean_X, scale=np.sqrt(var_X))
    elif mechanism == 'fixed_burst':
        return norm.pdf(x_grid, loc=mean_X, scale=np.sqrt(var_X))
    elif mechanism == 'random_normal_burst':
        return norm.pdf(x_grid, loc=mean_X, scale=np.sqrt(var_X))
    else:
        raise ValueError("Mechanism must be 'simple', 'fixed_burst', or 'random_normal_burst'.")