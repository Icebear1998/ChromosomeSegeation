import numpy as np
import matplotlib.pyplot as plt
import math
from math import exp, isfinite
from scipy.integrate import quad, IntegrationWarning
from scipy.special import gamma, gammaln
import warnings

N = 100


def f_tau_analytic(n, t, k):
    return k * math.factorial(N) / (math.factorial(n) * math.factorial(N-n-1))\
        * (np.exp(-(n+1)*k*t))*(1-np.exp(-k*t))**(N-n-1)


def f_tau_gamma(t, k, n, N):
    if (k <= 0) or (n < 0) or (N <= n):
        print("Invalid input, k = ", k, " n = ", n, " N = ", N)
        return 0.0
    try:
        log_comb_factor = gammaln(
            N + 1.0) - (gammaln(n + 1.0) + gammaln(N - n))
        comb_factor = np.exp(log_comb_factor)
    except Exception as e:
        print("Error in computing comb_factor: ", e)
        print("Invalid input, t = ", t," k = ", k, " n = ", n, " N = ", N)
        return 0.0

    # Compute the base; clamp it to avoid raising zero to a positive exponent.
    base = 1.0 - np.exp(-k * t)
    base = max(base, 1e-15)  # Avoid zero or negative base
    exponent = (N - n - 1.0)
    if exponent < 0:
        return 0.0

    val = k * comb_factor * np.exp(-(n + 1.0) * k * t) * (base ** exponent)
    if (not isfinite(val)) or (val < 0):
        print("Invalid value: ", val)
        print("Invalid input, t = ", t," k = ", k, " n = ", n, " N = ", N)
        return 0.0
    return val

###############################################################################
# 2) Difference PDF: f_diff_gamma(x; k1,n1,N1, k2,n2,N2)
###############################################################################


def f_diff_gamma(x, k, n1, N1, n2, N2):
    lower_bound = max(0.0, x)

    def integrand(t):
        return f_tau_gamma(t, k, n1, N1) * f_tau_gamma(t - x, k, n2, N2)

    try:
        val, _ = quad(integrand, lower_bound, np.inf,
                      limit=300, epsabs=1e-8, epsrel=1e-8)
        if not np.isfinite(val) or val < 0:
            print("Invalid value in f_diff_gamma: ", val)
            print("Parameters: k =", k, "n1 =", n1, "N1 =", N1, "n2 =", n2, "N2 =", N2, "x =", x)
            return 0.0
        return val
    except (ValueError, OverflowError, IntegrationWarning) as e:
        print("Error in computing f_diff_gamma:", e)
        print("Parameters: k =", k, "n1 =", n1, "N1 =", N1, "n2 =", n2, "N2 =", N2, "x =", x)
        return 0.0


def f_diff_analytic(x, k, n1, N1, n2, N2):
    """
    Compute the difference PDF analytically using f_tau_analytic.
    """
    lower_bound = max(0.0, x)

    def integrand(t):
        return f_tau_analytic(n1, t, k) * f_tau_analytic(n2, t - x, k)

    try:
        val, _ = quad(integrand, lower_bound, np.inf,
                      limit=300, epsabs=1e-8, epsrel=1e-8)
        if not np.isfinite(val) or val < 0:
            print("Invalid value in f_diff_analytic: ", val)
            print("Invalid input, k = ", k, " n1 = ", n1, " N1 = ", N1,
                  " n2 = ", n2, " N2 = ", N2)
            return 0.0
        return val
    except (ValueError, OverflowError, IntegrationWarning):
        print("Error in computing f_diff_analytic")
        print("Invalid input, k = ", k, " n1 = ", n1, " N1 = ", N1,
              " n2 = ", n2, " N2 = ", N2)
        return 0.0


def plot_chromosomes():
    t = np.linspace(0, 60, 100)
    k_values = [0.1, 0.2, 0.3, 0.4]  # Choose 4 values of k

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()  # Flatten the 2D array of axes for easy iteration
    for i, k in enumerate(k_values):
        for n in range(3, 10):
            axs[i].plot(t, f_tau_analytic(n, t, k), label=f'n={n}')
        axs[i].legend()
        axs[i].set_title(f'Plot for k={k}')

    plt.tight_layout()
    plt.show()


def plot_compare_f_diff():
    """
    Plot and compare f_diff_gamma and f_diff_analytic.
    """
    x_values = np.linspace(-20, 20, 100)  # Range of x values
    k = 0.1  # Example degradation rate
    n1, N1 = 5, 100  # Parameters for Chromosome 1
    n2, N2 = 4, 100  # Parameters for Chromosome 2

    # Compute f_diff_gamma and f_diff_analytic
    f_diff_gamma_values = [f_diff_gamma(
        x, k, n1, N1, n2, N2) for x in x_values]
    f_diff_analytic_values = [f_diff_analytic(
        x, k, n1, N1, n2, N2) for x in x_values]

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, f_diff_gamma_values,
             label="f_diff_gamma", linestyle='-', color='blue')
    plt.plot(x_values, f_diff_analytic_values,
             label="f_diff_analytic", linestyle='--', color='red')
    plt.xlabel("x")
    plt.ylabel("f_diff")
    plt.title("Comparison of f_diff_gamma and f_diff_analytic")
    plt.legend()
    plt.grid()
    plt.show()

def generate_threshold_values(n0_mean, n0_total, num_simulations):
    # n01_list = np.random.normal(loc=n0_mean[0], scale=1, size=num_simulations)
    # n02_list = np.random.normal(loc=n0_mean[1], scale=1, size=num_simulations)
    # n03_list = n0_total - n01_list - n02_list


    # n01_list = np.floor(np.clip(n01_list, 0.01, n0_total))
    # n02_list = np.floor(np.clip(n02_list, 0.01, n0_total))
    # n03_list = np.floor(np.clip(n03_list, 0.01, n0_total))

    n01_list = n0_mean[0] * np.ones(num_simulations)
    n02_list = n0_mean[1] * np.ones(num_simulations)
    n03_list = n0_total - n01_list - n02_list
    n0_list = np.column_stack((n01_list, n02_list, n03_list))
    return n0_list


###############################################################################
# Bootstrapping Functions for Handling Unequal Data Points
###############################################################################

class BootstrappingFitnessCalculator:
    """
    A fitness calculator that uses bootstrapping to handle unequal data points between strains.
    This ensures fair comparison by resampling datasets to have equal numbers of data points.
    """
    
    def __init__(self, target_sample_size=50, num_bootstrap_samples=100, random_seed=None):
        """
        Initialize the bootstrapping fitness calculator.
        
        Args:
            target_sample_size (int): Number of data points to resample each dataset to
            num_bootstrap_samples (int): Number of bootstrap samples to average over
            random_seed (int, optional): Random seed for reproducibility
        """
        self.target_sample_size = target_sample_size
        self.num_bootstrap_samples = num_bootstrap_samples
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def bootstrap_resample(self, data, sample_size=None):
        """
        Resample data with replacement to create a bootstrap sample.
        
        Args:
            data (array-like): Original data array
            sample_size (int, optional): Size of the bootstrap sample. 
                                       If None, uses self.target_sample_size
            
        Returns:
            np.ndarray: Bootstrap sample
        """
        data = np.asarray(data)
        if len(data) == 0:
            return np.array([])
        
        if sample_size is None:
            sample_size = self.target_sample_size
        
        # If original data is smaller than target, sample with replacement
        if len(data) < sample_size:
            indices = np.random.choice(len(data), size=sample_size, replace=True)
        else:
            # If original data is larger, sample without replacement
            indices = np.random.choice(len(data), size=sample_size, replace=False)
        
        return data[indices]
    
    def calculate_bootstrap_likelihood(self, experimental_data, simulated_data):
        """
        Calculate likelihood using bootstrapped experimental data.
        
        Args:
            experimental_data (array-like): Original experimental data
            simulated_data (array-like): Simulated data from model
            
        Returns:
            float: Average negative log-likelihood across bootstrap samples
        """
        try:
            from scipy.stats import gaussian_kde
            
            experimental_data = np.asarray(experimental_data)
            simulated_data = np.asarray(simulated_data)
            
            if len(simulated_data) < 10:
                return 1e6  # Penalty for insufficient simulated data
            
            if len(experimental_data) == 0:
                return 1e6  # Penalty for no experimental data
            
            # Create KDE from simulated data
            kde = gaussian_kde(simulated_data)
            
            bootstrap_nlls = []
            
            for _ in range(self.num_bootstrap_samples):
                # Resample experimental data
                bootstrap_data = self.bootstrap_resample(experimental_data)
                
                if len(bootstrap_data) == 0:
                    continue
                
                # Calculate likelihood for bootstrapped experimental data
                log_likelihoods = kde.logpdf(bootstrap_data)
                
                # Handle numerical issues
                log_likelihoods = np.clip(log_likelihoods, -50, 50)
                
                # Calculate negative log-likelihood (NORMALIZATION REMOVED)
                nll = -np.sum(log_likelihoods)  # REMOVED: / len(bootstrap_data)
                bootstrap_nlls.append(nll)
            
            if not bootstrap_nlls:
                return 1e6
            
            # Return average negative log-likelihood across bootstrap samples
            return np.mean(bootstrap_nlls)
            
        except Exception as e:
            warnings.warn(f"Bootstrap likelihood calculation error: {e}")
            return 1e6
    
    def calculate_weighted_likelihood(self, experimental_data, simulated_data):
        """
        Alternative approach: Calculate likelihood with sample size weighting.
        
        Args:
            experimental_data (array-like): Original experimental data
            simulated_data (array-like): Simulated data from model
            
        Returns:
            float: Weighted negative log-likelihood
        """
        try:
            from scipy.stats import gaussian_kde
            
            experimental_data = np.asarray(experimental_data)
            simulated_data = np.asarray(simulated_data)
            
            if len(simulated_data) < 10:
                return 1e6
            
            if len(experimental_data) == 0:
                return 1e6
            
            # Create KDE from simulated data
            kde = gaussian_kde(simulated_data)
            
            # Calculate likelihood for experimental data
            log_likelihoods = kde.logpdf(experimental_data)
            
            # Handle numerical issues
            log_likelihoods = np.clip(log_likelihoods, -50, 50)
            
            # Calculate negative log-likelihood (NORMALIZATION REMOVED)
            # sample_size = len(experimental_data)  # REMOVED
            
            return -np.sum(log_likelihoods)  # REMOVED: / sample_size
            
        except Exception as e:
            warnings.warn(f"Weighted likelihood calculation error: {e}")
            return 1e6


def calculate_bootstrap_likelihood(experimental_data, simulated_data, 
                                 target_sample_size=50, num_bootstrap_samples=100,
                                 random_seed=None):
    """
    Convenience function to calculate bootstrap likelihood without creating a class instance.
    
    Args:
        experimental_data (array-like): Original experimental data
        simulated_data (array-like): Simulated data from model
        target_sample_size (int): Number of data points to resample each dataset to
        num_bootstrap_samples (int): Number of bootstrap samples to average over
        random_seed (int, optional): Random seed for reproducibility
    
    Returns:
        float: Average negative log-likelihood across bootstrap samples
    """
    calculator = BootstrappingFitnessCalculator(
        target_sample_size=target_sample_size,
        num_bootstrap_samples=num_bootstrap_samples,
        random_seed=random_seed
    )
    return calculator.calculate_bootstrap_likelihood(experimental_data, simulated_data)


def calculate_weighted_likelihood(experimental_data, simulated_data):
    """
    Convenience function to calculate weighted likelihood without creating a class instance.
    
    Args:
        experimental_data (array-like): Original experimental data
        simulated_data (array-like): Simulated data from model
    
    Returns:
        float: Weighted negative log-likelihood
    """
    calculator = BootstrappingFitnessCalculator()
    return calculator.calculate_weighted_likelihood(experimental_data, simulated_data)


def analyze_dataset_sizes(datasets):
    """
    Analyze the sizes of datasets to help determine appropriate bootstrap parameters.
    
    Args:
        datasets (dict): Dictionary of datasets with experimental data
        
    Returns:
        dict: Analysis results including recommended target sample size
    """
    sizes = {}
    all_sizes = []
    
    print("Dataset Size Analysis:")
    print("-" * 40)
    
    for dataset_name, data_dict in datasets.items():
        if isinstance(data_dict, dict):
            size_12 = len(data_dict.get('delta_t12', []))
            size_32 = len(data_dict.get('delta_t32', []))
            sizes[dataset_name] = {'delta_t12': size_12, 'delta_t32': size_32}
            all_sizes.extend([size_12, size_32])
            print(f"{dataset_name}: {size_12} (T1-T2), {size_32} (T3-T2)")
        else:
            # Handle simple array case
            size = len(data_dict) if hasattr(data_dict, '__len__') else 0
            sizes[dataset_name] = size
            all_sizes.append(size)
            print(f"{dataset_name}: {size} points")
    
    if all_sizes:
        min_size = min(all_sizes)
        max_size = max(all_sizes)
        median_size = np.median(all_sizes)
        mean_size = np.mean(all_sizes)
        
        print(f"\nSize Statistics:")
        print(f"  Min: {min_size}")
        print(f"  Max: {max_size}")
        print(f"  Median: {median_size:.1f}")
        print(f"  Mean: {mean_size:.1f}")
        
        # Recommend target sample size as median or slightly below
        recommended_size = max(int(median_size * 0.8), min_size)
        
        print(f"\nRecommended target_sample_size: {recommended_size}")
        print(f"(80% of median size, but not less than minimum)")
        
        return {
            'sizes': sizes,
            'min_size': min_size,
            'max_size': max_size,
            'median_size': median_size,
            'mean_size': mean_size,
            'recommended_target_size': recommended_size
        }
    else:
        print("No data found!")
        return {'sizes': sizes, 'recommended_target_size': 50}


# if __name__ == "__main__":
#     plot_compare_f_diff()
