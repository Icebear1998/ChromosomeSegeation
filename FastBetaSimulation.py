#!/usr/bin/env python3
import numpy as np
from typing import Tuple, Optional
import warnings


def simulate_simple_beta_single(N: float, n: float, k: float) -> float:
    """
    Simulate segregation time for a single chromosome using Beta sampling.
    
    This is the O(1) alternative to the Gillespie loop for the 'simple' mechanism.
    
    Mathematical derivation:
    - N cohesins degrade independently with rate k each.
    - Each cohesin has lifetime T ~ Exp(k).
    - We want the time when only n remain, i.e., the (N-n)-th degradation.
    - The k-th smallest exponential r.v. has CDF that can be sampled via:
      1. Sample u ~ Beta(k, N-k+1) where k = N-n
      2. Transform: T = -ln(1-u)/k
    
    Args:
        N: Initial number of cohesins
        n: Threshold number (chromosome separates when count <= n)
        k: Degradation rate constant
        
    Returns:
        float: Time of chromosome separation
    """
    # Round inputs to nearest integer to avoid truncation issues
    N = int(round(N))
    n = int(round(n))
    
    # Number of degradation events needed
    num_events = N - n
    
    # Handle edge cases
    if num_events <= 0:
        return 0.0
    
    # Sample from Beta distribution (order statistic)
    # For the k-th smallest of N samples: Beta(k, N-k+1)
    alpha = num_events
    beta = N - num_events + 1
    
    u = np.random.beta(alpha, beta)
    
    # Inverse transform: Exponential quantile function
    # T = -ln(1-u)/k
    time = -np.log(1 - u) / k
    
    return time


def simulate_simple_beta_vectorized(N: np.ndarray, n: np.ndarray, k: np.ndarray, 
                                    num_simulations: int = 1) -> np.ndarray:
    """
    Vectorized Beta sampling for multiple chromosomes and/or multiple simulations.
    
    This function can handle:
    - Single chromosome, multiple simulations: N, n, k are scalars
    - Multiple chromosomes, single simulation: N, n, k are arrays of shape (3,)
    - Multiple chromosomes, multiple simulations: Broadcasting
    
    Args:
        N: Initial cohesin counts (scalar or array)
        n: Threshold counts (scalar or array)
        k: Rate constants (scalar or array)
        num_simulations: Number of independent simulations to run
        
    Returns:
        np.ndarray: Segregation times, shape depends on inputs
    """
    # Convert to arrays for consistent handling
    N = np.atleast_1d(N)
    n = np.atleast_1d(n)
    k = np.atleast_1d(k)
    
    # Round inputs to nearest integer
    N = np.round(N)
    n = np.round(n)
    
    # Calculate number of events needed
    num_events = N - n
    
    # Prepare output shape
    if num_simulations > 1:
        output_shape = (num_simulations,) + N.shape
    else:
        output_shape = N.shape
    
    # Initialize output
    times = np.zeros(output_shape)
    
    # Generate all Beta samples at once
    for idx in np.ndindex(N.shape):
        ne = num_events[idx]
        N_val = int(N[idx])
        k_val = k[idx]
        
        if ne <= 0:
            continue
            
        alpha = ne
        beta = N_val - ne + 1
        
        # Sample Beta for all simulations at once
        u = np.random.beta(alpha, beta, size=num_simulations)
        
        # Transform to times
        if num_simulations > 1:
            times[:, idx[0]] = -np.log(1 - u) / k_val
        else:
            times[idx] = -np.log(1 - u) / k_val
    
    return times.squeeze()


def simulate_fixed_burst_beta_single(N: float, n: float, k: float, burst_size: float) -> float:
    """
    Simulate fixed_burst mechanism using Beta sampling.
    
    Key insight: Fixed bursts of size q are equivalent to a simple model with:
    - N' = ceil(N / q) "super-cohesins"
    - n' = floor(n / q) threshold in super-cohesins
    - k' = k * q (rate per super-cohesin degradation event)
    
    Args:
        N: Initial number of cohesins
        n: Threshold number
        k: Base degradation rate
        burst_size: Number of cohesins degraded per event
        
    Returns:
        float: Time of chromosome separation
    """
    # Round inputs to nearest integer
    N = int(round(N))
    n = int(round(n))

    N_prime = int(np.ceil(N / burst_size))
    n_prime = int(N_prime - np.ceil((N - n) / burst_size))
    k_prime = k * burst_size
    
    # Use simple Beta sampling on transformed parameters
    return simulate_simple_beta_single(N_prime, n_prime, k_prime)


def simulate_time_varying_k_beta_single(N: float, n: float, k_1: float, k_max: float) -> float:
    """
    Simulate time_varying_k mechanism using Beta sampling.
    
    Args:
        N: Initial number of cohesins
        n: Threshold number
        k_1: Initial rate (slope)
        k_max: Maximum rate (cap)
        
    Returns:
        float: Time of chromosome separation
    """
    # Round inputs to nearest integer
    N = int(round(N))
    n = int(round(n))
    
    # Number of degradation events needed
    num_events = N - n
    
    # Handle edge cases
    if num_events <= 0:
        return 0.0
    
    # Sample from Beta distribution
    alpha = num_events
    beta = int(N) - num_events + 1
    u = np.random.beta(alpha, beta)
    
    # Convert to standard exponential variate
    y = -np.log(1 - u)
    
    # Invert the cumulative hazard H(t) = y
    # Critical value: y at transition time tau = k_max/k_1
    y_critical = k_max**2 / (2 * k_1)
    
    if y <= y_critical:
        # Linear phase: H(t) = k_1 * t²/2
        # => t = sqrt(2y / k_1)
        time = np.sqrt(2 * y / k_1)
    else:
        # Constant phase: H(t) = k_max²/(2k_1) + k_max*(t - k_max/k_1)
        # => t = k_max/k_1 + (y - k_max²/(2k_1)) / k_max
        tau = k_max / k_1
        time = tau + (y - y_critical) / k_max
    
    return time


def simulate_time_varying_k_fixed_burst_beta_single(N: float, n: float, k_1: float, 
                                                     k_max: float, burst_size: float) -> float:
    """
    Simulate time_varying_k_fixed_burst mechanism using Beta sampling.
    
    Args:
        N: Initial number of cohesins
        n: Threshold number
        k_1: Initial rate (slope)
        k_max: Maximum rate
        burst_size: Number of cohesins per burst
        
    Returns:
        float: Time of chromosome separation
    """
    N = int(round(N))
    n = int(round(n))
    
    N_prime = int(np.ceil(N / burst_size))
    n_prime = int(N_prime - np.ceil((N - n) / burst_size))
    k_1_prime = k_1 * burst_size
    k_max_prime = k_max * burst_size
    
    return simulate_time_varying_k_beta_single(N_prime, n_prime, k_1_prime, k_max_prime)



class FastBetaSimulator:
    """
    Class wrapper to match the interface of MultiMechanismSimulation.
    """
    
    def __init__(self, mechanism: str, initial_state_list: list, 
                 rate_params: dict, n0_list: list, max_time: float = 10000.0):
        """
        Initialize the fast Beta simulator.
        
        Args:
            mechanism: Must be 'simple' or 'fixed_burst'
            initial_state_list: [N1, N2, N3] initial cohesin counts
            rate_params: {'k': rate, 'burst_size': size} (burst_size optional)
            n0_list: [n1, n2, n3] thresholds
            max_time: Not used, kept for interface compatibility
        """
        if mechanism not in ['simple', 'fixed_burst']:
            raise ValueError(f"FastBetaSimulator only supports 'simple' and 'fixed_burst', got '{mechanism}'")
        
        self.mechanism = mechanism
        self.initial_state = np.round(np.array(initial_state_list, dtype=float))
        self.n0_list = np.round(np.array(n0_list, dtype=float))
        
        self.k = rate_params['k']
        self.burst_size = rate_params.get('burst_size', 1.0)
        self.max_time = max_time
        
    def simulate(self) -> Tuple[list, list, list]:
        """
        Run a single simulation and return results in Gillespie-compatible format.
        
        Returns:
            tuple: (times, states, separate_times)
                - times: [0.0, t1, t2, t3] (not used, kept for compatibility)
                - states: [[N1,N2,N3], [n1,n2,n3]] (simplified)
                - separate_times: [t1, t2, t3] segregation times
        """
        separate_times = []
        
        for i in range(3):
            N = self.initial_state[i]
            n = self.n0_list[i]
            
            if self.mechanism == 'simple':
                t = simulate_simple_beta_single(N, n, self.k)
            else:  # fixed_burst
                t = simulate_fixed_burst_beta_single(N, n, self.k, self.burst_size)
            
            separate_times.append(float(t))
        
        # Create minimal state trajectory for compatibility
        times = [0.0] + separate_times
        states = [self.initial_state.tolist(), self.n0_list.tolist()]
        
        return times, states, separate_times


def simulate_batch(mechanism: str, initial_states: np.ndarray, n0_lists: np.ndarray,
                   k: float = None, burst_size: float = None, k_1: float = None, 
                   k_max: float = None, num_simulations: int = 500) -> np.ndarray:
    """
    Ultra-fast batch simulation for optimization loops.
    
    This is the key function for integration into the optimization pipeline.
    Now supports: 'simple', 'fixed_burst', 'time_varying_k', 'time_varying_k_fixed_burst'
    
    Args:
        mechanism: Mechanism name
        initial_states: Array of shape (3,) for [N1, N2, N3]
        n0_lists: Array of shape (3,) for [n1, n2, n3]
        k: Degradation rate (for simple/fixed_burst)
        burst_size: Burst size (for fixed_burst mechanisms)
        k_1: Initial rate slope (for time_varying_k mechanisms)
        k_max: Maximum rate (for time_varying_k mechanisms)
        num_simulations: Number of Monte Carlo samples
        
    Returns:
        np.ndarray: Array of shape (num_simulations, 3) with segregation times
    """
    results = np.zeros((num_simulations, 3))
    
    initial_states = np.round(initial_states).astype(int)
    n0_lists = np.round(n0_lists).astype(int)
    
    for i in range(3):
        N = initial_states[i]
        n = n0_lists[i]
        
        if mechanism == 'simple':
            # Vectorized generation of all samples
            num_events = int(N - n)
            if num_events > 0:
                alpha = num_events
                beta = int(N) - num_events + 1
                u = np.random.beta(alpha, beta, size=num_simulations)
                results[:, i] = -np.log(1 - u) / k
                
        elif mechanism == 'fixed_burst':
            N_prime = int(np.ceil(N / burst_size))
            # Corrected formula for n_prime
            n_prime = int(N_prime - np.ceil((N - n) / burst_size))
            k_prime = k * burst_size
            num_events = N_prime - n_prime
            
            if num_events > 0:
                alpha = num_events
                beta = N_prime - num_events + 1
                u = np.random.beta(alpha, beta, size=num_simulations)
                results[:, i] = -np.log(1 - u) / k_prime
                
        elif mechanism == 'time_varying_k':
            N = round(N)
            n = round(n)
            num_events = int(N - n)
            if num_events > 0:
                alpha = num_events
                beta = int(N) - num_events + 1
                u = np.random.beta(alpha, beta, size=num_simulations)
                
                # Convert to exponential variate
                y = -np.log(1 - u)
                
                # Invert cumulative hazard
                y_critical = k_max**2 / (2 * k_1)
                tau = k_max / k_1
                
                # Vectorized piecewise inversion
                linear_phase = y <= y_critical
                results[linear_phase, i] = np.sqrt(2 * y[linear_phase] / k_1)
                results[~linear_phase, i] = tau + (y[~linear_phase] - y_critical) / k_max
                
        elif mechanism == 'time_varying_k_fixed_burst':
            N_prime = int(np.ceil(N / burst_size))
            # Corrected formula for n_prime
            n_prime = int(N_prime - np.ceil((N - n) / burst_size))
            k_1_prime = k_1 * burst_size
            k_max_prime = k_max * burst_size
            num_events = N_prime - n_prime
            
            if num_events > 0:
                alpha = num_events
                beta = N_prime - num_events + 1
                u = np.random.beta(alpha, beta, size=num_simulations)
                
                # Convert to exponential variate
                y = -np.log(1 - u)
                
                # Invert cumulative hazard with transformed parameters
                y_critical = k_max_prime**2 / (2 * k_1_prime)
                tau = k_max_prime / k_1_prime
                
                # Vectorized piecewise inversion
                linear_phase = y <= y_critical
                results[linear_phase, i] = np.sqrt(2 * y[linear_phase] / k_1_prime)
                results[~linear_phase, i] = tau + (y[~linear_phase] - y_critical) / k_max_prime
    
    return results



if __name__ == "__main__":
    # Quick demonstration
    print("FastBetaSimulation - O(1) Chromosome Segregation Simulator")
    print("="*60)
    
    # Test parameters
    N = 300.0
    n = 10.0
    k = 0.05
    
    print(f"\nTest: N={N}, n={n}, k={k}")
    print(f"Expected mean time ≈ {-np.log(1 - (N-n)/N) / k:.2f}")
    
    # Single simulation
    t = simulate_simple_beta_single(N, n, k)
    print(f"Single simulation result: {t:.2f}")
    
    # Batch simulation
    times = simulate_simple_beta_vectorized(
        np.array([N]), np.array([n]), np.array([k]), num_simulations=10000
    )
    print(f"\nBatch statistics (10,000 sims):")
    print(f"  Mean: {np.mean(times):.2f}")
    print(f"  Std:  {np.std(times):.2f}")
    
    print("\n✓ FastBetaSimulation ready for integration!")
