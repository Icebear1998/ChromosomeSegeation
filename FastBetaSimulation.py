#!/usr/bin/env python3
import numpy as np
from typing import Tuple, Optional
import warnings


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


def simulate_batch(mechanism: str, initial_states: np.ndarray, n0_lists: np.ndarray,
                   k_1: float = None, k_max: float = None, 
                   burst_size: float = None, num_simulations: int = 500) -> np.ndarray:
    """
    Ultra-fast batch simulation for optimization loops.
    
    Supports: 'time_varying_k', 'time_varying_k_fixed_burst'
    
    Args:
        mechanism: Mechanism name ('time_varying_k' or 'time_varying_k_fixed_burst')
        initial_states: Array of shape (3,) for [N1, N2, N3]
        n0_lists: Array of shape (3,) for [n1, n2, n3]
        k_1: Initial rate slope (for time_varying_k mechanisms)
        k_max: Maximum rate (for time_varying_k mechanisms)
        burst_size: Burst size (for fixed_burst mechanisms)
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
        
        if mechanism == 'time_varying_k':
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
