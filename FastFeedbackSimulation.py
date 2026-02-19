#!/usr/bin/env python3
import numpy as np
from typing import Tuple, Optional

def simulate_time_varying_combined_single(N: float, n: float, k_1: float, k_max: float, 
                                          burst_size: float, n_inner: float) -> float:
    """
    Simulate 'time_varying_k_combined' for a single chromosome.
    
    Combines:
    1. Time-varying rates (Chained Inverse method)
    2. Feedback rates (W(i))
    3. Fixed Bursts (jumps of size > 1)
    
    Path is deterministic: N -> N-b -> N-2b ...
    Waiting time at each step depends on current state via:
    Rate = k(t) * state * W(state)
    """
    # Round inputs
    N = int(round(N))
    n = int(round(n))
    
    current_time = 0.0
    tau = k_max / k_1
    
    # Helper for H_k(t)
    def get_H(t):
        if t <= tau:
            return 0.5 * k_1 * t**2
        else:
            return 0.5 * k_1 * tau**2 + k_max * (t - tau)
            
    current_H = 0.0
    
    # States iteration handles the bursts
    states = np.arange(int(N), int(n), -int(burst_size))
    
    if len(states) == 0:
        return 0.0
        
    for state in states:
        # Effective rate multiplier
        w = (state / n_inner) ** (-1/3) if state > n_inner else 1.0
        eff_rate = state * w
        
        # Random cost
        cost = np.random.exponential(1.0)
        
        # Delta Hazard
        delta_H = cost / eff_rate
        target_H = current_H + delta_H
        
        # Invert H_k(t)
        H_critical = 0.5 * (k_max**2) / k_1
        
        if target_H <= H_critical:
            new_time = np.sqrt(2 * target_H / k_1)
        else:
            new_time = tau + (target_H - H_critical) / k_max
            
        current_time = new_time
        current_H = target_H
        
    return current_time

def simulate_batch_feedback(mechanism: str, initial_states: np.ndarray, n0_lists: np.ndarray,
                          n_inner: float = None,
                          k_1: float = None, k_max: float = None,
                          burst_size: float = None,
                          num_simulations: int = 500) -> np.ndarray:
    """
    Batch simulation for feedback (steric hindrance) mechanisms.
    
    Supports: 'time_varying_k_steric_hindrance', 'time_varying_k_combined'
    
    Args:
        mechanism: 'time_varying_k_steric_hindrance' or 'time_varying_k_combined'
        initial_states: Array of shape (3,) for [N1, N2, N3]
        n0_lists: Array of shape (3,) for [n1, n2, n3]
        n_inner: Inner threshold for steric hindrance weight
        k_1: Initial rate slope
        k_max: Maximum rate
        burst_size: Burst size (for combined mechanism)
        num_simulations: Number of Monte Carlo samples
        
    Returns:
        np.ndarray: Array of shape (num_simulations, 3) with segregation times
    """
    results = np.zeros((num_simulations, 3))
    
    # Round inputs
    initial_states = np.round(initial_states).astype(int)
    n0_lists = np.round(n0_lists).astype(int)
    
    for i in range(3):
        N = initial_states[i]
        n = n0_lists[i]
        
        if mechanism == 'time_varying_k_steric_hindrance':
            states = np.arange(int(N), int(n), -1)
            if len(states) == 0:
                continue
            
            current_times = np.zeros(num_simulations)
            current_H = np.zeros(num_simulations)
            tau = k_max / k_1
            H_critical = 0.5 * (k_max**2) / k_1
            
            for state in states:
                w = (state / n_inner) ** (-1/3) if state > n_inner else 1.0
                eff_rate = state * w
                cost = np.random.exponential(1.0, size=num_simulations)
                delta_H = cost / eff_rate
                target_H = current_H + delta_H
                
                linear_mask = target_H <= H_critical
                constant_mask = ~linear_mask
                
                if np.any(linear_mask):
                    current_times[linear_mask] = np.sqrt(2 * target_H[linear_mask] / k_1)
                if np.any(constant_mask):
                    current_times[constant_mask] = tau + (target_H[constant_mask] - H_critical) / k_max
                
                current_H = target_H
                
            results[:, i] = current_times

        elif mechanism == 'time_varying_k_combined':
            # Similar to time_varying_k_steric_hindrance but with burst steps
            step = int(burst_size) if burst_size else 1
            states = np.arange(int(N), int(n), -step)
            
            if len(states) == 0:
                continue
                
            current_times = np.zeros(num_simulations)
            current_H = np.zeros(num_simulations)
            tau = k_max / k_1
            H_critical = 0.5 * (k_max**2) / k_1
            
            for state in states:
                w = (state / n_inner) ** (-1/3) if state > n_inner else 1.0
                eff_rate = state * w
                cost = np.random.exponential(1.0, size=num_simulations)
                delta_H = cost / eff_rate
                target_H = current_H + delta_H
                
                linear_mask = target_H <= H_critical
                constant_mask = ~linear_mask
                
                if np.any(linear_mask):
                    current_times[linear_mask] = np.sqrt(2 * target_H[linear_mask] / k_1)
                if np.any(constant_mask):
                    current_times[constant_mask] = tau + (target_H[constant_mask] - H_critical) / k_max
                
                current_H = target_H
                
            results[:, i] = current_times
            
    return results
