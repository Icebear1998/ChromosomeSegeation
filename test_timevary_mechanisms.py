#!/usr/bin/env python3
"""
Test script for the new MultiMechanismSimulationTimevary class.
Tests the three integrated time-varying mechanisms:
1. time_varying_k (pure time-varying)
2. time_varying_k_fixed_burst (time-varying + fixed burst)
3. time_varying_k_feedback_onion (time-varying + feedback onion)
"""

import numpy as np
from MultiMechanismSimulationTimevary import MultiMechanismSimulationTimevary


def test_mechanism(mechanism_name, rate_params, num_runs=100):
    """
    Test a specific mechanism with multiple simulation runs.
    
    Args:
        mechanism_name (str): Name of the mechanism to test
        rate_params (dict): Parameters for the mechanism
        num_runs (int): Number of simulation runs
    """
    print(f"\n=== Testing {mechanism_name} ===")
    print(f"Parameters: {rate_params}")
    
    # Common simulation parameters
    initial_state = [100, 150, 200]
    n0_list = [3, 5, 8]
    max_time = 1000
    
    separation_times = []
    
    for run in range(num_runs):
        try:
            # Create and run simulation
            sim = MultiMechanismSimulationTimevary(
                mechanism=mechanism_name,
                initial_state_list=initial_state,
                rate_params=rate_params,
                n0_list=n0_list,
                max_time=max_time
            )
            
            times, states, sep_times = sim.simulate()
            separation_times.append(sep_times)
            
        except Exception as e:
            print(f"Error in run {run}: {e}")
            return
    
    # Calculate statistics
    if separation_times:
        sep_times_array = np.array(separation_times)
        T1_mean = np.mean(sep_times_array[:, 0])
        T2_mean = np.mean(sep_times_array[:, 1])
        T3_mean = np.mean(sep_times_array[:, 2])
        
        delta_t12 = sep_times_array[:, 0] - sep_times_array[:, 1]
        delta_t32 = sep_times_array[:, 2] - sep_times_array[:, 1]
        
        print(f"Results from {num_runs} runs:")
        print(f"  Mean separation times: T1={T1_mean:.2f}, T2={T2_mean:.2f}, T3={T3_mean:.2f}")
        print(f"  T1-T2: Mean={np.mean(delta_t12):.3f}, Std={np.std(delta_t12):.3f}")
        print(f"  T3-T2: Mean={np.mean(delta_t32):.3f}, Std={np.std(delta_t32):.3f}")
        print(f"  ✅ Test completed successfully!")
    else:
        print("  ❌ No successful runs!")


def main():
    """
    Run tests for all three time-varying mechanisms.
    """
    print("Testing MultiMechanismSimulationTimevary class")
    print("=" * 50)
    
    # Test 1: Pure time-varying k
    test_mechanism(
        mechanism_name='time_varying_k',
        rate_params={'k_1': 0.001, 'k_max': 0.05}
    )
    
    # Test 2: Time-varying k with fixed burst
    test_mechanism(
        mechanism_name='time_varying_k_fixed_burst',
        rate_params={'k_1': 0.001, 'k_max': 0.05, 'burst_size': 8}
    )
    
    # Test 3: Time-varying k with feedback onion
    test_mechanism(
        mechanism_name='time_varying_k_feedback_onion',
        rate_params={'k_1': 0.001, 'k_max': 0.05, 'n_inner': 50}
    )
    
    print(f"\n{'=' * 50}")
    print("All tests completed!")


if __name__ == "__main__":
    main() 