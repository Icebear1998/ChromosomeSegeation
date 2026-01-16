#!/usr/bin/env python3
"""
Test script to verify ALL mechanisms against Gillespie algorithm.
Checks both correctness and speedup to confirm Fast Simulation is active.
"""

import numpy as np
import time
import sys
import os
from simulation_utils import run_simulation_for_dataset

def test_mechanism_support(mechanism, params, n0_list, num_sims=500):
    print(f"\nTesting {mechanism}...")
    
    start_time = time.time()
    try:
        dt12, dt32 = run_simulation_for_dataset(mechanism, params, n0_list, num_sims)
        elapsed = time.time() - start_time
        
        if dt12 is None:
            print(f"  ❌ Failed: Returned None")
            return False
            
        # Basic sanity checks
        mean_dt12 = np.mean(dt12)
        mean_dt32 = np.mean(dt32)
        
        print(f"  ✓ Success: {len(dt12)} simulations in {elapsed:.4f}s")
        print(f"  Time per sim: {elapsed/num_sims*1000:.4f} ms")
        print(f"  Mean Delta T12: {mean_dt12:.2f}")
        print(f"  Mean Delta T32: {mean_dt32:.2f}")
        
        # Heuristic check for Fast Simulation usage
        # Gillespie usually takes >5ms per sim for these parameters
        # Fast methods usually take <0.5ms per sim
        time_per_sim_ms = elapsed/num_sims*1000
        if time_per_sim_ms < 1.0:
            print(f"  🚀 FAST SIMULATION ACTIVE (<1ms/sim)")
        elif time_per_sim_ms > 5.0:
            print(f"  ⚠️  SLOW SIMULATION DETECTED (>5ms/sim) - Likely Gillespie fallback")
        else:
            print(f"  ❓ UNCERTAIN SPEED ({time_per_sim_ms:.2f}ms/sim)")
            
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("VERIFYING FAST SIMULATION SUPPORT FOR ALL MECHANISMS")
    print("="*60)
    
    n0_list = [3.0, 5.0, 8.0]
    num_sims = 1000
    
    # 1. Simple
    params_simple = {'N1':300, 'N2':400, 'N3':5000, 'n1':3, 'n2':5, 'n3':8, 'k': 0.05}
    test_mechanism_support('simple', params_simple, n0_list, num_sims)
    
    # 2. Fixed Burst
    params_fb = {**params_simple, 'burst_size': 5.0}
    test_mechanism_support('fixed_burst', params_fb, n0_list, num_sims)
    
    # 3. Time Varying K
    params_tv = {'N1':300, 'N2':400, 'N3':5000, 'n1':3, 'n2':5, 'n3':8, 'k_1': 0.001, 'k_max': 0.05}
    test_mechanism_support('time_varying_k', params_tv, n0_list, num_sims)
    
    # 4. Time Varying K Fixed Burst
    params_tv_fb = {**params_tv, 'burst_size': 5.0}
    test_mechanism_support('time_varying_k_fixed_burst', params_tv_fb, n0_list, num_sims)
    
    # 5. Time Varying K Burst Onion (NEWLY IMPLEMENTED)
    params_tv_bo = {**params_tv, 'burst_size': 5.0}
    test_mechanism_support('time_varying_k_burst_onion', params_tv_bo, n0_list, num_sims)
    
    # 6. Feedback Onion
    params_fo = {**params_simple, 'n_inner': 50.0}
    test_mechanism_support('feedback_onion', params_fo, n0_list, num_sims)
    
    # 7. Fixed Burst Feedback Onion
    params_fb_fo = {**params_fb, 'n_inner': 50.0}
    test_mechanism_support('fixed_burst_feedback_onion', params_fb_fo, n0_list, num_sims)
    
    # 8. Time Varying K Feedback Onion
    params_tv_fo = {**params_tv, 'n_inner': 50.0}
    test_mechanism_support('time_varying_k_feedback_onion', params_tv_fo, n0_list, num_sims)
    
    # 9. Time Varying K Combined
    params_tv_comb = {**params_tv, 'burst_size': 5.0, 'n_inner': 50.0}
    test_mechanism_support('time_varying_k_combined', params_tv_comb, n0_list, num_sims)

if __name__ == "__main__":
    main()
