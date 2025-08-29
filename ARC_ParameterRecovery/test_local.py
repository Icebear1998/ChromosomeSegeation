#!/usr/bin/env python3
"""
Quick local test for ARC Parameter Recovery Study

This script tests the main components locally with minimal settings
before deploying to ARC. Run this first to ensure everything works.
"""

import sys
import os
import time
from datetime import datetime

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        import numpy as np
        import pandas as pd
        import scipy
        from multiprocessing import Pool, cpu_count
        print(f"✓ NumPy: {np.__version__}")
        print(f"✓ Pandas: {pd.__version__}")
        print(f"✓ SciPy: {scipy.__version__}")
        print(f"✓ CPU cores available: {cpu_count()}")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_file_structure():
    """Test that all required files are present."""
    print("\nTesting file structure...")
    required_files = [
        "run_recovery_study.py",
        "SimulationOptimization_join.py",
        "simulation_utils.py",
        "MultiMechanismSimulationTimevary.py",
        "Chromosomes_Theory.py",
        "ground_truth_parameters_rounded.txt",
        "Data/All_strains_SCStimes.xlsx"
    ]
    
    all_found = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - NOT FOUND")
            all_found = False
    
    return all_found

def test_data_loading():
    """Test that data can be loaded successfully."""
    print("\nTesting data loading...")
    try:
        from simulation_utils import load_experimental_data
        datasets = load_experimental_data()
        print(f"✓ Loaded {len(datasets)} datasets:")
        for name, data in datasets.items():
            print(f"  {name}: {len(data['delta_t12'])} T1-T2, {len(data['delta_t32'])} T3-T2 points")
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False

def test_ground_truth_loading():
    """Test that ground truth parameters can be loaded."""
    print("\nTesting ground truth parameter loading...")
    try:
        ground_truth = {}
        with open('ground_truth_parameters_rounded.txt', 'r') as f:
            lines = f.readlines()
            
        in_params = False
        for line in lines:
            line = line.strip()
            if line == "Optimized Parameters (ratio-based):":
                in_params = True
                continue
            elif line.startswith("Derived Parameters:"):
                break
            elif in_params and '=' in line:
                param, value = line.split(' = ')
                ground_truth[param.strip()] = float(value.strip())
        
        print(f"✓ Loaded {len(ground_truth)} ground truth parameters:")
        for i, (param, value) in enumerate(ground_truth.items()):
            if i < 5:  # Show first 5
                print(f"  {param}: {value}")
            elif i == 5:
                print(f"  ... and {len(ground_truth)-5} more")
                break
        
        return True
    except Exception as e:
        print(f"✗ Ground truth loading failed: {e}")
        return False

def test_minimal_recovery():
    """Test a minimal recovery run."""
    print("\nTesting minimal recovery run...")
    print("This will take 1-2 minutes...")
    
    try:
        # Import the main script
        sys.path.insert(0, '.')
        from run_recovery_study import get_ground_truth_params, generate_synthetic_data
        from simulation_utils import load_experimental_data
        
        # Load data
        datasets = load_experimental_data()
        
        # Test ground truth parameter loading (skip actual optimization)
        print("  Skipping ground truth optimization (would take too long)")
        
        # Load ground truth from file instead
        ground_truth_params = {}
        with open('ground_truth_parameters_rounded.txt', 'r') as f:
            lines = f.readlines()
            
        in_params = False
        for line in lines:
            line = line.strip()
            if line == "Optimized Parameters (ratio-based):":
                in_params = True
                continue
            elif line.startswith("Derived Parameters:"):
                break
            elif in_params and '=' in line:
                param, value = line.split(' = ')
                ground_truth_params[param.strip()] = float(value.strip())
        
        print(f"  ✓ Loaded ground truth parameters")
        
        # Test synthetic data generation with minimal size
        print("  Testing synthetic data generation...")
        synthetic_datasets = generate_synthetic_data(
            'time_varying_k_combined', 
            ground_truth_params, 
            num_simulations=20  # Very small for testing
        )
        
        print(f"  ✓ Generated synthetic data for {len(synthetic_datasets)} strains")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Minimal recovery test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("ARC Parameter Recovery Study - Local Test")
    print("=" * 60)
    print(f"Date: {datetime.now()}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    tests = [
        ("Import test", test_imports),
        ("File structure test", test_file_structure),
        ("Data loading test", test_data_loading),
        ("Ground truth loading test", test_ground_truth_loading),
        ("Minimal recovery test", test_minimal_recovery)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print()
    print(f"Tests passed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("🎉 ALL TESTS PASSED!")
        print("The system is ready for ARC deployment.")
        print()
        print("Next steps:")
        print("1. Copy this directory to your ARC system")
        print("2. Modify submit_recovery.slurm for your ARC configuration")
        print("3. Submit the job with: sbatch submit_recovery.slurm")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please fix the issues before deploying to ARC.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
