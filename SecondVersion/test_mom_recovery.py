#!/usr/bin/env python3
"""
Quick test for MoM Parameter Recovery Study

This script tests the MoM parameter recovery system with minimal settings
to ensure everything works correctly.
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
        from MoMOptimization_join import joint_objective, get_mechanism_info, unpack_parameters
        from MoMCalculations import compute_pdf_for_mechanism
        print(f"✓ NumPy: {np.__version__}")
        print(f"✓ Pandas: {pd.__version__}")
        print(f"✓ SciPy: {scipy.__version__}")
        print(f"✓ CPU cores available: {cpu_count()}")
        print("✓ MoM optimization functions imported")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_file_structure():
    """Test that all required files are present."""
    print("\nTesting file structure...")
    required_files = [
        "parameter_recovery_mom.py",
        "analyze_parameter_recovery_mom.py",
        "MoMOptimization_join.py",
        "MoMCalculations.py",
        "Chromosomes_Theory.py",
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

def test_parameter_file():
    """Test that we can find and load a parameter file."""
    print("\nTesting parameter file loading...")
    try:
        # Look for parameter files
        mechanism = 'fixed_burst_feedback_onion'
        pattern = f"optimized_parameters_{mechanism}"
        files = [f for f in os.listdir('.') if f.startswith(pattern) and f.endswith('.txt')]
        
        if not files:
            print(f"✗ No parameter files found for {mechanism}")
            print("   You may need to run MoMOptimization_join.py first")
            return False
        
        filename = files[0]
        print(f"✓ Found parameter file: {filename}")
        
        # Try to parse it
        ground_truth = {}
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if ':' in line and not line.startswith('#'):
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if '_nll' in key or key in ['total_nll', 'wt_nll']:
                    continue
                if value == 'not_fitted' or '#' in value:
                    continue
                
                try:
                    ground_truth[key] = float(value.split()[0])
                except (ValueError, IndexError):
                    continue
        
        print(f"✓ Parsed {len(ground_truth)} parameters")
        if len(ground_truth) >= 5:
            print("  Sample parameters:")
            for i, (param, value) in enumerate(ground_truth.items()):
                if i < 3:
                    print(f"    {param}: {value:.4f}")
                elif i == 3:
                    print(f"    ... and {len(ground_truth)-3} more")
                    break
        
        return True
        
    except Exception as e:
        print(f"✗ Parameter file test failed: {e}")
        return False

def test_minimal_recovery():
    """Test a minimal recovery run."""
    print("\nTesting minimal MoM recovery run...")
    print("This will take 1-2 minutes...")
    
    try:
        from parameter_recovery_mom import MoMParameterRecoveryStudy
        
        # Create minimal study
        study = MoMParameterRecoveryStudy(
            mechanism='fixed_burst_feedback_onion',
            n_recovery_runs=2,  # Very few runs
            synthetic_data_size=50,  # Small synthetic dataset
            gamma_mode='separate'
        )
        
        print("  ✓ Study initialized")
        
        # Test ground truth loading
        try:
            ground_truth = study.load_ground_truth_parameters()
            print("  ✓ Ground truth loaded")
        except Exception as e:
            print(f"  ✗ Ground truth loading failed: {e}")
            return False
        
        # Test synthetic data generation
        try:
            synthetic_datasets = study.generate_synthetic_data(ground_truth)
            print("  ✓ Synthetic data generated")
            
            for strain, data in synthetic_datasets.items():
                print(f"    {strain}: {len(data['delta_t12'])} T1-T2, {len(data['delta_t32'])} T3-T2 points")
        except Exception as e:
            print(f"  ✗ Synthetic data generation failed: {e}")
            return False
        
        # Test single recovery objective evaluation
        try:
            # Create a test parameter vector
            import numpy as np
            test_params = np.array([param for param in ground_truth.values() if param is not None][:len(study.mechanism_info['params'])])
            
            if len(test_params) == len(study.mechanism_info['params']):
                nll = study.recovery_objective(test_params, synthetic_datasets, ground_truth)
                print(f"  ✓ Recovery objective evaluation successful (NLL: {nll:.2f})")
            else:
                print(f"  ⚠ Parameter count mismatch: {len(test_params)} vs {len(study.mechanism_info['params'])}")
                return False
                
        except Exception as e:
            print(f"  ✗ Recovery objective test failed: {e}")
            return False
        
        print("  ✓ All MoM recovery components working")
        return True
        
    except Exception as e:
        print(f"  ✗ Minimal recovery test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("MoM Parameter Recovery Study - Local Test")
    print("=" * 60)
    print(f"Date: {datetime.now()}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    tests = [
        ("Import test", test_imports),
        ("File structure test", test_file_structure),
        ("Parameter file test", test_parameter_file),
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
        print("The MoM parameter recovery system is working correctly.")
        print()
        print("Next steps:")
        print("1. Run parameter_recovery_mom.py for the full study")
        print("2. Run analyze_parameter_recovery_mom.py to analyze results")
        print("3. Compare with simulation-based results from ARC")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please fix the issues before running the full study.")
        
        if not results[2][1]:  # Parameter file test failed
            print("\nNote: If parameter file test failed, you may need to:")
            print("1. Run MoMOptimization_join.py first to generate optimized parameters")
            print("2. Ensure the mechanism name matches what you want to test")
        
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
