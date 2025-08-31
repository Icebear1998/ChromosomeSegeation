#!/usr/bin/env python3
"""
Test script to verify incremental saving functionality works correctly.
"""

import os
import pandas as pd
import numpy as np
from run_recovery_study import save_individual_result

def test_incremental_saving():
    """Test the incremental saving functionality"""
    print("Testing incremental saving functionality...")
    
    test_file = "test_incremental_results.csv"
    
    # Clean up any existing test file
    if os.path.exists(test_file):
        os.remove(test_file)
    
    # Test data - simulating recovery results
    test_results = [
        {
            'run_id': 0,
            'converged': True,
            'final_nll': 45.67,
            'elapsed_time': 123.4,
            'n2': 5.12,
            'N2': 168.84,
            'k_max': 0.01314,
            'tau': 47.26,
            'n2_truth': 5.12,
            'N2_truth': 168.84,
        },
        {
            'run_id': 1,
            'converged': True,
            'final_nll': 46.12,
            'elapsed_time': 134.5,
            'n2': 5.08,
            'N2': 170.12,
            'k_max': 0.01298,
            'tau': 48.15,
            'n2_truth': 5.12,
            'N2_truth': 168.84,
        },
        {
            'run_id': 2,
            'converged': False,
            'final_nll': 1e6,
            'elapsed_time': 89.2,
            'n2': np.nan,
            'N2': np.nan,
            'k_max': np.nan,
            'tau': np.nan,
            'n2_truth': 5.12,
            'N2_truth': 168.84,
        }
    ]
    
    # Test incremental saving
    print("Saving results incrementally...")
    for i, result in enumerate(test_results):
        print(f"Saving result {i+1}...")
        save_individual_result(result, test_file)
        
        # Verify the file exists and has the right number of rows
        if os.path.exists(test_file):
            df = pd.read_csv(test_file)
            expected_rows = i + 1
            actual_rows = len(df)
            print(f"  File has {actual_rows} rows (expected {expected_rows})")
            
            if actual_rows != expected_rows:
                print(f"  ❌ ERROR: Expected {expected_rows} rows, got {actual_rows}")
                return False
        else:
            print(f"  ❌ ERROR: File {test_file} doesn't exist after saving")
            return False
    
    # Verify final results
    print("\nVerifying final results...")
    final_df = pd.read_csv(test_file)
    
    print(f"Final CSV has {len(final_df)} rows and {len(final_df.columns)} columns")
    print("Columns:", list(final_df.columns))
    print("\nFirst few rows:")
    print(final_df.head())
    
    # Check that we have the expected data
    expected_run_ids = [0, 1, 2]
    actual_run_ids = sorted(final_df['run_id'].tolist())
    
    if actual_run_ids == expected_run_ids:
        print("✅ All run IDs present and correct")
    else:
        print(f"❌ ERROR: Expected run IDs {expected_run_ids}, got {actual_run_ids}")
        return False
    
    # Check convergence status
    converged_count = final_df['converged'].sum()
    expected_converged = 2  # First two should be True
    
    if converged_count == expected_converged:
        print(f"✅ Convergence status correct ({converged_count}/{len(final_df)} converged)")
    else:
        print(f"❌ ERROR: Expected {expected_converged} converged, got {converged_count}")
        return False
    
    # Clean up
    os.remove(test_file)
    print(f"✅ Cleaned up test file: {test_file}")
    
    return True

if __name__ == '__main__':
    print("=" * 60)
    print("INCREMENTAL SAVING TEST")
    print("=" * 60)
    
    try:
        success = test_incremental_saving()
        
        if success:
            print("\n🎉 ALL TESTS PASSED!")
            print("Incremental saving is working correctly.")
        else:
            print("\n❌ TESTS FAILED!")
            print("There are issues with incremental saving.")
            
    except Exception as e:
        print(f"\n💥 TEST CRASHED: {e}")
        import traceback
        traceback.print_exc()
