#!/usr/bin/env python3
"""
Quick test script for simple/fixed_burst mechanism optimization using KDE.
"""

import sys
import os

# Add SecondVersion to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SecondVersion'))

from SimulationOptimization_join import (
    load_experimental_data,
    run_optimization_simple_mechanisms,
    save_results
)

def main():
    print("=" * 70)
    print("Testing Simple/Fixed Burst Optimization with KDE")
    print("=" * 70)
    
    # Load data
    print("\nLoading experimental data...")
    datasets = load_experimental_data()
    
    if not datasets:
        print("Error: Could not load datasets!")
        return
    
    print(f"✓ Loaded {len(datasets)} datasets")
    
    # Test simple mechanism first (fewer parameters, faster)
    mechanism = 'simple'
    
    print(f"\n{'=' * 70}")
    print(f"Testing {mechanism.upper()} mechanism")
    print(f"{'=' * 70}")
    
    try:
        # Run with fewer iterations for testing
        results = run_optimization_simple_mechanisms(
            mechanism=mechanism,
            datasets=datasets,
            max_iterations=50,  # Small number for quick test
            num_simulations=10  # Small number for quick test
        )
        
        print("\n✅ Optimization completed!")
        print(f"Final NLL: {results['nll']:.2f}")
        print(f"Converged: {results['converged']}")
        
        # Save results
        save_results(mechanism, results)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

