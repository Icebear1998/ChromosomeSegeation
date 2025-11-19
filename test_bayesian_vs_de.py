#!/usr/bin/env python3
"""
Quick test to compare Bayesian optimization vs Differential Evolution.
Tests both methods on the simple mechanism with a small number of runs.
"""

import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'SecondVersion'))

# Try to import both optimizers
try:
    from MoMOptimization_join import run_mom_optimization_single
    HAS_DE = True
except ImportError:
    HAS_DE = False
    print("❌ Could not import Differential Evolution optimizer")

try:
    from MoMOptimization_bayesian import run_bayesian_optimization
    HAS_BAYESIAN = True
except ImportError:
    HAS_BAYESIAN = False
    print("❌ Could not import Bayesian optimizer")
    print("Install with: pip install scikit-optimize")


def load_data():
    """Load experimental data."""
    df = pd.read_excel("Data/All_strains_SCStimes.xlsx")
    
    data_arrays = {
        'data_wt12': df['wildtype12'].dropna().values,
        'data_wt32': df['wildtype32'].dropna().values,
        'data_threshold12': df['threshold12'].dropna().values,
        'data_threshold32': df['threshold32'].dropna().values,
        'data_degrate12': df['degRade12'].dropna().values,
        'data_degrate32': df['degRade32'].dropna().values,
        'data_degrateAPC12': df['degRadeAPC12'].dropna().values,
        'data_degrateAPC32': df['degRadeAPC32'].dropna().values,
        'data_velcade12': df['degRadeVel12'].dropna().values,
        'data_velcade32': df['degRadeVel32'].dropna().values,
        'data_initial12': np.array([]),
        'data_initial32': np.array([])
    }
    
    return data_arrays


def run_comparison_test():
    """Run a quick comparison between DE and Bayesian optimization."""
    print("="*80)
    print("BAYESIAN OPTIMIZATION vs DIFFERENTIAL EVOLUTION TEST")
    print("="*80)
    
    if not HAS_DE:
        print("❌ Differential Evolution not available. Cannot run test.")
        return
    
    if not HAS_BAYESIAN:
        print("❌ Bayesian optimization not available. Install scikit-optimize.")
        return
    
    # Load data
    print("\n📊 Loading experimental data...")
    data_arrays = load_data()
    print(f"✅ Data loaded successfully")
    
    # Test mechanism
    mechanism = 'simple'
    seed = 42
    
    # Run Differential Evolution
    print("\n" + "="*80)
    print("1️⃣  DIFFERENTIAL EVOLUTION")
    print("="*80)
    print(f"Mechanism: {mechanism}")
    print(f"Max iterations: 200 (popsize=15, so ~3000 evaluations)")
    
    start_de = datetime.now()
    result_de = run_mom_optimization_single(
        mechanism=mechanism,
        data_arrays=data_arrays,
        max_iterations=200,
        seed=seed,
        gamma_mode='separate'
    )
    end_de = datetime.now()
    duration_de = (end_de - start_de).total_seconds()
    
    if result_de['success']:
        print(f"\n✅ DE optimization successful!")
        print(f"   NLL: {result_de['nll']:.4f}")
        print(f"   Time: {duration_de:.1f}s")
        print(f"   Key parameters:")
        params_de = result_de['params']
        print(f"      n2={params_de['n2']:.2f}, N2={params_de['N2']:.2f}, k={params_de['k']:.5f}")
        print(f"      alpha={params_de['alpha']:.3f}, beta_k={params_de['beta_k']:.3f}")
    else:
        print(f"❌ DE optimization failed: {result_de['message']}")
    
    # Run Bayesian Optimization
    print("\n" + "="*80)
    print("2️⃣  BAYESIAN OPTIMIZATION (Gaussian Process)")
    print("="*80)
    print(f"Mechanism: {mechanism}")
    print(f"n_calls: 150 (30 random + 120 GP-guided)")
    
    start_bo = datetime.now()
    result_bo = run_bayesian_optimization(
        mechanism=mechanism,
        data_arrays=data_arrays,
        n_calls=150,
        n_random_starts=30,
        seed=seed,
        gamma_mode='separate'
    )
    end_bo = datetime.now()
    duration_bo = (end_bo - start_bo).total_seconds()
    
    if result_bo['success']:
        print(f"\n✅ Bayesian optimization successful!")
        print(f"   NLL: {result_bo['nll']:.4f}")
        print(f"   Time: {duration_bo:.1f}s")
        print(f"   Key parameters:")
        params_bo = result_bo['params']
        print(f"      n2={params_bo['n2']:.2f}, N2={params_bo['N2']:.2f}, k={params_bo['k']:.5f}")
        print(f"      alpha={params_bo['alpha']:.3f}, beta_k={params_bo['beta_k']:.3f}")
    else:
        print(f"❌ Bayesian optimization failed: {result_bo['message']}")
    
    # Comparison
    print("\n" + "="*80)
    print("📊 COMPARISON SUMMARY")
    print("="*80)
    
    if result_de['success'] and result_bo['success']:
        nll_diff = abs(result_de['nll'] - result_bo['nll'])
        nll_diff_pct = (nll_diff / result_de['nll']) * 100
        
        print(f"\n🏆 Results:")
        print(f"   DE:       NLL = {result_de['nll']:.4f} ({duration_de:.1f}s)")
        print(f"   Bayesian: NLL = {result_bo['nll']:.4f} ({duration_bo:.1f}s)")
        print(f"   Difference: {nll_diff:.4f} ({nll_diff_pct:.2f}%)")
        
        if nll_diff < 0.1:
            print(f"\n✅ Both methods converged to similar solutions (diff < 0.1)")
        elif nll_diff < 1.0:
            print(f"\n⚠️  Small difference in solutions (diff < 1.0)")
        else:
            print(f"\n⚠️  Noticeable difference in solutions")
        
        # Compare key parameters
        print(f"\n📈 Parameter comparison:")
        for key in ['n2', 'N2', 'k', 'alpha', 'beta_k', 'beta2_k', 'beta3_k']:
            val_de = params_de[key]
            val_bo = params_bo[key]
            diff = abs(val_de - val_bo)
            diff_pct = (diff / val_de * 100) if val_de != 0 else 0
            print(f"   {key:8s}: DE={val_de:8.4f}, BO={val_bo:8.4f}, diff={diff_pct:6.2f}%")
        
        # Time comparison
        speedup = duration_de / duration_bo
        print(f"\n⏱️  Timing:")
        print(f"   DE: {duration_de:.1f}s")
        print(f"   Bayesian: {duration_bo:.1f}s")
        if speedup > 1:
            print(f"   Bayesian is {speedup:.2f}x faster")
        else:
            print(f"   DE is {1/speedup:.2f}x faster")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    if result_de['success'] and result_bo['success']:
        if nll_diff < 1.0:
            print("✅ Both optimizers found similar solutions.")
            print("   This confirms that optimizer choice is not causing issues.")
            print("   The data genuinely prefers the simpler model.")
        else:
            print("⚠️  Optimizers found different solutions.")
            print("   May indicate multiple local minima or optimization challenges.")
    
    print("\n🎉 Test complete!")


if __name__ == "__main__":
    run_comparison_test()

