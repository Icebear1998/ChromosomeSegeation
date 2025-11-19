# Simulation-Based Optimization with KDE - Summary

## What Was Added

Successfully integrated simulation-based optimization for simple and fixed_burst mechanisms using Kernel Density Estimation (KDE) instead of the Method of Moments (MoM) normal approximation.

---

## New Files Created

### 1. **`SecondVersion/simulation_kde.py`** (~140 lines)
Simple, clean KDE module based on scipy.stats.gaussian_kde:

**Key Functions:**
- `build_kde_from_simulations(data, bandwidth=None)` - Build KDE (uses Scott's rule if bandwidth=None)
- `evaluate_kde_pdf(kde, x_values)` - Evaluate KDE at points
- `calculate_kde_likelihood(kde, data)` - Calculate NLL
- `quick_kde_pdf(sim_data, x_grid, bandwidth=10.0)` - One-liner convenience function

**No manual bandwidth needed** - uses Scott's rule by default for automatic selection.

---

### 2. **`test_simple_optimization.py`** (~70 lines)
Quick test script to verify the simple/fixed_burst optimization works correctly.

---

## Modified Files

### 1. **`SimulationOptimization_join.py`**
Added support for simple/fixed_burst mechanisms:

**New Functions:**
- `run_simple_simulation_for_dataset()` - Run simulations using `SecondVersion/MultiMechanismSimulation`
- `calculate_likelihood_kde()` - Calculate NLL using KDE (no manual bandwidth)
- `joint_objective_simple_mechanisms()` - Objective function for simple mechanisms
- `get_parameter_bounds_simple_mechanisms()` - Parameter bounds
- `run_optimization_simple_mechanisms()` - Main optimization runner

**Updated `main()` function:**
- Now automatically detects mechanism type
- Routes to appropriate optimization function

---

### 2. **`SecondVersion/TestAllMechanisms.py`**
Enhanced visualization with KDE:

**Changes:**
- Added KDE import and integration
- Now shows 3 comparisons:
  1. Simulation histogram (blue transparent)
  2. **Simulation KDE** (blue solid line) - NEW!
  3. MoM PDF (red dashed line)
- Added vertical lines for means
- Grid for better readability
- Configurable KDE bandwidth parameter

---

### 3. **`model_comparison_aic_bic.py`**
Integrated simulation-based mechanisms:

**New Mechanism Names:**
- `simple_simulation` - Simple mechanism with KDE (9 params)
- `fixed_burst_simulation` - Fixed burst with KDE (10 params)
- `feedback_onion_simulation` - Feedback onion with KDE (10 params)
- `fixed_burst_feedback_onion_simulation` - Combined with KDE (11 params)

**Changes:**
- Added `run_optimization_simple_mechanisms` import
- Updated `get_parameter_count()` to include simulation variants
- Modified `run_single_optimization()` to detect `_simulation` suffix
- Updated mechanism type labels

---

## Key Differences: MoM vs Simulation

| Aspect | MoM-based | Simulation-based (KDE) |
|--------|-----------|------------------------|
| **Method** | Normal approximation | KDE from simulations |
| **Parameters** | 11-13 (includes beta2_k, beta3_k) | 9-11 (no beta2_k, beta3_k) |
| **Speed** | Very fast (~seconds) | Slower (~minutes) |
| **Accuracy** | Assumes normal distribution | Captures full distribution |
| **Mutants** | All 5 strains | Simplified (3 main types) |

---

## Bug Fixes (from earlier)

### 1. **Fixed `feedback_onion` mechanism**
**Problem:** Simulation used initial state `N_i` instead of current state `m` for feedback weight.

**Fix:** Changed to use `current_state` in propensity calculation.

### 2. **Fixed `fixed_burst_feedback_onion` mechanism**
**Problem:** MoM calculation summed over individual cohesins instead of burst events.

**Fix:** Now properly iterates through burst events (stepping by `burst_size`).

---

## Usage Examples

### Test Simple Mechanism Optimization
```bash
python test_simple_optimization.py
```

### Run Model Comparison
```python
# In model_comparison_aic_bic.py, update mechanisms list:
mechanisms = [
    'simple_simulation',              # 9 params
    'fixed_burst_simulation',         # 10 params
]

# Then run:
python model_comparison_aic_bic.py
```

### Test All Mechanisms (Validation)
```bash
cd SecondVersion
python TestAllMechanisms.py
```

---

## Parameter Counts

### MoM-based (Normal Approximation)
- `simple`: 11 params (n2, N2, k, r21, r23, R21, R23, alpha, beta_k, beta2_k, beta3_k)
- `fixed_burst`: 12 params (+ burst_size)
- `feedback_onion`: 12 params (+ n_inner)
- `fixed_burst_feedback_onion`: 13 params (+ burst_size, n_inner)

### Simulation-based (KDE)
- `simple_simulation`: 9 params (no beta2_k, beta3_k)
- `fixed_burst_simulation`: 10 params (+ burst_size)
- `feedback_onion_simulation`: 10 params (+ n_inner)
- `fixed_burst_feedback_onion_simulation`: 11 params (+ burst_size, n_inner)

### Time-varying (Simulation-based)
- `time_varying_k`: 12 params
- `time_varying_k_fixed_burst`: 13 params
- `time_varying_k_feedback_onion`: 13 params
- `time_varying_k_combined`: 14 params

---

## Design Philosophy

1. **Simplicity**: KDE module is ~140 lines, easy to understand
2. **Automatic bandwidth**: Uses Scott's rule by default (no manual tuning needed)
3. **No bootstrapping**: Removed complexity, direct KDE approach
4. **Reusable**: `simulation_kde.py` can be used anywhere in the project
5. **Clean separation**: MoM vs Simulation clearly separated in code

---

## Next Steps

To compare MoM vs Simulation approaches:

1. **Run both versions:**
   ```python
   mechanisms = [
       'simple',              # MoM with normal approx
       'simple_simulation',   # Simulation with KDE
   ]
   ```

2. **Compare results:**
   - Parameter estimates
   - NLL values
   - AIC/BIC scores
   - Computational time

3. **Expected outcome:**
   - Simulation-KDE should fit better (lower NLL)
   - But has fewer parameters (9 vs 11)
   - Trade-off between model complexity and fit quality

---

## Files Modified Summary

✅ **Created:**
- `SecondVersion/simulation_kde.py`
- `test_simple_optimization.py`

✅ **Modified:**
- `SimulationOptimization_join.py` (added ~200 lines)
- `SecondVersion/TestAllMechanisms.py` (enhanced visualization)
- `SecondVersion/MultiMechanismSimulation.py` (fixed bugs)
- `SecondVersion/MoMCalculations.py` (fixed bugs)
- `model_comparison_aic_bic.py` (integrated new mechanisms)

✅ **Bug fixes:**
- feedback_onion: Now uses current state for feedback weight
- fixed_burst_feedback_onion: Now sums over burst events correctly

---

**Date:** November 17, 2024  
**Status:** ✅ Complete and tested

