# Simulation vs MoM: Fixed_Burst Mechanism Testing

## 🎯 Purpose

This document explains the key differences between **Method of Moments (MoM)** and **Simulation-based** optimization, and why simulation naturally avoids the discrete summation issue.

---

## 📊 Why Simple Model Has No Issue

### The Simple Model Uses Discrete Cohesin States

```python
# SecondVersion/MoMCalculations.py (lines 81-95)
sum1_Ti = sum(1/m for m in range(final_state_i + 1, int(N_i) + 1))
#                         m = cohesin count (naturally discrete!)
```

**Key insight**: The simple model sums over **cohesin counts** `m`, not over "number of events"!

- Cohesins are discrete: You can have 355 or 356 cohesins, but not 355.7
- `m` goes from `n+1` to `N` (both rounded to integers)
- Each term is `1/m` where `m` is an integer
- **No continuous parameter needs discretization!**

### The Fixed_Burst Model Uses Discrete Bursts

```python
# SecondVersion/MoMCalculations.py (original, lines 97-160)
num_bursts_i = delta_i / burst_size  # e.g., 355 / 1.5 = 236.67
full_bursts_i = int(np.floor(num_bursts_i))  # 236
frac_burst_i = num_bursts_i - full_bursts_i  # 0.67

# Sum over FULL bursts + fractional last burst
mean_Ti = sum(...) + frac_burst_i * (...)
#         ^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^
#         discrete   continuous correction
```

**The problem**: When `burst_size` is continuous (e.g., 1.5), the **number of bursts** becomes fractional!

- `burst_size = 1.5` → `num_bursts = 236.67`
- Can't sum over 236.67 discrete terms!
- **Solution (Option 2)**: Allow fractional last burst

---

## 🔬 How Simulation Avoids This Issue

### Simulation Uses Continuous State Updates

Looking at `MultiMechanismSimulationTimevary.py` (lines 198-209):

```python
# Apply degradation (mechanism-specific burst size)
burst_size = self.calculate_burst_size()  # Can be 1.5 ✅
current_state[selected_chromosome] = max(0, current_state[selected_chromosome] - burst_size)
#                                         Continuous subtraction! ✅

# Check for chromosome separation
if current_state[i] <= round(self.n0_list[i]):
    separate_times[i] = current_time
```

### Key Differences

| Aspect | MoM (Analytical) | Simulation (Stochastic) |
|--------|------------------|------------------------|
| **burst_size** | Continuous parameter | Continuous parameter ✅ |
| **State updates** | Discrete summation | **Continuous subtraction** ✅ |
| **Threshold** | Rounded `n` | Rounded `n` |
| **Calculation** | `Σ(full bursts) + frac` | `state -= burst_size` |
| **Issue** | Discretization mismatch | **No issue!** ✅ |

### Example: burst_size = 1.5

**MoM Approach:**
```python
# Must sum over integer bursts
num_bursts = 355 / 1.5 = 236.67
# Problem: Can't have 0.67 of a summation term!

# Solution: Split into full + fractional
full_bursts = 236
frac_burst = 0.67
mean = Σ(0 to 235) + 0.67 * (term_236)
```

**Simulation Approach:**
```python
# Just subtract continuously!
state = 355.0
state -= 1.5  # 353.5 ✅
state -= 1.5  # 352.0 ✅
state -= 1.5  # 350.5 ✅
# No discretization needed!
```

---

## 🧪 Test Script: `test_simulation_fixed_burst.py`

### What It Does

This comprehensive test script:

1. **Optimizes simple model** with MoM (fast, baseline)
2. **Tests fixed_burst with MoM** (Option 2 - Fractional Burst)
   - Varies `burst_size` from 1.0 to 10.0
   - All other params fixed from simple model
3. **Tests fixed_burst with Simulation**
   - Same `burst_size` values
   - Uses `time_varying_k_fixed_burst` mechanism
4. **Full optimization with Simulation**
   - Optimizes only `burst_size`
   - See if it converges to 1.0
5. **Comparison and visualization**
   - Smoothness metrics
   - NLL profiles
   - Publication-quality plots

### Parameters in Simulation

From your question: **"All parameters are kept continuous, right?"**

**Answer**: Almost all!

```python
# Continuous parameters
burst_size = 1.5           # ✅ Continuous
current_state = [355.0, ...]  # ✅ Continuous (becomes 353.5, 352.0, etc.)
n0_list = [1.5, 2.3, ...]  # ✅ Continuous

# Discrete check (but using continuous values)
if current_state[i] <= round(self.n0_list[i]):  # round() only for threshold check
    separate_times[i] = current_time
```

So:
- ✅ `burst_size` is continuous
- ✅ `current_state` becomes continuous (355.0 → 353.5 → 352.0)
- ✅ `n0_list` can be continuous (e.g., 1.53)
- ⚠️ Threshold check uses `round(n0_list[i])` for discrete comparison

This is the **best of both worlds**:
- Continuous evolution of the state
- Discrete threshold for biological realism (can't have 1.53 cohesins)

---

## 📈 Expected Results

### If Data Prefers Simple Model (burst_size ≈ 1):

```
✅ MoM optimization: burst_size → 1.0
✅ Simulation optimization: burst_size → 1.0
✅ Both methods agree!
```

### If Simulation Shows Different Result:

```
⚠️ MoM: burst_size = 1.0
⚠️ Sim: burst_size = 2.5

Possible reasons:
1. MoM approximation breaks down (unlikely)
2. Simulation needs more runs (stochastic noise)
3. Different parameterization (k_1/k_max vs k)
```

### Smoothness Comparison

**Expected:**
- MoM: Max jump ≈ 150 (from Option 2 test)
- Simulation: Max jump ≈ 50-200 (stochastic noise)
- Both should be **monotonic** (NLL increases with burst_size)

---

## 🚀 How to Run

### Quick Test (recommended):

```bash
cd /Users/kienphan/WorkingSpace/ResearchProjs/ChromosomeSegeation
python test_simulation_fixed_burst.py
```

**Runtime**: ~10-15 minutes (300 simulations per evaluation)

### Full Test (more robust):

Edit the script to increase:
```python
num_simulations=500  # More simulations (line 95)
maxiter=50           # More DE iterations (line 227)
```

**Runtime**: ~30-45 minutes

---

## 📊 Output

### Terminal Output

The script prints:
1. Simple model optimization results
2. MoM fixed_burst NLL profile
3. Simulation fixed_burst NLL profile
4. Full simulation optimization
5. Smoothness comparison
6. Summary and recommendations

### Generated Files

- `simulation_vs_mom_fixed_burst.png`: Publication-quality comparison plot
  - Left panel: Full NLL profile (burst_size 1.0 to 10.0)
  - Right panel: Zoomed view (burst_size 1.0 to 2.0)

---

## 🎯 Key Questions Answered

### Q1: Why doesn't simple model have the discrete summation issue?

**A**: Simple model sums over **discrete cohesin counts** (m = 1, 2, ..., N), not over "number of events". Cohesins are naturally discrete, so no continuous parameter needs discretization!

### Q2: In simulation, are all parameters continuous?

**A**: Almost!
- ✅ `burst_size` is continuous (e.g., 1.5)
- ✅ `current_state` becomes continuous (355.0 → 353.5)
- ✅ `n0_list` thresholds are continuous
- ⚠️ Threshold **check** uses `round(n0_list)` for comparison

### Q3: When does simulation finish?

**A**: When `current_state[i] <= round(n0_list[i])` for all chromosomes. The state is continuous, but the threshold is rounded for comparison.

### Q4: Would simulation completely fix the discrete summation issue?

**A**: **YES!** Simulation uses continuous state updates (`state -= burst_size`), not discrete summations. There's no discretization mismatch because the Gillespie algorithm naturally handles continuous decrements.

---

## 🏆 Recommendation

Based on your analysis goals:

### For Model Comparison (Current Work):
✅ **Use MoM with Option 2 (Fractional Burst)**
- Fast enough for 1000s of runs
- Smooth enough for optimization
- Accurate enough for your data

### For Final Validation:
🔬 **Use Simulation-based Optimization**
- Run `test_simulation_fixed_burst.py`
- Verify burst_size → 1.0
- Confirm simple model is best

### For Publication:
📊 **Report Both Methods**
- Main results: MoM (fast, efficient)
- Validation: Simulation (robust, exact)
- Show they agree!

---

## 📝 Technical Notes

### Why Simulation is Slower

```python
# MoM: Analytical formula (fast)
mean_T = Σ(...) + frac * (...)  # ~0.001 seconds

# Simulation: Stochastic runs (slow)
for run in range(num_simulations):
    # Gillespie algorithm: many events
    while state > threshold:
        # Sample exponential times
        # Update states
        # Check thresholds
    record_separation_time()
# ~0.1-1.0 seconds for 300 runs
```

**Speed ratio**: MoM is **100-1000x faster** than simulation!

### Why Both Are Valuable

| Method | Pros | Cons | Use Case |
|--------|------|------|----------|
| **MoM** | Fast, analytical, smooth | Approximation | Model comparison, screening |
| **Simulation** | Exact, robust, no assumptions | Slow, stochastic | Validation, final results |

---

## 🎉 Bottom Line

1. ✅ **Simple model has no issue** because it sums over discrete cohesins, not events
2. ✅ **Simulation naturally handles continuous burst_size** via continuous state updates
3. ✅ **Option 2 (Fractional Burst) is good enough** for your MoM analysis
4. ✅ **Simulation validation is recommended** to confirm your findings
5. ✅ **Both methods should agree** that burst_size ≈ 1.0 is optimal!

Run the test script to verify! 🚀

