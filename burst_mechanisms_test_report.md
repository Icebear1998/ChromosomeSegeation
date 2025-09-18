# Burst Mechanisms Comparison Test Report

**Date:** September 17, 2025  
**Parameters Source:** `optimized_parameters_fixed_burst_feedback_onion_join.txt`  
**Simulation Runs:** 50 per mechanism

## Test Parameters

- **Initial States:** [265.1, 399.2, 965.5] cohesins (N1, N2, N3)
- **Thresholds:** [14.3, 24.2, 70.6] cohesins (n1, n2, n3)
- **Base Rate:** k = 0.012391
- **Burst Size:** 13.72 (mean for variable distributions)
- **Max Time:** 1000.0 time units

## Mechanisms Tested

### 1. Fixed Burst (`fixed_burst`)

- **Description:** Constant burst size of 13.72 cohesins per event
- **Parameters:** `{'k': 0.012391, 'burst_size': 13.719599}`

### 2. Random Normal Burst (`random_normal_burst`)

- **Description:** Burst sizes from normal distribution with mean=13.72, CV=0.3
- **Parameters:** `{'k': 0.012391, 'burst_size': 13.719599, 'var_burst_size': 16.96}`

### 3. Geometric Burst (`geometric_burst`)

- **Description:** Burst sizes from geometric distribution with mean=13.72
- **Parameters:** `{'k': 0.012391, 'burst_size': 13.719599}`

## Key Results

### Time Difference Statistics (T1-T2)

| Mechanism           | Mean  | Std Dev | Min    | Max   |
| ------------------- | ----- | ------- | ------ | ----- |
| Fixed Burst         | 0.81  | 8.78    | -23.44 | 29.38 |
| Random Normal Burst | -0.63 | 5.66    | -14.57 | 10.05 |
| Geometric Burst     | 0.20  | 9.66    | -16.23 | 31.97 |

### Time Difference Statistics (T3-T2)

| Mechanism           | Mean  | Std Dev | Min    | Max  |
| ------------------- | ----- | ------- | ------ | ---- |
| Fixed Burst         | -1.93 | 6.11    | -17.82 | 8.72 |
| Random Normal Burst | -0.51 | 4.98    | -12.23 | 7.92 |
| Geometric Burst     | -2.66 | 5.25    | -14.78 | 7.29 |

### Computational Performance

| Mechanism           | Mean Events | Mean Runtime (ms) |
| ------------------- | ----------- | ----------------- |
| Fixed Burst         | 117         | 0.9               |
| Random Normal Burst | 115         | 0.6               |
| Geometric Burst     | 116         | 0.7               |

## Statistical Analysis

### Significance Tests (T1-T2 differences)

- **Fixed vs Normal:** p = 0.3357 (not significant)
- **Fixed vs Geometric:** p = 0.7445 (not significant)
- **Normal vs Geometric:** p = 0.6026 (not significant)

_None of the mechanisms show statistically significant differences in mean separation times at α = 0.05 level._

## Key Observations

### 1. **Variability Patterns**

- **Random Normal Burst** shows the **lowest variability** (std dev ~5.66 for T1-T2)
- **Geometric Burst** shows the **highest variability** (std dev ~9.66 for T1-T2)
- **Fixed Burst** shows intermediate variability

### 2. **Computational Efficiency**

- **Random Normal Burst** is the **fastest** (0.6 ms average)
- **Geometric Burst** is intermediate (0.7 ms average)
- **Fixed Burst** is the slowest (0.9 ms average)

### 3. **Biological Implications**

- **Fixed Burst:** Deterministic degradation events, suitable for highly regulated processes
- **Random Normal Burst:** Moderate variability with symmetric distribution, good for processes with controlled noise
- **Geometric Burst:** High variability with "memoryless" property, suitable for stochastic biological processes

### 4. **Distribution Characteristics**

- **Normal distribution** produces more predictable, symmetric timing patterns
- **Geometric distribution** produces more extreme values and longer tails
- **Fixed size** provides baseline deterministic behavior

## Recommendations

### For Modeling Applications:

1. **Use Random Normal Burst** when you need:

   - Moderate stochasticity with controlled variance
   - Faster computational performance
   - Symmetric timing distributions

2. **Use Geometric Burst** when you need:

   - High biological realism for memoryless processes
   - Maximum variability in burst sizes
   - Modeling of rare large degradation events

3. **Use Fixed Burst** when you need:
   - Baseline deterministic comparison
   - Simplified model validation
   - Highly regulated biological processes

### For Parameter Fitting:

- The lack of significant differences suggests that **burst size distribution shape** may be less critical than **mean burst size** for fitting experimental data
- Consider using **Random Normal Burst** as a good compromise between biological realism and computational efficiency

## Files Generated

- `burst_mechanisms_comparison.png` - Comprehensive comparison plots
- `burst_mechanisms_test_report.md` - This summary report
- `test_burst_mechanisms.py` - Test script (can be reused)

## Conclusion

All three burst mechanisms produce similar mean separation times but differ significantly in their variability patterns and computational requirements. The choice between mechanisms should be based on the specific biological process being modeled and the desired balance between realism and computational efficiency.
