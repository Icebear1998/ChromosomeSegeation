# Chromosome Segregation Modeling Project

## Project Overview

This project implements mathematical models to study chromosome segregation timing in biological systems. The goal is to understand how different molecular mechanisms affect the timing of chromosome separation during cell division by fitting model parameters to experimental data from various mutant strains.

## Biological Context

### Chromosome Segregation Process

During cell division, chromosomes must be properly separated to ensure each daughter cell receives the correct genetic material. This process involves:

1. **Cohesin proteins** that hold sister chromatids together
2. **Degradation mechanisms** that remove cohesin proteins
3. **Separation timing** that determines when chromosomes are released

### Experimental Data

The project uses experimental data from different mutant strains:

- **Wild-type**: Normal chromosome segregation timing
- **Threshold mutants**: Reduced cohesin threshold for separation (alpha parameter)
- **Separase mutants**: Reduced degradation rate (beta_k parameter)
- **APC mutants**: Increased tau (time to reach maximum degradation rate) (beta2_tau parameter)
- **Initial protein mutants**: Reduced initial cohesin protein levels (gamma parameters)

## Mathematical Framework

### Core Model

The model describes the time difference between degradation of two proteins:

```
X = T₁ - T₂
```

Where:

- `T₁` = Time for chromosome 1 to reach threshold
- `T₂` = Time for chromosome 2 to reach threshold
- `X` = Time difference (the quantity of interest)

### Key Parameters

- `Nᵢ`: Initial cohesin count for chromosome i
- `nᵢ`: Threshold cohesin count for chromosome i
- `k`: Base degradation rate
- Additional mechanism-specific parameters

### Method of Moments (MoM) Approximation

The complex probability density function (PDF) is approximated as a normal distribution:

```
f_X(x) ≈ N(μ_X, σ²_X)
```

Where:

- `μ_X = μ_T₁ - μ_T₂` (mean time difference)
- `σ²_X = σ²_T₁ + σ²_T₂` (variance of time difference)

## Implemented Mechanisms

### 1. Simple Mechanism

**Description**: Basic harmonic sum degradation
**Mathematical Form**:

- Mean: `μ_T = (1/k) * Σ(1/m) for m from n+1 to N`
- Variance: `σ²_T = (1/k²) * Σ(1/m²) for m from n+1 to N`

### 2. Fixed Burst Mechanism

**Description**: Degradation occurs in fixed-size bursts
**Parameters**: `burst_size`
**Mathematical Form**:

- Number of bursts: `ceil((N - n) / burst_size)`
- Each burst removes `burst_size` cohesins

### 3. Time-Varying k Mechanism

**Description**: Degradation rate increases linearly with time until reaching maximum
**Parameters**: `k_1`, `k_max`, `tau = k_max/k_1`
**Mathematical Form**: `k(t) = min(k_1 * t, k_max)`
**Time Scale**: `tau` represents the time (in minutes) to reach maximum degradation rate
**Biological Significance**: `tau` reflects the activation timescale of the degradation machinery

### 4. Feedback Mechanisms

#### 4.1 Linear Feedback

**Description**: Degradation rate modified by linear feedback
**Parameters**: `w₁, w₂, w₃`
**Mathematical Form**: `W(m) = 1 - w * m`

#### 4.2 Onion Feedback

**Description**: Degradation rate modified by onion-like structure
**Parameters**: `n_inner`
**Mathematical Form**:

- `W(m) = (N/n_inner)^(-1/3)` if N > n_inner
- `W(m) = 1` otherwise

#### 4.3 Zipper Feedback

**Description**: Degradation rate inversely proportional to initial count
**Parameters**: `z₁, z₂, z₃`
**Mathematical Form**: `W(m) = z / N`

### 5. Combined Mechanisms

#### 5.1 Fixed Burst + Linear Feedback

**Description**: Combines burst degradation with linear feedback effects
**Parameters**: `burst_size, w₁, w₂, w₃`

#### 5.2 Fixed Burst + Onion Feedback

**Description**: Combines burst degradation with onion feedback effects
**Parameters**: `burst_size, n_inner`

#### 5.3 Time-Varying k + Fixed Burst

**Description**: Combines time-varying degradation rate with fixed-size bursts
**Parameters**: `k_1, k_max, burst_size`
**Mathematical Form**: `k(t) = min(k_1 * t, k_max)` with burst-based degradation

#### 5.4 Time-Varying k + Onion Feedback

**Description**: Combines time-varying degradation rate with onion feedback
**Parameters**: `k_1, k_max, n_inner`
**Mathematical Form**: `k(t) = min(k_1 * t, k_max)` with feedback modification

#### 5.5 Time-Varying k + Fixed Burst + Onion Feedback (Combined)

**Description**: Triple combination of time-varying rate, burst degradation, and onion feedback
**Parameters**: `k_1, k_max, burst_size, n_inner`
**Mathematical Form**: All three mechanisms applied simultaneously

## Implementation Structure

### Core Files

#### 1. `MoMCalculations.py`

- **Purpose**: Implements Method of Moments calculations for all mechanisms
- **Key Functions**:
  - `compute_moments_mom()`: Calculate mean and variance for any mechanism
  - `compute_pdf_mom()`: Generate PDF using normal approximation
  - `compute_pdf_for_mechanism()`: Route parameters to correct mechanism

#### 2. `MultiMechanismSimulation.py`

- **Purpose**: Implements stochastic Gillespie simulations for all mechanisms
- **Key Features**:
  - Discrete state simulation (integer cohesin counts)
  - Proper rounding for threshold values
  - Support for all implemented mechanisms

#### 2b. `MultiMechanismSimulationTimevary.py`

- **Purpose**: Extended simulation engine for time-varying mechanisms
- **Key Features**:
  - Time-varying degradation rates: `k(t) = min(k_1 * t, k_max)`
  - Combined mechanisms (time-varying + burst + feedback)
  - Support for all time-varying mechanism combinations
  - Optimized for simulation-based optimization

#### 3. MoM-based Optimization

##### 3a. `MoMOptimization_join.py`

- **Purpose**: Joint MoM-based optimization across all mutant strains
- **Strategy**: Optimize all parameters simultaneously using Method of Moments
- **Advantages**: Fast computation, analytical PDF
- **Disadvantages**: Normal approximation may not be accurate

##### 3b. `MoMOptimization_independent.py`

- **Purpose**: Independent MoM-based optimization strategy
- **Strategy**: Optimize wild-type first, then mutants separately
- **Advantages**: More robust convergence
- **Disadvantages**: May miss parameter correlations

#### 4. Simulation-based Optimization

##### 4a. `SimulationOptimization_join.py`

- **Purpose**: Joint optimization using stochastic simulations
- **Strategy**: Use KDE from simulation data for likelihood calculation
- **Key Features**:
  - Direct simulation-based parameter fitting
  - Tau parameterization: `tau = k_max/k_1` (time to reach max rate)
  - Support for time-varying mechanisms
  - Strain selection capability
- **Advantages**: No approximation, captures full distribution
- **Disadvantages**: Computationally expensive

##### 4b. `SimulationOptimization_independent.py`

- **Purpose**: Independent simulation-based optimization
- **Strategy**: Optimize wild-type first, then mutants independently
- **Key Features**: Same as joint but with sequential optimization
- **Advantages**: More robust convergence, easier debugging
- **Disadvantages**: Longer computation time

##### 4c. `SimulationOptimization_bayesian.py`

- **Purpose**: Bayesian optimization for simulation-based fitting
- **Strategy**: Uses Gaussian Process surrogate models
- **Key Features**: Efficient exploration of parameter space
- **Status**: Experimental, requires scikit-optimize

#### 5. Visualization and Testing

##### 5a. `TestDataPlot.py`

- **Purpose**: Visualization for MoM-based optimization results
- **Features**: Comparison of experimental data, simulation, and MoM PDF

##### 5b. `TestDataPlotSimulation.py`

- **Purpose**: Visualization for simulation-based optimization results
- **Features**:
  - Single and multi-dataset plotting
  - Support for tau parameterization
  - Backward compatibility with old parameter files
  - Statistical comparison overlays

### Parameter Handling

#### Wild-Type Parameters

**Base Parameters:**

- `n₂, N₂`: Base threshold and initial count for chromosome 2
- `r₂₁, r₂₃`: Ratios for chromosome 1 and 3 thresholds (n₁/n₂, n₃/n₂)
- `R₂₁, R₂₃`: Ratios for chromosome 1 and 3 initial counts (N₁/N₂, N₃/N₂)

**Rate Parameters (New Tau Parameterization):**

- `k_max`: Maximum degradation rate
- `tau`: Time to reach maximum rate (minutes), where `tau = k_max/k_1`
- `k_1`: Initial degradation rate (derived: `k_1 = k_max/tau`)

**Mechanism-Specific Parameters:**

- `burst_size`: Size of degradation bursts (for burst mechanisms)
- `n_inner`: Inner threshold for onion feedback
- `w₁, w₂, w₃`: Linear feedback weights

#### Mutant Parameters

- `alpha`: Threshold reduction factor (threshold mutants)
- `beta_k`: Maximum degradation rate reduction factor (separase mutants)
- `beta2_tau`: Tau increase factor (APC mutants) - makes tau 2-3x larger
- `gamma` or `gamma₁, gamma₂, gamma₃`: Initial protein reduction factors

#### Gamma Mode Options

- **Unified**: Single `gamma` affects all chromosomes
- **Separate**: Individual `gamma₁, gamma₂, gamma₃` for each chromosome

## Data Structure

### Experimental Data File: `Data/All_strains_SCStimes.xlsx`

Contains time difference measurements for different mutant strains:

| Column              | Description                          | Data Points |
| ------------------- | ------------------------------------ | ----------- |
| `wildtype12`        | Wild-type Chrom1-Chrom2              | 126         |
| `wildtype32`        | Wild-type Chrom3-Chrom2              | 145         |
| `threshold12`       | Threshold mutant Chrom1-Chrom2       | 41          |
| `threshold32`       | Threshold mutant Chrom3-Chrom2       | 37          |
| `degRate12`         | Separase mutant Chrom1-Chrom2        | 74          |
| `degRate32`         | Separase mutant Chrom3-Chrom2        | 189         |
| `degRateAPC12`      | APC mutant Chrom1-Chrom2             | 103         |
| `degRateAPC32`      | APC mutant Chrom3-Chrom2             | 78          |
| `initialProteins12` | Initial protein mutant Chrom1-Chrom2 | 123         |
| `initialProteins32` | Initial protein mutant Chrom3-Chrom2 | 27          |

## Optimization Process

### Objective Function

**Negative Log-Likelihood (NLL)**:

```
NLL = -Σ log(f_X(x_i))
```

Where `f_X(x_i)` is the PDF value for experimental data point `x_i`.

### Optimization Strategy

1. **Global Optimization**: Differential Evolution

   - Population size: 30
   - Maximum iterations: 500
   - Strategy: 'best1bin'
   - Mutation range: (0.5, 1.0)
   - Recombination rate: 0.7

2. **Local Refinement**: L-BFGS-B

   - Bounded optimization
   - Refines top 5 solutions from global optimization

3. **Independent Strategy**: Basinhopping
   - Used for mutant parameter optimization
   - Helps escape local minima

### Parameter Constraints

#### Biological Constraints:

- **Threshold**: `n_i < N_i` (threshold less than initial count)
- **Ratio bounds**: `r₂₁, r₂₃ ∈ [0.1, 10]` (reasonable chromosome differences)
- **Initial count ratios**: `R₂₁, R₂₃ ∈ [0.1, 10]` (reasonable chromosome differences)

#### Rate Parameter Bounds (Tau Parameterization):

- **Maximum rate**: `k_max ∈ [0.01, 0.2]` (per minute)
- **Time scale**: `tau ∈ [2, 240]` minutes (2 seconds to 4 hours)
- **Derived**: `k_1 = k_max / tau` (automatically calculated)

#### Mutant Parameter Bounds:

- **Threshold mutant**: `alpha ∈ [0.1, 2.0]` (threshold modification)
- **Separase mutant**: `beta_k ∈ [0.1, 2.0]` (rate modification)
- **APC mutant**: `beta2_tau ∈ [2.0, 3.0]` (tau increase factor)

#### Mechanism-Specific Bounds:

- **Burst size**: `burst_size ∈ [1, 50]` (integer values)
- **Inner threshold**: `n_inner ∈ [1, 200]` (feedback parameter)

## Current Status

### Implemented Features

✅ All 8 mathematical mechanisms  
✅ Joint and independent optimization strategies  
✅ Support for 5 mutant strains  
✅ Comprehensive visualization tools  
✅ Parameter validation and constraints  
✅ Stochastic simulation validation

### Recent Improvements

✅ **Simulation-based Optimization Pipeline**: Complete implementation with KDE-based likelihood  
✅ **Tau Parameterization**: Changed from `k_1, k_max` to `k_max, tau` for better biological interpretation  
✅ **Combined Time-Varying Mechanisms**: Added 3 new mechanism combinations  
✅ **APC Mutant Tau Effects**: APC mutants now affect tau (2-3x increase) instead of k_1  
✅ **Enhanced Visualization**: Updated plotting for simulation results and tau parameters  
✅ **Strain Selection**: Added capability to optimize subsets of strains  
✅ **Bayesian Optimization**: Experimental Gaussian Process-based optimization  
✅ **Sanity Check Scripts**: Comprehensive validation for both MoM and simulation pipelines

### Output Files

#### MoM-based Optimization Results:

- `optimized_parameters_{mechanism}_join.txt`
- `optimized_parameters_{mechanism}_independent.txt`

#### Simulation-based Optimization Results:

- `simulation_optimized_parameters_{mechanism}.txt`
- `simulation_optimized_parameters_{mechanism}_{strain_selection}.txt`

Each file contains:

**MoM Files:**

- Wild-type parameters (ratio-based)
- Derived parameters (n₁, n₃, N₁, N₃)
- Mechanism-specific parameters
- Mutant parameters
- Individual and total negative log-likelihoods

**Simulation Files:**

- Base parameters (n₂, N₂)
- Ratio parameters (r₂₁, r₂₃, R₂₁, R₂₃)
- Rate parameters (k_max, tau, derived k_1)
- Mechanism-specific parameters (burst_size, n_inner)
- Mutant parameters (alpha, beta_k, beta2_tau)
- Optimization convergence information

## Usage Examples

### Running Optimization

#### MoM-based Optimization:

```python
# Joint MoM optimization
python MoMOptimization_join.py

# Independent MoM optimization
python MoMOptimization_independent.py
```

#### Simulation-based Optimization:

```python
# Joint simulation optimization
python SimulationOptimization_join.py

# Independent simulation optimization
python SimulationOptimization_independent.py

# Bayesian optimization (experimental)
python SimulationOptimization_bayesian.py
```

### Visualizing Results

#### MoM Results:

```python
# MoM-based results
python TestDataPlot.py
```

#### Simulation Results:

```python
# Simulation-based results
python TestDataPlotSimulation.py
```

### Testing and Validation

```python
# Test all mechanisms
python TestAllMechanisms.py

# Sanity check for MoM pipeline
python SanityCheck_FittingPipeline.py

# Sanity check for simulation pipeline
python SanityCheck_SimulationFitting.py
```

## Future Directions

### Potential Improvements

1. **Model Selection**: Implement AIC/BIC for mechanism comparison
2. **Uncertainty Quantification**: Bootstrap confidence intervals
3. **Cross-Validation**: Assess predictive power
4. **Additional Mechanisms**: Explore new mathematical forms
5. **Parallel Optimization**: Speed up computation

### Biological Applications

1. **Parameter Interpretation**: Relate model parameters to molecular mechanisms
2. **Mutant Prediction**: Predict behavior of new mutant strains
3. **Drug Effects**: Model pharmaceutical interventions
4. **Evolutionary Analysis**: Compare mechanisms across species

## Technical Notes

### Numerical Stability

- Use `max(value, 1e-10)` for safe division
- Return `np.inf` for invalid parameter combinations
- Validate inputs before computation

### Computational Efficiency

**MoM-based Optimization:**

- MoM calculations: ~1ms per evaluation
- Full optimization: 5-30 minutes depending on mechanism complexity

**Simulation-based Optimization:**

- Single simulation run: ~1-10ms per simulation
- KDE calculation: ~10-50ms per evaluation
- Full optimization: 2-8 hours depending on mechanism and iterations
- Bayesian optimization: 30 minutes - 2 hours (more efficient parameter exploration)

### Dependencies

**Core Dependencies:**

- NumPy: Numerical computations
- SciPy: Optimization algorithms and KDE
- Matplotlib: Visualization
- Pandas: Data handling

**Optional Dependencies:**

- scikit-optimize: Bayesian optimization (for SimulationOptimization_bayesian.py)
- seaborn: Enhanced plotting (optional)

## Contact and References

This project implements mathematical models for chromosome segregation timing analysis. For questions about the biological context or experimental data, please consult the relevant biological literature and experimental protocols.

---

_Documentation created: [Current Date]_
_Last updated: [Current Date]_
_Project version: 1.0_
