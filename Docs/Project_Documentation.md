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
- **Threshold mutants**: Reduced cohesin threshold for separation
- **Separase mutants**: Reduced degradation rate (beta_k parameter)
- **APC mutants**: Alternative degradation rate reduction (beta2_k parameter)
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

**Description**: Degradation rate changes over time
**Parameters**: `k_1`
**Mathematical Form**: `k(t) = k₀ + k₁ * t`

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

#### 3. `MoMOptimization_join.py`

- **Purpose**: Joint optimization across all mutant strains
- **Strategy**: Optimize all parameters simultaneously
- **Advantages**: Better parameter consistency across strains
- **Disadvantages**: More complex optimization landscape

#### 4. `MoMOptimization_independent.py`

- **Purpose**: Independent optimization strategy
- **Strategy**:
  1. Optimize wild-type parameters first
  2. Optimize mutant parameters separately using basinhopping
- **Advantages**: More robust, easier to converge
- **Disadvantages**: May miss global correlations

#### 5. `TestDataPlot.py`

- **Purpose**: Visualization and validation of model fits
- **Features**:
  - Single dataset plotting
  - All datasets in 2x5 layout
  - Comparison of experimental data, simulation, and MoM PDF
  - Statistical summaries

### Parameter Handling

#### Wild-Type Parameters

- `n₂, N₂, k`: Base parameters for chromosome 2
- `r₂₁, r₂₃`: Ratios for chromosome 1 and 3 thresholds
- `R₂₁, R₂₃`: Ratios for chromosome 1 and 3 initial counts
- Mechanism-specific parameters (e.g., `burst_size`, `w₁`, etc.)

#### Mutant Parameters

- `alpha`: Threshold reduction factor (threshold mutants)
- `beta_k`: Degradation rate reduction factor (separase mutants)
- `beta2_k`: Alternative degradation rate reduction factor (APC mutants)
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

- **Biological**: `n_i < N_i` (threshold less than initial count)
- **Mathematical**: All parameters must be positive
- **Numerical**: Degradation rates bounded to prevent instability

## Current Status

### Implemented Features

✅ All 8 mathematical mechanisms  
✅ Joint and independent optimization strategies  
✅ Support for 5 mutant strains  
✅ Comprehensive visualization tools  
✅ Parameter validation and constraints  
✅ Stochastic simulation validation

### Recent Improvements

✅ Fixed discrete state handling (rounding vs. truncation)  
✅ Added APC mutant strain support  
✅ Enhanced 2x5 plotting layout  
✅ Improved parameter bounds and validation

### Output Files

Optimization results are saved as text files:

- `optimized_parameters_{mechanism}_join.txt`
- `optimized_parameters_{mechanism}_independent.txt`

Each file contains:

- Wild-type parameters
- Mechanism-specific parameters
- Mutant parameters
- Individual and total negative log-likelihoods

## Usage Examples

### Running Optimization

```python
# Joint optimization
python MoMOptimization_join.py

# Independent optimization
python MoMOptimization_independent.py
```

### Visualizing Results

```python
# Single dataset
python TestDataPlot.py

# All datasets (uncomment in main section)
plot_all_datasets_2x2(params, mechanism="simple", num_sim=1500)
```

### Testing Mechanisms

```python
# Test all mechanisms
python TestAllMechanisms.py
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

- MoM calculations: ~1ms per evaluation
- Stochastic simulation: ~1-10 seconds per 1500 runs
- Optimization: 5-30 minutes depending on mechanism complexity

### Dependencies

- NumPy: Numerical computations
- SciPy: Optimization algorithms
- Matplotlib: Visualization
- Pandas: Data handling

## Contact and References

This project implements mathematical models for chromosome segregation timing analysis. For questions about the biological context or experimental data, please consult the relevant biological literature and experimental protocols.

---

_Documentation created: [Current Date]_
_Last updated: [Current Date]_
_Project version: 1.0_
