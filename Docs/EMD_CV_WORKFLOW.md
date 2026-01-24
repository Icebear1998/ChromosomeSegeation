# EMD Optimization and Cross-Validation Workflow

This document details the new workflow for optimizing chromosome segregation models using Earth Mover's Distance (EMD) and validating them via Cross-Validation (CV).

## 🚀 Overview: from NLL to EMD and CV

We have transitioned from Negative Log-Likelihood (NLL) with Kernel Density Estimation (KDE) to **Earth Mover's Distance (EMD)** as our primary optimization metric.

### Why EMD (Wasserstein Distance)?

1.  **Robustness**: EMD measures the "work" needed to transform one distribution into another. Unlike NLL/KDE, it does not require a bandwidth parameter, making it less sensitive to hyperparameter tuning.
2.  **Stability**: It provides a more stable gradient for optimization, especially when distributions are non-overlapping or sparse.
3.  **Interpretability**: The resulting score is in the same units as the data (minutes), representing the average "distance" between the simulated and experimental timing distributions.

### Why Cross-Validation?

To ensure our models generalize well and are not overfitting to specific data points, we implemented **5-Fold Cross-Validation**. This splits the experimental data into training and validation sets, allowing us to assess how well the optimized parameters predict unseen data.

---

## 📂 Key Files & Functions

### 1. `simulation_utils.py`
Contains the core metric implementation.

*   `calculate_emd(exp_data, sim_data)`: Calculates the Wasserstein distance between experimental and simulated datasets using `scipy.stats.wasserstein_distance`.
*   `joint_objective(...)`: updated to accept `objective_metric='emd'`.

### 2. `SimulationOptimization_join.py`
The main training script for fitting models to the full dataset.

*   **Default Metric**: Now defaults to `objective_metric='emd'`.
*   **Usage**: Optimizes parameters to minimize the total EMD across all strain datasets.

### 3. `CrossValidation_EMD.py`
 **[NEW]** Script for rigorous model validation.

*   **Logic**:
    1.  Splits data into 5 folds (randomized, stratified by strain).
    2.  For each fold:
        *   **Train**: Optimize parameters on ~80% of the data.
        *   **Validate**: Calculate EMD on the remaining ~20% held-out data.
    3.  Reports average Train EMD and Validation EMD.

---

## 🛠️ Usage Instructions

### 1. Running Standard Optimization (Full Data)

To optimize a mechanism using EMD on the complete dataset (use this for final parameter estimation):

```bash
python SimulationOptimization_join.py
```

*Configuration in `main()`:*
```python
mechanism = 'feedback_onion'  # or 'time_varying_k_combined', etc.
objective_metric = 'emd'      # 'nll' is still available as a fallback
```

### 2. Running Cross-Validation

To benchmark a mechanism's predictive performance:

```bash
python CrossValidation_EMD.py
```

*Configuration in `__main__`:*
```python
mechanism = 'time_varying_k_combined'
n_simulations = 10000        # Simulations for EMD calculation
max_iter = 1000              # Optimization iterations
```

### 3. Output

*   **Optimization**: Saves results to `simulation_optimized_parameters_{mechanism}.txt` with EMD scores.
*   **Cross-Validation**: Saves fold-by-fold results to `ModelComparisonEMDResults/cv_results_{mechanism}.csv`.

---

## 📊 Interpreting Results

*   **Optimization Score (EMD)**: Lower is better. A score of X means, on average, the simulated points are X minutes away from the experimental points.
*   **CV Generalization**:
    *   Small gap between **Train EMD** and **Val EMD**: Good generalization (not overfitting).
    *   Large gap (Val EMD >> Train EMD): Overfitting.

## 🔄 Workflow Summary

1.  **Develop/Select Mechanism**: Choose a mechanism (e.g., `time_varying_k_combined`).
2.  **Validate**: Run `CrossValidation_EMD.py` to ensure the model generalizes.
3.  **Fit**: Run `SimulationOptimization_join.py` to get the final best-fit parameters on the full dataset.
4.  **Visualize**: Use `TestDataPlotSimulation.py` (ensure it uses the EMD-optimized parameters) to visualize the fit.
