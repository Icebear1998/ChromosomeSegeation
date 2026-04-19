#!/usr/bin/env python3
"""
Script to perform 5-fold Cross-Validation using EMD optimization.
"""

import numpy as np
import pandas as pd
import sys
import os
import time
import json
from scipy.optimize import differential_evolution
from multiprocessing import Pool, cpu_count
from functools import partial

# Import simulation utilities
from simulation_utils import (
    load_experimental_data, 
    apply_mutant_params,
    calculate_emd,
    run_simulation_for_dataset,
    get_parameter_bounds,
    get_parameter_names,
    parse_parameters
)

def create_folds(data_dict, k_folds=5, seed=42):
    """
    Split experimental data into k folds.
    Returns list of (train_data, val_data) tuples.
    Structure matches the input dict format.
    """
    np.random.seed(seed)
    
    # We need to split the data arrays for each strain
    # data_dict = {'wildtype': {'delta_t12': [...], 'delta_t32': [...]}, ...}
    
    dataset_names = list(data_dict.keys())
    
    # We will create k_folds dictionary structures
    folds = [{'train': {}, 'val': {}} for _ in range(k_folds)]
    
    for name in dataset_names:
        # Get data arrays
        t12 = data_dict[name]['delta_t12']
        t32 = data_dict[name]['delta_t32']  # may be None for mis4N2
        has_t32 = t32 is not None and len(t32) > 0
        
        # Split t12
        indices_12 = np.random.permutation(len(t12))
        fold_indices_12 = np.array_split(indices_12, k_folds)
        
        # Split t32 (only when available)
        if has_t32:
            indices_32 = np.random.permutation(len(t32))
            fold_indices_32 = np.array_split(indices_32, k_folds)
        else:
            fold_indices_32 = [[] for _ in range(k_folds)]
        
        for k in range(k_folds):
            # Validation indices for this fold
            val_idx_12 = fold_indices_12[k]
            val_idx_32 = fold_indices_32[k]
            
            # Training indices (concatenate all other folds)
            train_idx_12 = np.concatenate([fold_indices_12[j] for j in range(k_folds) if j != k])
            if has_t32:
                train_idx_32 = np.concatenate([fold_indices_32[j] for j in range(k_folds) if j != k])
            else:
                train_idx_32 = []
            
            # Assign to folds — preserve None for delta_t32 when not available
            folds[k]['val'][name] = {
                'delta_t12': t12[val_idx_12],
                'delta_t32': t32[val_idx_32] if has_t32 else None
            }
            
            folds[k]['train'][name] = {
                'delta_t12': t12[train_idx_12],
                'delta_t32': t32[train_idx_32] if has_t32 else None
            }
                
    return folds

def objective_function(params_vector, mechanism, train_data, n_simulations=2000):
    """
    Objective function for optimization (Minimize EMD on Train Data).
    """
    try:
        param_dict = parse_parameters(params_vector, mechanism)
        
        total_emd = 0
        dataset_names = train_data.keys()
        
        for name in dataset_names:
            # Apply mutant params (logic from simulation_utils/Analyze script)
            alpha    = param_dict.get('alpha',    1.0)
            beta_k   = param_dict.get('beta_k',   None)
            beta_tau  = param_dict.get('beta_tau',  None)
            beta_tau2 = param_dict.get('beta_tau2', None)
            gamma_N2  = param_dict.get('gamma_N2',  None)
            
            mutant_params, n0_list = apply_mutant_params(
                param_dict, name, alpha, beta_k, beta_tau, beta_tau2, gamma_N2
            )
                
            t12, t32 = run_simulation_for_dataset(mechanism, mutant_params, n0_list, num_simulations=n_simulations)
            
            if t12 is None:
                return 1e6
            
            exp_fold = train_data[name]
            exp_dict = {}
            sim_dict = {}
            if exp_fold['delta_t12'] is not None and len(exp_fold['delta_t12']) > 0:
                exp_dict['delta_t12'] = exp_fold['delta_t12']
                sim_dict['delta_t12'] = t12
            if exp_fold['delta_t32'] is not None and len(exp_fold['delta_t32']) > 0:
                exp_dict['delta_t32'] = exp_fold['delta_t32']
                sim_dict['delta_t32'] = t32
            
            if not exp_dict:
                continue  # no usable data for this dataset in this fold
            
            emd = calculate_emd(exp_dict, sim_dict)
            total_emd += emd
            
        return total_emd
        
    except Exception as e:
        return 1e6

def run_cross_validation(mechanism, k_folds=5, n_simulations=2000, max_iter=1000, tol=0.01):
    exp_data = load_experimental_data()
    folds = create_folds(exp_data, k_folds)
    
    bounds = get_parameter_bounds(mechanism)
    
    cv_results = []
    
    print(f"Starting {k_folds}-Fold Cross-Validation for {mechanism}...")
    
    for k in range(k_folds):
        print(f"\n--- Fold {k+1}/{k_folds} ---")
        
        train_data = folds[k]['train']
        val_data = folds[k]['val']
        
        # Optimize on Train
        print("Optimizing on Training Set...")
        
        # Differential Evolution settings (can tune speed vs accuracy)
        result = differential_evolution(
            objective_function,
            bounds,
            args=(mechanism, train_data, n_simulations),
            strategy='best1bin',
            maxiter=max_iter,     # Limited iterations for standard timeframe
            popsize=10,     # Smaller popsize for speed
            tol=tol,
            disp=False,
            workers=-1
        )
        
        best_params_vec = result.x
        train_emd = result.fun
        
        # Validate
        print("Validating...")
        val_emd = objective_function(best_params_vec, mechanism, val_data, n_simulations)
        
        print(f"Fold {k+1} Results: Train EMD={train_emd:.2f}, Val EMD={val_emd:.2f}")
        
        # Convert params to dictionary
        param_names = get_parameter_names(mechanism)
        param_dict = {name: float(val) for name, val in zip(param_names, best_params_vec)}
        param_json = json.dumps(param_dict)
        
        cv_results.append({
            'fold': k+1,
            'train_emd': train_emd,
            'val_emd': val_emd,
            'params': param_json
        })
        
    # Average
    avg_val_emd = np.mean([r['val_emd'] for r in cv_results])
    std_val_emd = np.std([r['val_emd'] for r in cv_results])
    
    print(f"\nCross-Validation Complete.")
    print(f"Average Validation EMD: {avg_val_emd:.2f} ± {std_val_emd:.2f}")
    
    # Save results
    df = pd.DataFrame(cv_results)
    df.to_csv(f'ModelComparisonEMDResults/cv_results_{mechanism}.csv', index=False)

if __name__ == "__main__":
    mechanism = 'time_varying_k' # Default to simple for next run
    n_simulations=10000
    max_iter=1000
    tol=0.01
    run_cross_validation(mechanism, n_simulations=n_simulations, max_iter=max_iter, tol=tol)
