#!/usr/bin/env python3
"""
Simulation KDE Module - Simplified Version

This module provides simple KDE functionality for simulation data.
Uses scipy.stats.gaussian_kde for straightforward kernel density estimation.

Usage:
    from simulation_kde import build_kde_from_simulations, evaluate_kde_pdf
    
    # Build KDE from simulation data
    kde = build_kde_from_simulations(sim_data, bandwidth=10.0)
    
    # Evaluate on grid
    x_grid = np.linspace(-50, 50, 500)
    pdf_values = evaluate_kde_pdf(kde, x_grid)
"""

import numpy as np
from scipy.stats import gaussian_kde


def build_kde_from_simulations(simulation_data, bandwidth=None):
    """
    Build a KDE from simulation data using scipy's gaussian_kde.
    
    Args:
        simulation_data: 1D array of simulation results
        bandwidth: Bandwidth parameter (optional)
            - If None: uses Scott's rule (default)
            - If float: uses custom bandwidth value
    
    Returns:
        KDE object from scipy.stats.gaussian_kde
    """
    # Clean data
    simulation_data = np.asarray(simulation_data).flatten()
    simulation_data = simulation_data[np.isfinite(simulation_data)]
    
    if len(simulation_data) < 10:
        raise ValueError(f"Not enough data: {len(simulation_data)} points (need at least 10)")
    
    # Build KDE
    if bandwidth is None:
        # Use Scott's rule (default)
        kde = gaussian_kde(simulation_data)
    else:
        # Use custom bandwidth
        # scipy's bw_method: we need to convert our bandwidth to scipy's format
        # scipy internally computes: h = bw_method * data.std() * n^(-1/5)
        # We want: h = bandwidth
        # So: bw_method = bandwidth / (data.std() * n^(-1/5))
        data_std = np.std(simulation_data, ddof=1)
        n = len(simulation_data)
        scott_factor = n ** (-1/5)
        
        if data_std > 0:
            bw_method = bandwidth / (data_std * scott_factor)
        else:
            bw_method = 1.0
        
        kde = gaussian_kde(simulation_data, bw_method=bw_method)
    
    return kde


def evaluate_kde_pdf(kde, x_values):
    """
    Evaluate KDE at given points.
    
    Args:
        kde: KDE object from build_kde_from_simulations
        x_values: Points at which to evaluate the PDF
    
    Returns:
        Array of PDF values
    """
    x_values = np.asarray(x_values).flatten()
    return kde(x_values)


def calculate_kde_likelihood(kde, data):
    """
    Calculate negative log-likelihood of data under the KDE.
    
    Args:
        kde: KDE object
        data: Data points to evaluate
    
    Returns:
        Negative log-likelihood (NLL)
    """
    data = np.asarray(data).flatten()
    data = data[np.isfinite(data)]
    
    if len(data) == 0:
        return np.inf
    
    # Get PDF values
    pdf_values = kde(data)
    
    # Avoid log(0)
    pdf_values = np.maximum(pdf_values, 1e-300)
    
    # Calculate NLL
    nll = -np.sum(np.log(pdf_values))
    
    return nll


def quick_kde_pdf(simulation_data, x_grid, bandwidth=10.0):
    """
    Quick one-line function to get KDE PDF.
    
    Args:
        simulation_data: Simulation results
        x_grid: Points to evaluate PDF
        bandwidth: Bandwidth parameter (default: 10.0)
    
    Returns:
        PDF values on x_grid
    """
    kde = build_kde_from_simulations(simulation_data, bandwidth=bandwidth)
    return evaluate_kde_pdf(kde, x_grid)


if __name__ == "__main__":
    # Simple test
    print("=" * 60)
    print("Simulation KDE Module - Simple Test")
    print("=" * 60)
    
    # Generate test data
    np.random.seed(42)
    test_data = np.random.normal(0, 10, 1000)
    
    print(f"\nTest data: {len(test_data)} points")
    print(f"  Mean: {np.mean(test_data):.2f}")
    print(f"  Std: {np.std(test_data):.2f}")
    
    # Build KDE
    kde = build_kde_from_simulations(test_data, bandwidth=5.0)
    print("\n✓ KDE built successfully")
    
    # Evaluate on grid
    x_grid = np.linspace(-30, 30, 500)
    pdf_values = evaluate_kde_pdf(kde, x_grid)
    
    print(f"\n✓ PDF evaluated on {len(x_grid)} points")
    print(f"  PDF max: {np.max(pdf_values):.6f}")
    print(f"  PDF integral: {np.trapz(pdf_values, x_grid):.3f} (should be ~1.0)")
    
    # Calculate self-likelihood
    nll = calculate_kde_likelihood(kde, test_data)
    print(f"\n✓ NLL calculated: {nll:.2f}")
    
    print("\n" + "=" * 60)
    print("Module ready!")
    print("=" * 60)
