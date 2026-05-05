#!/usr/bin/env python3
"""
SweepSymmetricNnN.py
--------------------
Single-chromosome sensitivity analysis: how do E[T] and SD[T] of one
individual chromosome depend on its own n (threshold) and N (initial count)?

Runs a 2-D grid sweep over the full optimisation ranges:
    n ∈ [0, 50]    (threshold for cohesin cleavage)
    N ∈ [50, 1000] (initial cohesin count)

Produces:
  1. Two heatmaps: E[T](n, N) and SD[T](n, N)
  2. Multi-line plots: E[T] vs N for several fixed n values
  3. Multi-line plots: SD[T] vs n for several fixed N values
  4. CSV of the full grid data

Key insight expected:
  • SD[T] is governed mainly by n
  • E[T] is governed mainly by (N - n)
  • ΔT statistics can be derived analytically from individual T statistics

Usage
-----
Edit the CONFIGURATION block at the bottom, then run:
    python SweepSymmetricNnN.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')

from simulation_utils import (
    load_optimized_parameters,
    run_simulation_raw,
)


# ─── Single-chromosome simulation ────────────────────────────────────────────

def simulate_single_chromosome(mechanism, params, n_val, N_val,
                                num_simulations=5000, n_repeat=3):
    """
    Simulate a single chromosome with threshold n_val and initial count N_val.

    We set all three chromosomes to (n_val, N_val) and read only the first
    column.  Since chromosomes are independent in the simulation, this is
    equivalent to simulating a single chromosome.

    Returns
    -------
    mean_T, sd_T : floats   (averaged over n_repeat batches)
    """
    p = params.copy()
    p['n2'] = n_val
    p['N2'] = N_val
    n0_list = [p['n1'], n_val, p['n3']]

    if 'k_max' in p and 'tau' in p:
        p['k_1'] = p['k_max'] / p['tau']

    means, sds = [], []
    for _ in range(n_repeat):
        _, t2, _ = run_simulation_raw(mechanism, p, n0_list, num_simulations)
        if t2 is None:
            means.append(np.nan)
            sds.append(np.nan)
        else:
            means.append(np.mean(t2))
            sds.append(np.std(t2))

    return np.nanmean(means), np.nanmean(sds)


# ─── 2-D grid sweep ──────────────────────────────────────────────────────────

def run_grid_sweep(mechanism, params, n_values, N_values,
                   num_simulations=5000, n_repeat=3):
    """
    Sweep over a 2-D grid of (n, N) values for a single chromosome.

    Returns
    -------
    mean_grid : 2-D array of shape (len(N_values), len(n_values))
    sd_grid   : same shape
    """
    n_N = len(N_values)
    n_n = len(n_values)
    total = n_N * n_n

    mean_grid = np.full((n_N, n_n), np.nan)
    sd_grid   = np.full((n_N, n_n), np.nan)

    count = 0
    for i, N_val in enumerate(N_values):
        for j, n_val in enumerate(n_values):
            count += 1
            # Skip if n >= N (no cohesins to cleave past threshold)
            if n_val >= N_val:
                continue

            m, s = simulate_single_chromosome(
                mechanism, params, n_val, N_val,
                num_simulations, n_repeat
            )
            mean_grid[i, j] = m
            sd_grid[i, j]   = s

            if count % 20 == 0 or count == total:
                print(f'  [{count:>5}/{total}]  n={n_val:>6.1f}  N={N_val:>7.1f}  '
                      f'E[T]={m:>7.1f}  SD[T]={s:>6.1f}')

    return mean_grid, sd_grid


# ─── Heatmap plotting ────────────────────────────────────────────────────────

def plot_heatmaps(n_values, N_values, mean_grid, sd_grid, mechanism, output_dir):
    """
    Two side-by-side heatmaps:
      Left  — E[T](n, N)
      Right — SD[T](n, N)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                    constrained_layout=True)
    fig.suptitle(
        f'Single-chromosome sensitivity  —  {mechanism}',
        fontsize=13, fontweight='bold'
    )

    # ── E[T] heatmap ──────────────────────────────────────────────────────────
    im1 = ax1.imshow(
        mean_grid, origin='lower', aspect='auto',
        extent=[n_values[0], n_values[-1], N_values[0], N_values[-1]],
        cmap='viridis',
    )
    cb1 = fig.colorbar(im1, ax=ax1, shrink=0.85, pad=0.02)
    cb1.set_label('E[T] (sec)', fontsize=10)
    ax1.set_xlabel(r'Threshold $n$', fontsize=11)
    ax1.set_ylabel(r'Initial count $N$', fontsize=11)
    ax1.set_title(r'Mean separation time  $E[T]$', fontsize=11, fontweight='bold')

    # Add iso-(N-n) contour lines to show E[T] depends on N-n
    nn, NN = np.meshgrid(n_values, N_values)
    diff = NN - nn
    # Only draw where n < N
    masked_diff = np.where(nn < NN, diff, np.nan)
    contours = ax1.contour(nn, NN, masked_diff, levels=8,
                           colors='white', linewidths=0.8, linestyles='--', alpha=0.6)
    ax1.clabel(contours, inline=True, fontsize=7, fmt='N−n=%.0f')

    # ── SD[T] heatmap ─────────────────────────────────────────────────────────
    im2 = ax2.imshow(
        sd_grid, origin='lower', aspect='auto',
        extent=[n_values[0], n_values[-1], N_values[0], N_values[-1]],
        cmap='viridis',
    )
    cb2 = fig.colorbar(im2, ax=ax2, shrink=0.85, pad=0.02)
    cb2.set_label('SD[T] (sec)', fontsize=10)
    ax2.set_xlabel(r'Threshold $n$', fontsize=11)
    ax2.set_ylabel(r'Initial count $N$', fontsize=11)
    ax2.set_title(r'Variability  $\mathrm{SD}[T]$', fontsize=11, fontweight='bold')

    for fmt in ('pdf', 'svg'):
        fpath = os.path.join(output_dir, f'heatmap_single_chr_{mechanism}.{fmt}')
        fig.savefig(fpath, dpi=300, bbox_inches='tight', format=fmt)
        print(f'  Saved: {fpath}')
    plt.show()
    plt.close(fig)


# ─── Multi-line plots ────────────────────────────────────────────────────────

def plot_mean_vs_N_for_fixed_n(n_values, N_values, mean_grid, mechanism, output_dir,
                                n_slices=None):
    """
    Plot E[T] vs N, with one line per fixed n value.
    Demonstrates that E[T] is mainly determined by (N − n).
    """
    if n_slices is None:
        # Pick ~6 representative n values spanning the range
        idx = np.linspace(0, len(n_values) - 1, 6, dtype=int)
        n_slices = n_values[idx]

    cmap = plt.cm.coolwarm
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    for k, n_val in enumerate(n_slices):
        j = np.argmin(np.abs(n_values - n_val))
        y = mean_grid[:, j]
        valid = np.isfinite(y)
        colour = cmap(k / max(len(n_slices) - 1, 1))
        ax.plot(N_values[valid], y[valid], lw=2.2, color=colour,
                label=rf'$n = {n_val:.1f}$')

    ax.set_xlabel(r'Initial count $N$', fontsize=11)
    ax.set_ylabel(r'$E[T]$  (sec)', fontsize=11)
    ax.set_title(rf'$E[T]$ vs $N$ for fixed $n$  —  {mechanism}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9, title=r'Threshold $n$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, ls='--', alpha=0.3, lw=0.7)
    ax.set_axisbelow(True)

    for fmt in ('pdf', 'svg'):
        fpath = os.path.join(output_dir, f'meanT_vs_N_fixed_n_{mechanism}.{fmt}')
        fig.savefig(fpath, dpi=300, bbox_inches='tight', format=fmt)
        print(f'  Saved: {fpath}')
    plt.show()
    plt.close(fig)


def plot_sd_vs_n_for_fixed_N(n_values, N_values, sd_grid, mechanism, output_dir,
                              N_slices=None):
    """
    Plot SD[T] vs n, with one line per fixed N value.
    Demonstrates that SD[T] is mainly determined by n (lines should roughly
    collapse onto each other).
    """
    if N_slices is None:
        idx = np.linspace(0, len(N_values) - 1, 6, dtype=int)
        N_slices = N_values[idx]

    cmap = plt.cm.cool
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    for k, N_val in enumerate(N_slices):
        i = np.argmin(np.abs(N_values - N_val))
        y = sd_grid[i, :]
        valid = np.isfinite(y)
        colour = cmap(k / max(len(N_slices) - 1, 1))
        ax.plot(n_values[valid], y[valid], lw=2.2, color=colour,
                label=rf'$N = {N_val:.0f}$')

    ax.set_xlabel(r'Threshold $n$', fontsize=11)
    ax.set_ylabel(r'$\mathrm{SD}[T]$  (sec)', fontsize=11)
    ax.set_title(rf'$\mathrm{{SD}}[T]$ vs $n$ for fixed $N$  —  {mechanism}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9, title=r'Initial count $N$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, ls='--', alpha=0.3, lw=0.7)
    ax.set_axisbelow(True)

    for fmt in ('pdf', 'svg'):
        fpath = os.path.join(output_dir, f'sdT_vs_n_fixed_N_{mechanism}.{fmt}')
        fig.savefig(fpath, dpi=300, bbox_inches='tight', format=fmt)
        print(f'  Saved: {fpath}')
    plt.show()
    plt.close(fig)





# ─── Main entry point ────────────────────────────────────────────────────────

def run_single_chromosome_analysis(
    mechanism       = 'time_varying_k',
    param_file      = None,
    n_grid_points   = 25,      # grid resolution for n
    N_grid_points   = 25,      # grid resolution for N
    n_range         = (0.5, 50.0),
    N_range         = (50.0, 1000.0),
    num_simulations = 5000,
    n_repeat        = 3,
    output_dir      = 'SweepAnalysis',
):
    """
    Single-chromosome 2-D sensitivity analysis.

    Parameters
    ----------
    mechanism       : model mechanism string
    param_file      : path to optimised parameter .txt file
    n_grid_points   : number of n values in the grid
    N_grid_points   : number of N values in the grid
    n_range         : (min, max) for threshold n
    N_range         : (min, max) for initial count N
    num_simulations : simulations per (n, N) evaluation
    n_repeat        : repeats per grid point (for averaging)
    output_dir      : directory for CSV and PDF/SVG outputs
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load parameters (for k_max, tau, etc. — kept fixed) ────────────────
    if param_file is None:
        param_file = (
            f'ParameterFiles/simulation_optimized_parameters_{mechanism}.txt'
        )
    print(f'Loading parameters from: {param_file}')
    params = load_optimized_parameters(mechanism, param_file)
    if not params:
        raise RuntimeError(f'Could not load parameters from {param_file}')

    if 'k_1' not in params and 'k_max' in params and 'tau' in params:
        params['k_1'] = params['k_max'] / params['tau']

    print(f'\nOptimised values (for reference):')
    print(f'  n1={params["n1"]:.3f}  n2={params["n2"]:.3f}  n3={params["n3"]:.3f}')
    print(f'  N1={params["N1"]:.1f}  N2={params["N2"]:.1f}  N3={params["N3"]:.1f}')

    # ── 2. Build grid ─────────────────────────────────────────────────────────
    n_values = np.linspace(n_range[0], n_range[1], n_grid_points)
    N_values = np.linspace(N_range[0], N_range[1], N_grid_points)

    total = n_grid_points * N_grid_points
    print(f'\n{"="*60}')
    print(f'2-D grid:  n ∈ [{n_range[0]:.1f}, {n_range[1]:.1f}]  ({n_grid_points} pts)')
    print(f'           N ∈ [{N_range[0]:.0f}, {N_range[1]:.0f}]  ({N_grid_points} pts)')
    print(f'           Total grid points: {total}')
    print(f'           Sims per point: {num_simulations} × {n_repeat} repeats')
    print(f'{"="*60}\n')

    mean_grid, sd_grid = run_grid_sweep(
        mechanism, params, n_values, N_values,
        num_simulations, n_repeat
    )

    # ── 3. Save CSV ───────────────────────────────────────────────────────────
    records = []
    for i, N_val in enumerate(N_values):
        for j, n_val in enumerate(n_values):
            records.append({
                'n': n_val,
                'N': N_val,
                'N_minus_n': N_val - n_val,
                'mean_T': mean_grid[i, j],
                'sd_T':   sd_grid[i, j],
            })
    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, f'single_chr_grid_{mechanism}.csv')
    df.to_csv(csv_path, index=False)
    print(f'\n  Grid data saved: {csv_path}')

    # ── 4. Plots ──────────────────────────────────────────────────────────────
    print('\n--- Generating heatmaps ---')
    plot_heatmaps(n_values, N_values, mean_grid, sd_grid, mechanism, output_dir)

    print('\n--- Generating E[T] vs N for fixed n ---')
    plot_mean_vs_N_for_fixed_n(n_values, N_values, mean_grid, mechanism, output_dir)

    print('\n--- Generating SD[T] vs n for fixed N ---')
    plot_sd_vs_n_for_fixed_N(n_values, N_values, sd_grid, mechanism, output_dir)

    print('\nAll done.')


# ─── CONFIGURATION — edit here then run ──────────────────────────────────────
if __name__ == '__main__':
    run_single_chromosome_analysis(
        mechanism       = 'time_varying_k',
        param_file      = 'ParameterFiles/simulation_optimized_parameters_time_varying_k_1.txt',
        n_grid_points   = 20,       # grid resolution (n axis)
        N_grid_points   = 20,       # grid resolution (N axis)
        n_range         = (0.5, 40.0),
        N_range         = (50.0, 1000.0),
        num_simulations = 5000,
        n_repeat        = 3,
        output_dir      = 'SweepAnalysis',
    )
