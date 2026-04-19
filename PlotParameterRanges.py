#!/usr/bin/env python3
"""
PlotParameterRanges.py
-----------------------
Plot cross-validated parameter distributions as box-and-whisker plots.

Each CV results CSV contains a 'params' column with JSON-encoded parameter
dictionaries, one row per fold.  This script:
  - Loads one or more CSV files (each representing a model/mechanism variant)
  - Discovers the union of all parameter names across all files
  - Lays them out in a grid, one subplot per parameter
  - Draws one box-whisker per file in each subplot (blank if that file lacks
    the parameter)
  - Saves the figure as PDF and SVG

Usage
-----
Edit FILES_CONFIG at the bottom to point to your CSV files and give each
a short display label, then run:

    python PlotParameterRanges.py
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')


# ─── Pretty labels for known parameter names ─────────────────────────────────
PARAM_LABELS = {
    'n2':       r'$n_2$  (threshold, chr2)',
    'N2':       r'$N_2$  (initial count, chr2)',
    'k_max':    r'$k_\mathrm{max}$  (max cleavage rate)',
    'tau':      r'$\tau$  (ramp timescale, s)',
    'r21':      r'$r_{21}$  ($n_1 / n_2$)',
    'r23':      r'$r_{23}$  ($n_3 / n_2$)',
    'R21':      r'$R_{21}$  ($N_1 / N_2$)',
    'R23':      r'$R_{23}$  ($N_3 / N_2$)',
    'alpha':    r'$\alpha$  (threshold mutant)',
    'beta_k':   r'$\beta_k$  (separase mutant)',
    'beta_tau': r'$\beta_\tau$  (APC mutant)',
    'beta_tau2':r'$\beta_{\tau 2}$  (velcade mutant)',
    'gamma_N2': r'$\gamma_{N_2}$  (Mis4→chr2 scale)',
    'burst_size':r'burst size $b$',
    'n_inner':  r'$n_\mathrm{inner}$  (steric hindrance)',
}

# Preferred display order (any unknown params are appended alphabetically)
PARAM_ORDER = [
    'n2', 'N2', 'k_max', 'tau',
    'r21', 'r23', 'R21', 'R23',
    'alpha', 'beta_k', 'beta_tau', 'beta_tau2',
    'gamma_N2',
    'burst_size', 'n_inner',
]


# ─── Colour palette (one per file) ───────────────────────────────────────────
PALETTE = [
    '#4C72B0', '#DD8452', '#55A868', '#C44E52',
    '#8172B3', '#937860', '#DA8BC3', '#8C8C8C',
    '#CCB974', '#64B5CD',
]


def load_cv_params(filepath: str) -> pd.DataFrame:
    """
    Load a CV results CSV and return a DataFrame where each column is a
    parameter and each row is a fold.
    """
    df = pd.read_csv(filepath)
    records = [json.loads(p) for p in df['params']]
    return pd.DataFrame(records)


def plot_parameter_ranges(
    files_config: list[tuple[str, str]],
    ncols: int = 4,
    figsize_per_cell: tuple[float, float] = (3.0, 3.2),
    output_prefix: str = 'parameter_ranges',
):
    """
    Parameters
    ----------
    files_config : list of (filepath, label)
        Each entry is a CSV path and a short display label.
    ncols : int
        Number of columns in the subplot grid.
    figsize_per_cell : (w, h)
        Width and height of each individual subplot (inches).
    output_prefix : str
        Prefix for the saved PDF/SVG files.
    """
    # ── Load all files ────────────────────────────────────────────────────────
    loaded: list[tuple[str, pd.DataFrame]] = []
    for fpath, label in files_config:
        if not os.path.isfile(fpath):
            print(f"[WARNING] File not found, skipping: {fpath}")
            continue
        param_df = load_cv_params(fpath)
        loaded.append((label, param_df))
        print(f"Loaded '{label}': {len(param_df)} folds, "
              f"params: {list(param_df.columns)}")

    if not loaded:
        raise RuntimeError("No valid CSV files could be loaded.")

    # ── Determine parameter union and display order ───────────────────────────
    all_params = set()
    for _, pdf in loaded:
        all_params.update(pdf.columns)

    ordered = [p for p in PARAM_ORDER if p in all_params]
    ordered += sorted(all_params - set(ordered))   # append any unknowns
    n_params = len(ordered)

    # ── Figure layout ─────────────────────────────────────────────────────────
    nrows = (n_params + ncols - 1) // ncols
    fig_w = figsize_per_cell[0] * ncols
    fig_h = figsize_per_cell[1] * nrows

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(fig_w, fig_h),
                              constrained_layout=True)
    axes_flat = np.array(axes).flatten()

    n_models = len(loaded)
    positions = np.arange(1, n_models + 1)          # x positions for boxes
    colors    = PALETTE[:n_models]

    # ── Draw one subplot per parameter ───────────────────────────────────────
    for idx, param in enumerate(ordered):
        ax = axes_flat[idx]
        ax.set_title(PARAM_LABELS.get(param, param), fontsize=9, pad=4)

        boxes_drawn = 0
        for model_idx, (label, pdf) in enumerate(loaded):
            pos = positions[model_idx]
            color = colors[model_idx]

            if param not in pdf.columns:
                # Leave this position blank — draw a subtle marker to preserve
                # alignment without cluttering the plot
                ax.plot(pos, np.nan, 'o', color='none')
                continue

            values = pdf[param].dropna().values
            if len(values) == 0:
                ax.plot(pos, np.nan, 'o', color='none')
                continue

            bp = ax.boxplot(
                values,
                positions=[pos],
                widths=0.55,
                patch_artist=True,
                notch=False,
                showfliers=True,
                boxprops=dict(facecolor=color, alpha=0.75, linewidth=1.2),
                medianprops=dict(color='white', linewidth=2.0),
                whiskerprops=dict(color=color, linewidth=1.2),
                capprops=dict(color=color, linewidth=1.5),
                flierprops=dict(marker='o', markerfacecolor=color,
                                markersize=4, linestyle='none',
                                markeredgewidth=0.5, markeredgecolor='white'),
            )
            boxes_drawn += 1

        # x-axis: one tick per model, labelled with the short label
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [lbl for lbl, _ in loaded],
            fontsize=7, rotation=30, ha='right'
        )
        ax.set_xlim(0.3, n_models + 0.7)
        ax.tick_params(axis='y', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Light horizontal grid to help read values
        ax.yaxis.grid(True, linestyle='--', alpha=0.4, linewidth=0.6)
        ax.set_axisbelow(True)

    # ── Hide unused subplot cells ─────────────────────────────────────────────
    for idx in range(n_params, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(facecolor=colors[i], alpha=0.75, label=label)
        for i, (label, _) in enumerate(loaded)
    ]
    fig.legend(
        handles=legend_patches,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(n_models, 5),
        fontsize=9,
        frameon=False,
    )

    fig.suptitle('Cross-Validated Parameter Distributions', fontsize=13, y=1.01)

    # ── Save ──────────────────────────────────────────────────────────────────
    for fmt in ('pdf', 'svg'):
        out = f'{output_prefix}.{fmt}'
        fig.savefig(out, dpi=300, bbox_inches='tight', format=fmt)
        print(f"Saved: {out}")

    plt.show()


# ─── Configuration — edit here ───────────────────────────────────────────────
if __name__ == '__main__':

    BASE = 'ModelComparisonEMDResults'

    FILES_CONFIG = [
        # (filepath,                                                            short label)
        (f'{BASE}/cv_results_time_varying_k_normal_1.csv', 'Basic'),
        (f'{BASE}/cv_results_time_varying_k.csv', 'Basic with Mis4'),
    ]

    plot_parameter_ranges(
        files_config=FILES_CONFIG,
        ncols=4,                     # subplots per row
        figsize_per_cell=(3.0, 3.2), # (width, height) of each cell in inches
        output_prefix='parameter_ranges_comparison',
    )
