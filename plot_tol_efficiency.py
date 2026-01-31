"""
Plot tolerance efficiency analysis results for time_varying_k mechanism.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('tol_efficiency_summary_time_varying_k_20260121_231636.csv')

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Mean Train and Validation EMD
ax1.plot(df['Tolerance'], df['Mean Train EMD'], 'o-', label='Mean Train EMD', linewidth=2, markersize=8)
ax1.plot(df['Tolerance'], df['Mean Val EMD'], 's-', label='Mean Val EMD', linewidth=2, markersize=8)
ax1.set_xlabel('Tolerance', fontsize=12)
ax1.set_ylabel('EMD', fontsize=12)
ax1.set_title('Mean Train and Validation EMD vs Tolerance', fontsize=13, fontweight='bold')
ax1.set_xscale('log')
ax1.legend(fontsize=11)
ax1.invert_xaxis()  # Higher tolerance (less strict) on the left
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Plot 2: Standard Deviation of Validation EMD
ax2.plot(df['Tolerance'], df['Std Val EMD'], 'o-', color='C2', linewidth=2, markersize=8)
ax2.set_xlabel('Tolerance', fontsize=12)
ax2.set_ylabel('Std Val EMD', fontsize=12)
ax2.set_title('Validation EMD Standard Deviation vs Tolerance', fontsize=13, fontweight='bold')
ax2.set_xscale('log')
ax2.invert_xaxis()  # Higher tolerance (less strict) on the left
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('tol_efficiency_emd_plots.pdf', dpi=300, bbox_inches='tight')
plt.savefig('tol_efficiency_emd_plots.svg', dpi=300, bbox_inches='tight')
print("Plot saved as 'tol_efficiency_emd_plots.pdf'")
plt.show()
