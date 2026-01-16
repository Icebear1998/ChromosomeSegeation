# Understanding NLL Differences vs. Simulation Count

Why does the Negative Log-Likelihood (NLL) change significantly (e.g., by 500+ points) between 500 and 5000 simulations, even when the KDE plots look almost identical?

## 1. Visual Deception of Linear Plots
Human eyes compare distributions on a **linear scale**. We look at the "bulk" of the distribution (the peak).
*   **Peak**: $P(x) \approx 0.05$ vs $0.051$ is invisible.
*   **Tail**: $P(x) \approx 0.0001$ vs $0.00001$ is also invisible (both look like "zero" on the line).

However, NLL operates on a **logarithmic scale**:
*   $\ln(0.0001) \approx -9.2$
*   $\ln(0.00001) \approx -11.5$
*   **Difference**: 2.3 points *per data point*.

If you have 100 data points in the tails, that tiny, invisible difference results in a **230 point difference** in total NLL.

## 2. Bandwidth Scaling (The "Sharpness" Factor)
Kernel Density Estimation (KDE) uses a **bandwidth** ($h$) to smooth the discrete simulation points into a continuous curve. We use **Scott's Rule**, which automatically scales bandwidth based on the number of data points ($N$):

$$ h \propto N^{-1/5} $$

*   **Low N (500)**: Large bandwidth. The distribution is "smoothed out".
    *   Peaks are shorter/flatter.
    *   Tails are wider/fatter (higher probability in tails).
*   **High N (5000)**: Small bandwidth. The distribution is "sharper".
    *   Peaks are taller.
    *   Tails drop off faster.

### The Impact on NLL
*   **If the Model fits well**: most experimental data points fall near the **peak**.
    *   High N (sharper) $\to$ Higher Density at peak $\to$ Better NLL (Lower value).
*   **If the Model fits poorly**: data points fall in the **tails**.
    *   High N (sharper) $\to$ Lower Density in tails $\to$ Worse NLL (Higher value).

This systematic change in "sharpness" causes a global shift in the absolute NLL values, even if the optimal parameters remain roughly the same.

## 3. Stability vs. Absolute Value
For optimization, we care about **relative NLL** (Param Set A vs Param Set B), not the absolute number.
*   It doesn't matter if the minimum is at 7000 or 7500.
*   It matters that **Set A is consistently better than Set B**.

**The Real Danger: Noise (Variance)**
If we run the simulation twice with N=500:
*   Run 1: NLL = 8000
*   Run 2: NLL = 8050
*   **Noise Floor**: $\pm 50$

If modifying a biological parameter only improves NLL by 20, **N=500 is not enough** because the signal (20) is lost in the noise (50).

**Conclusion**: We must measure the **Standard Deviation of the NLL** at N=500 to see if it's precise enough for our optimization needs.
