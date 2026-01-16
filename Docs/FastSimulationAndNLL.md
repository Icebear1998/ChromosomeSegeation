Technical Summary: Simulation Methods & Likelihood Analysis
1. Fast Simulation Algorithms
Traditional stochastic simulation (Gillespie/SSA) scales linearly with molecule count ($O(M)$), making optimization slow for systems with high copy numbers. We implemented two Fast Simulation methods that achieve $O(1)$ complexity relative to molecule count while maintaining statistical exactness.

A. Beta Sampling Method (FastBetaSimulation)
Use Case: Linear degradation/segregation without feedback.
Logic: For independent particles degrading with rate $k$, the number of particles remaining after time $\Delta t$ follows a Binomial distribution. For large numbers, this is approximated by a Normal distribution, or Beta distribution for ratios.
Performance: ~100-1000x faster than SSA for high $N$. Vectorized using NumPy.
B. Sum of Waiting Times (FastFeedbackSimulation)
Use Case: Feedback-based segregation (e.g., "Onion" model).
Logic: Instead of simulating every reaction event, we compute the waiting time for the $N$-th event directly. For independent processes, the time for $k$ events is a sum of Exponential variables (Gamma distribution).
Logic Extension: For feedback loops where rates change, we simulate the "time chunks" between rate changes analytically.
2. Likelihood Estimation (KDE)
We compute the Negative Log-Likelihood (NLL) of experimental data $x_{exp}$ given simulated data $x_{sim}$ using Kernel Density Estimation (KDE).

$$NLL = -\sum_{i} \log(\hat{f}{KDE}(x{i, exp}))$$

Bandwidth Selection
Scott's Rule (Adaptive): $h = \sigma \cdot N_{sim}^{-1/5}$.
Pros: Adapts to sample size. As $N$ increases, $h$ decreases, allowing finer resolution of the distribution.
Cons: NLL values change systematically with $N$ as the smoothing changes.
Fixed Bandwidth: $h = \text{const}$.
Pros: Consistent smoothing across different $N$.
Cons: May over-smooth large datasets or under-smooth small ones.
Robust KDE
To prevent infinite NLLs when experimental data falls outside the simulated range, we use a mixture model: $$P(x) = (1-\epsilon)\text{KDE}(x) + \epsilon P_{background}$$ where $\epsilon=10^{-3}$ and $P_{background}=10^{-6}$.

3. NLL Convergence & Statistical Bias
Users typically observe that Mean NLL decreases gradually as Simulation Count ($N$) increases, eventually plateauing.

Why NLL Decreases (Jensen's Inequality)
The NLL estimator has a positive bias at low sample sizes. Since $NLL = -\log(\hat{f})$, and $-\log$ is convex, Jensen's Inequality implies: $$E[-\log(\hat{f})] > -\log(E[\hat{f}])$$

Low N: High variance in $\hat{f}$. The "penalty" from variance pushes Expected NLL significantly higher than the true NLL.
High N: Variance decreases ($\propto 1/N$). The bias diminishes, and NLL converges downward to the true theoretical value.
Tail Coverage
Experimental outliers (tails) are critical. At low $N$, tails are sparsely sampled, leading to $\hat{f} \approx 0$ and massive NLL spikes. High $N$ ensures non-zero density in tails, stabilizing the NLL.

4. Optimization Best Practices
Selection Bias ("Optimistic Bias")
During differential evolution, the algorithm evaluates thousands of parameter sets. Due to stochastic noise, the "best" NLL reported by the optimizer is often a "lucky outlier"—a run where variance swung in favor of low NLL (e.g., 8 std devs below mean).

Observation: Optimizer reports NLL=7047. Re-running the same parameters gives NLL=7098.
Solution: Never trust the single NLL value from the optimization log. Always re-evaluate the optimized parameters with repeats (e.g., 50-100 trials) to find the true mean NLL.
Recommendation
For model comparison, ensure:

Consistent N: Compare models using the same number of simulations.
High N: Use $N \ge 100,000$ to minimize variance bias.
Averaging: Report the mean NLL over multiple replicates.