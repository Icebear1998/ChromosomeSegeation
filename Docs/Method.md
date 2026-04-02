## Section 2.2: Method {#section-2.2:-method}

### **2.2.1  Stochastic models for cohesin cleavage** {#2.2.1-stochastic-models-for-cohesin-cleavage}

To understand the experimentally observed distributions of chromosome separation times, we modeled the stochastic cleavage of cohesin complexes that hold sister chromatids together. The model assumes that the *i*\-th chromosome begins with an initial number of cohesin complexes, Ni, and its sister chromatids separate once its cohesin count decays to a threshold ni. Cohesin cleavage is modeled as an inhomogeneous Poisson process. The rate constant of cohesin cleavage is assumed to be identical across all chromosomes and increases gradually over a timeframe , reflecting the gradual release and/or activation of separase in the cell (Eq. 5).

kt={kmaxt, 0≤t≤τ, kmax, t\>τ.  5 

The primary output of the model is the distribution of two pairwise separation-time differences, T12=T1-T2 and T32=T3-T2 where Ti denotes the separation time for chromosome *i*. 

The perturbed cases are assumed to undergo the following parameter changes:

***MBC treatment***: Destabilization of microtubules by MBC is expected to cause a decrease in the threshold number of cohesins that are required to hold sister chromatids together. Since this effect applies to all three chromosomes, a common multiplicative factor, \<1, is applied to all three threshold values, n1,n2,n3. 

***Separase mutation**:* Mutated separase is expected to have lower catalytic activity, corresponding to a lower kmax. To represent this effect, a multiplicative factor, k\<1, is applied to kmax.

***APC/C mutation**:* Mutated APC/C is expected to degrade securin more slowly and elongate the ramp phase for separase activity. To represent this effect, a multiplicative factor, \>1, is applied to 𝜏.

***Velcade treatment**:* Velcade has a similar effect to the APC/C mutation. To represent this effect, a multiplicative factor, 2\>1, is applied to 𝜏.

Additionally, we implemented two mechanistic variants that encode different hypotheses about cohesin cleavage:

***Processive separase action***: This model variant assumes processive cohesin cleavage with a fixed burst size b (number of cohesins cleaved and removed within one event).

***Steric hindrance*****:** This model variant considers steric hindrance by packed chromosomes that lowers accessibility of separase molecules to cohesins in the interior of chromosomes. For simplicity, we assume that cohesin complexes are distributed in a sphere and that only those at the surface are accessible. As outer cohesin complexes get cleaved, inner complexes are exposed. Under these assumptions, the effective cleavage rate scales with the surface-to-volume ratio of the region occupied by the remaining cohesin. Assuming a uniform density of cohesin in the sphere, the volume of the sphere scales with the number of remaining cohesin complexes, i.e., V\~N. Consequently, the surface-to-volume ratio scales as AV\~V-13\~N-13. Therefore, in this model variant, we assume (Eq. 6)

keffN,t={kt, N≤ninner, ktNninner-13, N\>ninner,  6 

where ninner is the number of cohesins accommodated by the innermost core, a region with no more steric hindrance effect.

Additionally, to evaluate if positive feedback causes a sharp increase in separase activity (e.g., Fig 4B), we further considered a variation for the three models above, in which the time duration for the rate increase, , is constrained at low values (τ\<5 sec). 

The six models resulting from the combinations above are summarized in **Table 1**. They were each fitted to the experimental data (wild type and perturbations). The bounds for parameter fitting are listed in **Table 2**.

# **Table 1**

|  |  | Parameters | Separase activity ramp |
| :---: | :---: | :---: | :---: |
| **Basic model** |  | N2, n2, R12, R32, r12, r32, kmax, τ Perturbation conditions: α, βk, βτ, βτ2  | slow, 2\<τ\<240 |
|  |  |  | fast, 0.5\<τ\<5 |
| **Processive separase action** | b cohesin molecules degraded within one degradation event | Same as basic model,  and b | slow, 2\<τ\<240 |
|  |  |  | fast, 0.5\<τ\<5 |
| **Steric Hindrance** | effective k scales with surface-to-volume ratio (Eq. S2) | Same as basic model,  and ninner | slow, 2\<τ\<240 |
|  |  |  | fast, 0.5\<τ\<5 |

1. ## **2\. Stochastic simulation algorithms**

   2. ### **2.1 Gillespie stochastic simulation**

To simulate cohesin cleavage, we implemented a modified Gillespie algorithm. Each simulation tracks the number of cohesin complexes on each of the three chromosomes and records the time when each chromosome reaches its corresponding cohesin count threshold for separation. 

To accommodate the time-dependent degradation rate constant (Eq. 5), we modified the formula for sampling the next-reaction time to 

T=Tnext-Tprev=-2τlnrkmaxNtotal+Tprev2-Tprev, \#7 

where Tnext is the next degradation time to be sampled, Tprev is the previous degradation time, Ntotal is the sum of number of cohesin complexes currently remaining on each chromosome, and r is a uniform random number between 0 and 1\. After sampling for the next cleavage time, the cohesin cleavage is randomly chosen to happen to the *i*\-th chromosome with probability NiNtotal. For the processive separase action model, every cleavage event removes *b* cohesin complexes on the chosen chromosome. For the steric hindrance model, Ntotal is modified to the sum of the currently accessible number of cohesin complexes on each chromosome, where the accessible number is, Ni,access=NiNininner-13 for Ni\>ninner and Ni,access=Ni for Nininner; chromosome selection probability is also modified to Ni, accessNtotal.

Note that Eq. 7 gives the inverse-CDF sampling for the next-reaction time with CDF, PT\<t\=1 \- exp(-kmaxNcurrent (Tprev+t)² \-Tprev2/2τ), which can be derived from Eq. 5 through PT\<t=1 \- exp\-TprevTprev+tNcurrentkt'dt'. Beyond the ramp phase, the regular Gillespie algorithm (Gillespie, 1976, 1977\) based on rate constant kmax was used.

The Gillespie simulation was used to generate sample time trajectories (Fig. 5F) and to benchmark the fast simulation method described below.

3. ### **2.2 Fast simulation**

To enable efficient parameter optimization and large-scale validation, we adopted alternative exact simulation methods, using either order statistics (models without steric hindrance) or vectorized sum of waiting times approach (model with steric hindrance). These methods are statistically exact, alike the Gillespie algorithm, but offer considerable computational speedups (100x-1000x), allowing scalable model fitting.

*Accelerated sampling via order statistics:* For the basic model, the cleavage of individual cohesins are independent events following an identical waiting time distribution with the following CDF,

PTwait\<t={1-exp-kmaxt22τ, 0≤t≤τ, 1-exp-kmax2-kmaxt-τ, t\>τ.  \#8 

The separation time for the *i*\-th chromosome is the (ni+1)-th largest cohesin cleavage time out of a total of Ni cleavage times that follow the distribution in Eq. 8. As each cleavage time can be converted from a uniform random number through CDF inverse, which is a monotonically increasing function, the (ni+1)-th largest cleavage time is the CDF inverse of the (ni+1)-th largest value of Ni uniform random numbers. The latter is known to follow the beta distribution Beta(Ni-ni,ni+1). Therefore, by directly drawing one random number from this beta distribution and converting it through the inverse of Eq. 8, we obtain a sample separation time of a chromosome with a highly efficient *O*(1) operation.

The accelerated sampling method applies to the processive separase action model, too, when cohesin complexes are treated as predefined groups. A single cleavage event eliminates one group of b cohesins. To utilize the order statistics framework, we transform the parameters as effective counts:

Ni=⌈Nib⌉,ni=Ni-⌈Ni-nib⌉. \#9 

The process then proceeds as the non-processive case with Ni and ni. 

*Vectorized simulation:* For the steric hindrance model, the effective rate constant depends on the system's current state (Eq. 6). Therefore, cleavage of individual cohesin complexes are no longer independent of each other, making the fast-sampling method invalid. To accelerate the simulation, we take advantage of the convenient feature that there is only one reaction in the system and employ vectorization across simulations. Instead of simulating one trajectory at a time, we simultaneously simulate *M* independent cohesin loss trajectories. For each step from state *N* to *N*−1, we use Eq. 7 to determine the next event times across all *M* simulations. This approach leverages efficient array operations to achieve a significant speedup.​

4. ## **3\. Parameter estimation and model selection**

*Objective metric — Earth Mover’s Distance (EMD).*

For each candidate parameter set, the simulation methods described above generated a batch of *M* \= 10,000 independent samples of *T12* and *T32*, respectively, for each experimental scenario. Model fit was assessed using the Earth Mover's Distance (EMD; Wasserstein-1 distance) (Bazán et al., 2019; Rubner et al., 1998), a robust goodness-of-fit metric that quantifies the similarity between the empirical distribution of experimental data and that of the simulated data.

The objective function was defined as the summed EMD values across five experiments (control, MBC, separase mutant, APC/C mutant, velcade), which is also termed aggregate EMD. Sample size of 10,000 was confirmed to generate sufficiently precise estimate of EMD (Fig. S8A).

*Optimizer — Differential evolution (global) \+ local refinement.*

Model parameters were estimated using a two-stage optimization protocol. First, a global search was performed using differential evolution, an efficient population-based stochastic algorithm that explores the parameter space (Storn and Price, 1997). The algorithm was implemented using the scipy.optimize.differential\_evolution() function in the SciPy library. We used the 'best1bin' strategy with a population size of 10 (corresponding to 10x the number of parameters in each model variant), mutation factor (0.5, 1.0), recombination constant 0.7, and a relative convergence tolerance of 0.01. The top-performing candidate solutions were then refined locally using the L-BFGS-B algorithm (Byrd et al., 1995\) to maximize fit quality. Choices of population size and convergence tolerance were informed by benchmark results shown in Fig. S8B,C.

*Cross-validation.*

To reduce bias and avoid overfitting, 5-fold cross-validation was performed on the experimental dataset. Data were split into five stratified folds; for each fold, optimization was conducted on 80 % of the data, and EMD was computed on the held-out 20 %. Model comparison between mechanisms in Fig. 5C was performed using the aggregate cross-validated EMD metric (mean ± SE across folds).

5. ## **4\. OAT parameter sensitivity analysis**

To evaluate the impact of each parameter on separation synchrony, we performed a One-At-a-Time (OAT) sensitivity analysis. Each parameter was perturbed across the range of this parameter while holding all others fixed. For each parameter set we ran 10 batches of 10,000 separation time difference (∆t)  simulations. 

6. ## **5\. Algorithm implementation**

All simulations and optimization procedures were implemented in Python using standard scientific libraries. Computationally intensive parameter fitting and cross-validation were performed in parallel on a high-performance computing cluster. Parameter estimates are reported from best-fit solutions with uncertainties derived from cross-validated ensembles and model performance was summarized by the mean cross-validated aggregate EMD. All analyses are reproducible using the archived code and parameter settings. 

![][image3]

**Figure S8. Benchmarks and selection of key stochastic simulation and optimization hyperparameters. (A)** EMD estimate is unbiased, exhibiting a stable mean across sample sizes, and its precision (standard error) stabilizes beyond a sample size of 10,000. The optimal parameter set fitted to the complete experimental dataset was used, yielding lower EMD values than the five-fold cross-validation EMDs shown in (B) and (C). **(B)** Varying the population size in differential evolution has no significant effect on the optimization results. We therefore used the default p=10 in this study. Five-fold cross-validation EMD values are shown. **(C)** Mean and standard error of five-fold cross-validation EMD are stable across the tested range of relative tolerances (upper panel). However, because stochastic simulations are limited by finite sampling, convergence is constrained by simulation sample size. Consequently, stringent relative tolerances (\< 0.01) trigger spurious non-convergence warnings and incur unnecessary computational cost (lower panel; maximum iteration limit reached for the two smallest tolerances) after the practical convergence limit has already been achieved. We therefore selected a relative tolerance of 0.01. Error bars represent mean ± SE. Red circles indicate the hyperparameter values chosen in this study.

# **Table 2**

| Symbol | Description | Unit | Parameter Bounds | Parameter Justification |
| :---: | :---: | :---: | :---: | :---: |
| N2 | Initial cohesin count for the reference chromosome (chromosome 2\) | Count | \[50 \- 1000\] | Range was set broadly to include (i) cohesin-binding patterns in fission yeast (Schmidt et al., 2009; Mizuguchi et al., 2014), (ii) absolute protein abundance showing cohesin subunits exist at 102\-103 \+ molecules per cell (Marguerat et al., 2012), and (iii) additional quantification showing large numbers of cohesin complex complexes in other cell types, both cohesive and non-cohesive (Holzmann et al., 2019), and (iv) the fact that not all cohesin contributes to cohesion (Gerlich et al., 2006; Feytout et al., 2011; Tomonaga et al., 2000)  |
| n2 | Cohesin threshold count for the reference chromosome (chromosome 2\) | Count | \[0 \- 50\] | Low number regime reflects that cohesion may persist with few cohesin complexes, consistent with evidence that only a small pool of cohesive cohesin needs to be removed at anaphase (Tomonaga et al., 2000). Upper bound allows thresholds that reflect (i) estimates of spindle pulling forces (Gay et al., 2012; Grishchuk et al., 2005; Suzuki et al., 2016; Fisher et al., 2009; Chacón et al., 2014), (ii) the number of microtubules attached to fission yeast kinetochores (Ding et al., 1993; Joglekar et al., 2008), (iii) the force required to mechanically break cohesin (Richeldi et al., 2024; Pobegalov et al., 2023), (iv) and potential impacts of cohesion fatigue in response to spindle forces (Sapkota et al., 2018; Daum et al., 2011) . |
| R12​ | Ratios for the initial cohesin counts of chromosome 1 over chromosome 2 (N1/N2) | None | \[0.4 \- 2\] | Allows chromosome I and chromosome III to have 0.4-2x or 0.5-5x the amount of starting cohesin relative to chromosome II, respectively, based on (i) genome-wide cohesin binding patterns in fission yeast (Schmidt et al., 2009; Mizuguchi et al., 2014), (ii) the fact that centromere organization may influence cohesin levels (Paldi et al., 2020), and (iii) assumptions that the size of centromeres may influence cohesin load (Nonaka et al., 2002; Bernard et al., 2001)  |
| R32 | Ratios for the initial cohesin counts of chromosome 3 over chromosome 2 (N3/N2) | None | \[0.5 \- 5\] |  |
| r12 | Ratios for the cohesin thresholds of chromosome 1 over chromosome 2 (n1/n2​​) | None | \[0.25 \- 4\] | Range is set to allow up to \~4x difference in effective threshold between chromosomes, based on the observed 2-4 microtubules per kinetochore in fission yeast (Ding et al., 1993; Joglekar et al., 2008) . |
| r32 | Ratios for the cohesin thresholds of chromosome 3 over chromosome 2 (n3/n2​​) | None | \[0.25 \- 4\] |  |
| kmax | Maximum cohesin degradation rate | second-1 | \[0.001 \- 0.1\] | Range spans a large scale of possible maximum rates of cohesin cleavage.   |
|  | Timescale to reach k\_max​ in the Minimal model | *second* | \[2 \- 240\]  \[0.5 \- 5\] for separase autoactivation | Range spans a minutes- to seconds-scale rate of separase activation, to allow gradual activity ramps or rapid separase activation. |
| *b* | Number of cohesins removed per event in the Processive Separase Action model | Count | \[1 \- 50\] | Processive separase action model-specific parameter. Range allows small or large “bursts” of separase activity. |
| ninner | Number of cohesins in the innermost layer for the Steric Hindrance model | Count | \[1 \- 100\] | Steric hindrance model-specific parameter. Range allows a subset of cohesin to remain protected, but remains well below the upper bound of N2. |
| α | Multiplier modifying the cohesin threshold (n2) in threshold mutants | None | \[0.1 \- 0.7\] | Constrains MBC treatment conditions to lower n2 range to reflect reduced microtubule forces. |
| k | Multiplier modifying the max degradation rate (kmax​) in separase mutants | None | \[0.1 \- 1\] | Allows lower kmax in separase mutant conditions to reflect reduced separase activity. |
|  | Multiplier modifying the activation timescale (τ) in APC mutants | None | \[1 \- 10\] \[1 \- 3\] for separase autoactivation | Allows higher  in APC/C mutant and velcade treatment conditions to reflect slower separase activation.  |
| 2 | Multiplier modifying the activation timescale (τ) in Velcade mutants | None | \[1 \- 20\] \[1 \- 3\] for separase autoactivation |  |