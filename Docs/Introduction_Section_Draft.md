# Introduction

## The Critical Role of Chromosome Segregation in Cell Division

Accurate chromosome segregation during cell division is fundamental to maintaining genomic stability and preventing diseases ranging from developmental disorders to cancer [1,2]. During anaphase, the precise timing of chromosome separation determines whether daughter cells receive the correct complement of genetic material. This process requires the coordinated degradation of cohesin proteins that hold sister chromatids together, mediated primarily by the enzyme Separase [3,4]. Despite its critical importance, the molecular mechanisms governing the timing and regulation of chromosome segregation remain incompletely understood, particularly regarding how different cellular perturbations affect segregation dynamics.

## Molecular Machinery of Chromosome Segregation

The chromosome segregation process involves a complex regulatory network centered on cohesin protein complexes and their degradation machinery [5]. Cohesin proteins form ring-like structures that encircle sister chromatids, maintaining their cohesion from DNA replication through early mitosis [6]. The transition to anaphase requires the activation of Separase, a cysteine protease that cleaves the Rad21/Scc1 subunit of cohesin, thereby releasing sister chromatids for segregation to opposite spindle poles [7,8].

Separase activity is tightly regulated through multiple inhibitory mechanisms to prevent premature chromosome separation. The protein Securin acts as a direct inhibitor of Separase, while Cyclin B-CDK1 complexes provide additional inhibitory phosphorylation [9,10]. The anaphase-promoting complex/cyclosome (APC/C), activated by the spindle assembly checkpoint (SAC), coordinates the degradation of both Securin and Cyclin B, thereby synchronizing Separase activation with proper spindle formation [11,12]. This multi-layered regulatory system ensures that chromosome segregation occurs only when all chromosomes are properly attached to the spindle apparatus.

## Timing Variability and Biological Significance

While the core segregation machinery is highly conserved, substantial variability exists in the timing of chromosome separation both within and between cell types [13,14]. This timing variability is not merely noise but appears to be functionally significant, with different chromosomes often segregating at slightly different times even within the same cell [15]. Understanding the sources and consequences of this timing heterogeneity is crucial for comprehending how cells maintain fidelity while accommodating the inherent stochasticity of molecular processes.

Recent experimental advances have enabled precise measurement of chromosome segregation timing at single-cell resolution, revealing that segregation timing follows characteristic probability distributions that vary with genetic background and cellular conditions [16,17]. These distributions provide a quantitative fingerprint of the underlying molecular processes, offering an opportunity to develop mathematical models that can predict how genetic or pharmacological perturbations affect segregation dynamics.

## Challenges in Modeling Chromosome Segregation

Mathematical modeling of chromosome segregation faces several fundamental challenges. First, the process involves discrete molecular events (individual cohesin cleavage reactions) occurring in a stochastic manner, requiring models that can capture both the discrete nature of the process and its inherent randomness [18]. Second, the regulatory network involves multiple feedback mechanisms and time-varying reaction rates as regulatory proteins accumulate and are degraded during cell cycle progression [19,20]. Third, different chromosomes within the same cell may have different cohesin densities and degradation kinetics, necessitating models that account for inter-chromosomal variability [21].

Previous modeling efforts have largely focused on simplified representations that assume constant degradation rates and ignore the complex regulatory dynamics governing Separase activation [22,23]. While these models provide valuable insights into basic segregation mechanics, they cannot capture the rich temporal dynamics observed experimentally, particularly the characteristic changes in degradation rates as cells progress through anaphase.

## The Need for Multi-Mechanism Models

Emerging evidence suggests that chromosome segregation involves multiple distinct molecular mechanisms operating simultaneously. These include: (1) time-varying degradation rates reflecting the progressive activation of Separase and accumulation of regulatory factors [24,25], (2) cooperative degradation events where multiple cohesin complexes are cleaved in rapid succession, potentially due to local concentration effects or processive enzyme activity [26], and (3) feedback mechanisms where the extent of cohesin degradation influences the rate of further degradation, possibly through structural changes in chromatin organization [27,28].

No current model integrates all these mechanisms in a unified framework capable of quantitative prediction across different genetic backgrounds and experimental conditions. Such a comprehensive model would not only advance our fundamental understanding of chromosome segregation but also provide a platform for predicting the effects of therapeutic interventions targeting the segregation machinery.

## Parameter Identifiability: A Critical Challenge

A major obstacle in developing complex biological models is parameter identifiability—the ability to reliably estimate model parameters from experimental data [29,30]. Many biological models suffer from "sloppiness," where multiple parameter combinations can produce indistinguishable fits to data, making biological interpretation ambiguous [31]. This problem is particularly acute for models with many parameters, where compensation effects between parameters can mask their individual biological significance.

The parameter identifiability problem has received limited attention in chromosome segregation modeling, despite its critical importance for model reliability and biological interpretation. Systematic assessment of parameter identifiability requires sophisticated computational approaches, including parameter recovery studies where known parameter values are used to generate synthetic data, which is then used to test whether the original parameters can be reliably recovered [32].

## Experimental Model Systems

Studies of chromosome segregation timing have been greatly facilitated by the development of yeast model systems, particularly _Saccharomyces cerevisiae_, where genetic manipulations can precisely perturb different components of the segregation machinery [33,34]. Key experimental approaches include:

1. **Threshold mutants** that reduce the amount of cohesin required for sister chromatid cohesion, effectively lowering the degradation threshold needed for separation [35]
2. **Separase mutants** with reduced enzymatic activity, leading to slower cohesin degradation rates [36]
3. **APC mutants** that alter the timing of Separase activation, affecting the temporal dynamics of the degradation process [37]
4. **Cohesin mutants** with altered initial protein levels, changing the starting conditions for the degradation process [38]

These mutant strains provide a systematic way to perturb different aspects of the segregation machinery while maintaining cellular viability, enabling quantitative analysis of how specific molecular defects affect segregation timing distributions.

## Research Objectives and Approach

In this study, we develop a comprehensive mathematical framework for modeling chromosome segregation timing that integrates multiple biological mechanisms within a unified stochastic model. Our approach addresses three key objectives:

**First**, we develop a series of increasingly sophisticated mathematical models that incorporate: (a) time-varying degradation rates reflecting Separase activation dynamics, (b) burst-like degradation events representing cooperative cohesin cleavage, and (c) feedback mechanisms linking degradation extent to degradation rate. These models are formulated as continuous-time Markov processes that can be simulated exactly using the Gillespie algorithm [39].

**Second**, we implement both analytical approximation methods (Method of Moments) for rapid parameter estimation and exact stochastic simulation approaches for high-fidelity modeling. This dual approach enables both efficient parameter screening and rigorous model validation across different levels of biological complexity.

**Third**, we conduct comprehensive parameter identifiability analysis using parameter recovery studies to assess which model parameters can be reliably estimated from experimental data. This analysis addresses a critical gap in current chromosome segregation modeling by providing quantitative measures of parameter uncertainty and model reliability.

Our experimental validation uses timing data from five different yeast strains, including wild-type controls and four mutant backgrounds that perturb different aspects of the segregation machinery. By fitting our models to these data, we aim to identify the minimal set of biological mechanisms required to explain observed segregation timing variability and to predict how different molecular perturbations affect segregation dynamics.

## Broader Implications

This work addresses fundamental questions about the relationship between molecular complexity and model identifiability in systems biology. By systematically comparing models of different complexity and rigorously assessing their parameter identifiability, we provide insights into the trade-offs between biological realism and predictive reliability that are relevant across many areas of quantitative biology [40,41].

From a biomedical perspective, understanding chromosome segregation timing has direct relevance to cancer biology, where segregation defects contribute to the chromosomal instability characteristic of many tumor types [42,43]. Mathematical models that can predict how genetic or pharmacological perturbations affect segregation dynamics could inform the development of cancer therapeutics targeting the cell division machinery [44].

More broadly, our approach demonstrates how rigorous parameter identifiability analysis can be integrated into the model development process, providing a framework for developing reliable predictive models in complex biological systems. This methodology is increasingly important as biological models become more sophisticated and are applied to problems requiring quantitative prediction rather than qualitative understanding.

---

## References

_[Note: In the actual paper, these would be replaced with full citations]_

[1] Holland, A.J. & Cleveland, D.W. Boveri revisited: chromosomal instability, aneuploidy and tumorigenesis. _Nature Reviews Molecular Cell Biology_ (2009).

[2] Thompson, S.L. & Compton, D.A. Examining the link between chromosomal instability and aneuploidy in human cells. _Journal of Cell Biology_ (2008).

[3] Uhlmann, F. Separase regulation during mitosis. _Biochemical Society Transactions_ (2016).

[4] Konečná, K. et al. Separase and Roads to Disengage Sister Chromatids during Anaphase. _International Journal of Molecular Sciences_ (2023).

[5] Nasmyth, K. Cohesin: a catenase with separate entry and exit gates? _Nature Cell Biology_ (2011).

[6-44] _[Additional references would be included in the full manuscript]_
