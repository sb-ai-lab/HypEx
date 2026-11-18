# Advanced HypEx tutorials

These self-contained notebooks use reproducible synthetic data to show how to
design, diagnose and extend a HypEx experiment. The statistical chapters explain
how to interpret the results. The engineering chapters show how to configure
execution and add backend implementations.

## Notebooks

| # | Notebook | Main question | HypEx functionality |
|---|---|---|---|
| 01 | [Experiment design and A/A](aa/01_Experiment_Design_and_AA.ipynb) | Can we trust the assignment and the planned sample? | Roles, `HomogeneityTest`, stratified and multi-arm `AATest`, dry-test, FPR diagnostics, `MinSampleSize` |
| 07 | [A/A assignment and planning](aa/07_AA_Assignment_and_Planning.ipynb) | How do we turn a historical split into an experiment assignment? | Stratified `best_split`, fixed iterations, `early_stopping`, A/B on the recorded assignment, `MDEBySize` |
| 02 | [A/B/n inference and variance reduction](ab/02_AB_Inference_and_Variance_Reduction.ipynb) | Which effects survive correction, and can history make them more precise? | t/U/chi-square tests, confidence intervals, Holm correction, CUPED, CUPAC, feature importances |
| 06 | [From raw events to an A/B decision](ab/06_End_to_End_Experiment.ipynb) | How do we turn assignment and event logs into a rollout decision? | User-level `Dataset`, historical `AATest`, `HomogeneityTest` on the recorded groups, `ABTest`, CUPED and CUPAC |
| 08 | [Multiple testing in A/B/n](ab/08_AB_Multiple_Testing.ipynb) | Which comparisons survive adjustment, and is there a winning arm? | One-sided categorical counts, Holm, Benjamini–Hochberg, pandas `quantile` correction |
| 03 | [Matching and causal inference](03_Matching_and_Causal_Inference.ipynb) | What changes when treatment is not randomised? | L2/Mahalanobis matching, ATT/ATC/ATE, bias correction, quality tests, matched data, multiple neighbours, grouped matching |
| 04 | [Custom pipelines](04_Custom_Pipelines.ipynb) | How do we extend HypEx instead of only configuring it? | Preprocessing, `ExperimentData`, `Experiment`, `OnRoleExperiment`, custom comparator and reporter, `ExperimentShell`, `GroupExperiment` |
| 05 | [Spark backend](backends/05_Spark_Backend.ipynb) | What changes when execution becomes distributed? | pandas/Spark parity, aggregate-first tests, proportion z-test, caching, A/A and matching on Spark |
| 11 | [Backend factory](backends/11_Backend_Factory.ipynb) | How does one interface select different implementations? | `BackendFactory`, registration, backend resolution and missing backend handling |
| 09 | [Dataset configuration](configuration/09_Dataset_Configuration.ipynb) | How do we configure previews and limit local collection? | `DatasetConfig`, temporary overrides, backend conversion, joint counts |
| 10 | [Spark matching configuration](configuration/10_Spark_Matching_Configuration.ipynb) | How do search settings affect neighbour agreement and run time? | `MatchingConfig`, exact and shuffle modes, `FAISS_N_PROBES` |

## Feature coverage

This table tracks features demonstrated through the HypEx API in the basic
and advanced notebooks. Backend setup, synthetic data preparation, plots and
independent checks help readers interpret the examples. They do not count as
HypEx feature coverage. Custom analyses assembled outside HypEx are outside
the scope of this table.

**Covered** means there is a runnable example of the HypEx feature.
**Missing** means a HypEx feature still needs a notebook example.
Check the linked notebook for backend requirements.

| Area | Feature or workflow | Coverage | Where to look or what to add |
|---|---|---|---|
| A/A | Stratification, unequal allocation, multiple groups, `dry_test` | Covered | [01 · Design and A/A](aa/01_Experiment_Design_and_AA.ipynb) |
| A/A | Partially fixed assignment via `ConstGroupRole` | Covered | [Basic A/A](../tutorials/AATestTutorial.ipynb) shows fixed and unassigned users on pandas and Spark. |
| A/A | Use `best_split` as the actual assignment for A/B | Covered | [07 · A/A assignment](aa/07_AA_Assignment_and_Planning.ipynb#assignment-to-ab) uses the selected split to simulate delivery and read out A/B. |
| A/A | Fast assignment search via `early_stopping` versus fixed iterations | Covered | [07 · A/A assignment](aa/07_AA_Assignment_and_Planning.ipynb#candidate-assignments) compares completed iterations on the same data and seeds. |
| Design | Calculate sample size or detectable effect | Covered | [01 · Design and A/A](aa/01_Experiment_Design_and_AA.ipynb) uses `MinSampleSize`. [07 · A/A assignment](aa/07_AA_Assignment_and_Planning.ipynb#fixed-cohort-mde) uses `MDEBySize` for a fixed cohort. |
| A/B | A/B/n, numerical and categorical targets, Holm and Bonferroni | Covered | [02 · A/B inference](ab/02_AB_Inference_and_Variance_Reduction.ipynb) and [basic A/B](../tutorials/ABTestTutorial.ipynb). |
| A/B | Categorical A/B/n when a level occurs in just one arm | Covered | [08 · Multiple testing](ab/08_AB_Multiple_Testing.ipynb#comparison-family) checks that the category is retained in the chi-square comparison. |
| A/B | FDR and resampling-based quantile correction | Covered | [08 · Multiple testing](ab/08_AB_Multiple_Testing.ipynb) compares Holm and `fdr_bh`, then interprets the pandas `quantile` report. |
| Variance reduction | Raw A/B versus CUPED and CUPAC on the same cohort | Covered | [06 · End to end](ab/06_End_to_End_Experiment.ipynb) compares effects, confidence intervals and variance reduction. [02 · A/B inference](ab/02_AB_Inference_and_Variance_Reduction.ipynb) also reads feature importances. |
| Variance reduction | CUPAC with several historical periods and lagged features | Covered | [Basic CUPED/CUPAC](../tutorials/CUPED_CUPAC.ipynb) includes multiple lag columns. [02 · A/B inference](ab/02_AB_Inference_and_Variance_Reduction.ipynb) and [06 · End to end](ab/06_End_to_End_Experiment.ipynb#reduce-variance) show how to assign `PreTargetRole` and `FeatureRole` for two historical periods. |
| Variance reduction | CUPAC model selection and report access | Covered | [Basic CUPED/CUPAC](../tutorials/CUPED_CUPAC.ipynb) compares built-in models, adds CatBoost when installed, and reads the method-specific reports. |
| Matching | Distances, weights, bias correction, multiple neighbours and matching within groups | Covered | [03 · Matching](03_Matching_and_Causal_Inference.ipynb) uses `Matching` and reads `summary`, `quality_results` and `indexes`. |
| Matching | FAISS search modes and categorical encoding | Covered | [Basic matching](../tutorials/MatchingTutorial.ipynb) compares `faiss_mode="base"` and `"fast"` and shows `encode_categories`. |
| Pipelines | Role-based preprocessing and custom comparators, reporters and grouped analyses | Covered | [04 · Custom pipelines](04_Custom_Pipelines.ipynb) uses `NaDropper`, `CategoryAggregator`, feature filters, `Float32Caster` and `DummyEncoder`. |
| Pipelines | Parameter sweeps with `ParamsExperiment` | Missing | Show a standalone grid for a custom pipeline and how to read the per-configuration report. A/A uses parameter sweeps internally. |
| Dataset | Construct, filter, join and aggregate with `Dataset`, report with `SmallDataset` | Covered | [Dataset workflow](../tutorials/DatasetWorkflowTutorial.ipynb) runs the same join and summary on pandas and Spark. [Dataset methods](../tutorials/DatasetTutorial.ipynb) is the original reference example. |
| Spark | Backend parity, aggregate tests, cache lifecycle and local Spark matching | Covered | [05 · Spark](backends/05_Spark_Backend.ipynb), including `StatsZTest.calc` and `persist` / `unpersist`. |
| Spark | Backend conversion with `Dataset.to_backend` and joint counts with `value_counts` | Covered | [09 · Dataset configuration](configuration/09_Dataset_Configuration.ipynb#conversion) runs a backend round trip and counts combinations of segment and visit count. |
| Configuration | Dataset display and Spark-to-pandas conversion limits | Covered | [09 · Dataset configuration](configuration/09_Dataset_Configuration.ipynb) changes `DISPLAY_ROWS` and `DISPLAY_COLS` and demonstrates the conversion guard. |
| Configuration | Fine-tune distributed FAISS matching | Covered | [10 · Spark matching configuration](configuration/10_Spark_Matching_Configuration.ipynb) compares exact search with shuffle mode at two probe counts. |
| Configuration | Clean up temporary FAISS index files | Covered | [10 · Spark matching configuration](configuration/10_Spark_Matching_Configuration.ipynb#index-cleanup) shows a separate cleanup cell and cleanup with `try/except/finally` for direct extension calls. |
| Backends | Register, inspect and resolve implementations through `BackendFactory` | Covered | [11 · Backend factory](backends/11_Backend_Factory.ipynb) uses the HypEx registry to list mappings and resolve a component for the input backend. |

## Suggested routes

- **Experiment analyst:** 06 → 01 → 02. Follow the full workflow first, then
  explore assignment validity, multiple comparisons and variance reduction in depth.
  Continue with 07 for assignment and planning, or 08 for multiple testing.
- **Causal-inference practitioner:** 01 → 03. Keep randomised and observational designs
  clearly separated.
- **Pipeline developer:** 04 → 05 → 11. Learn the executor and reporting
  contracts, then work with backend-specific implementations.
- **Spark configuration:** 05 → 09 → 10. Start with distributed execution,
  then tune dataset limits and matching search.

The notebooks are independent: opening only the one you need is fine. Together they
cover the user-facing shells, the main statistical blocks, preprocessing, custom
composition and both execution backends without requiring a separate dataset download.

## Run

From the repository root:

```bash
poetry install --all-extras --with dev
poetry run jupyter notebook examples/advanced_tutorials
```

Execute the whole series end to end with the project's notebook runner:

```bash
tox -e notebooks -- examples/advanced_tutorials
```

Notebooks 05, 09, 10 and 11 use local Spark and require Java 8, 11 or 17
and PySpark 3.5.1. Set `JAVA_HOME` or make Java available on `PATH`.
Notebook 05 also checks common Homebrew locations. Notebook 10's exact
search uses `SparkFiles`. On Windows, that requires Hadoop's `winutils`.

## Reading conventions

- `OK` means **no detected difference** in A/A and matching-quality reports, but a
  rejected null in A/B reports. Always read the accompanying p-value and effect size.
- Synthetic effects are known by construction, so estimates can be compared with the
  truth. One run illustrates behaviour; it is not a performance benchmark.
- Historical covariates always precede treatment. Using post-treatment variables for
  A/A checks, CUPED/CUPAC or matching would invalidate the examples' logic.
- Outputs are executed and stored so the series is readable without running code.
