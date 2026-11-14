# Advanced HypEx tutorials

A practical series for users who have already seen the basic tutorials and want to
understand **how to design, diagnose and extend an experiment**, not only how to call
its shell. Every notebook is self-contained, uses reproducible synthetic data with a
known answer, and includes interpretation and methodological caveats.

## Notebooks

| # | Notebook | Main question | HypEx functionality |
|---|---|---|---|
| 01 | [Experiment design and A/A](01_Experiment_Design_and_AA.ipynb) | Can we trust the assignment and the planned sample? | Roles, `HomogeneityTest`, stratified and multi-arm `AATest`, dry-test, FPR diagnostics, `MinSampleSize` |
| 02 | [A/B/n inference and variance reduction](02_AB_Inference_and_Variance_Reduction.ipynb) | Which effects survive correction, and can history make them more precise? | t/U/chi-square tests, confidence intervals, Holm correction, CUPED, CUPAC, feature importances |
| 03 | [Matching and causal inference](03_Matching_and_Causal_Inference.ipynb) | What changes when treatment is not randomised? | L2/Mahalanobis matching, ATT/ATC/ATE, bias correction, quality tests, matched data, multiple neighbours, grouped matching |
| 04 | [Custom pipelines](04_Custom_Pipelines.ipynb) | How do we extend HypEx instead of only configuring it? | Preprocessing, `ExperimentData`, `Experiment`, `OnRoleExperiment`, custom comparator and reporter, `ExperimentShell`, `GroupExperiment` |
| 05 | [Spark backend](05_Spark_Backend.ipynb) | What changes when execution becomes distributed? | pandas/Spark parity, aggregate-first tests, proportion z-test, caching, A/A and matching on Spark |

## Suggested routes

- **Experiment analyst:** 01 → 02. Start with assignment validity, multiple comparisons
  and variance reduction.
- **Causal-inference practitioner:** 01 → 03. Keep randomised and observational designs
  clearly separated.
- **Pipeline developer:** 04 → 05. Learn the executor/state/reporter contracts, then
  their backend-specific behaviour.

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

Notebook 05 requires Java 8/11/17 and PySpark 3.5.1. It detects Java on `PATH` and
common Homebrew locations, starts a local two-core Spark session, and stops it at the
end. The other notebooks need only the regular HypEx dependencies.

## Reading conventions

- `OK` means **no detected difference** in A/A and matching-quality reports, but a
  rejected null in A/B reports. Always read the accompanying p-value and effect size.
- Synthetic effects are known by construction, so estimates can be compared with the
  truth. One run illustrates behaviour; it is not a performance benchmark.
- Historical covariates always precede treatment. Using post-treatment variables for
  A/A checks, CUPED/CUPAC or matching would invalidate the examples' logic.
- Outputs are executed and stored so the series is readable without running code.
