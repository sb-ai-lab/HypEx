# HypEx: Advanced Causal Inference and AB Testing Toolkit

[![PyPI version](https://img.shields.io/pypi/v/hypex?color=darkgreen)](https://pypi.org/project/hypex/)
![Python versions](https://img.shields.io/badge/python-3.8_|_3.9_|_3.10_|_3.11_|_3.12-blue)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Telegram](https://img.shields.io/badge/chat-on%20Telegram-2ba2d9.svg)](https://t.me/hypexchat)

HypEx (Hypotheses and Experiments) is a library for causal inference and AB testing. It runs the same experiments on
**pandas** for everyday analysis and on **Apache Spark** for data that does not fit on one machine.

## What's new in 2.0

HypEx 2.0 is built on a new architecture: experiments are pipelines of small composable blocks working on a
backend-agnostic `Dataset`.

- **New interface.** Import paths, class names and result objects differ from 0.1.x. Migrate with the
  [tutorials](https://github.com/sb-ai-lab/HypEx/tree/master/examples/tutorials).
- **Spark backend.** `Dataset` runs on pandas or Spark with the same API for AA tests, AB tests, homogeneity tests and
  matching.
- **CUPED and CUPAC** variance reduction in `ABTest`, with a variance reduction report.
- **Multiple-testing corrections** in `ABTest` (Holm by default, plus Bonferroni, Sidak, FDR and others).
- Version 0.1.x is no longer supported. If you need it: `pip install hypex==0.1.10`.

## Introduction

HypEx employs Rubin's Causal Model (RCM) for matching closely related pairs, ensuring equitable group comparisons when
estimating treatment effects. Its automated pipeline calculates the Average Treatment Effect (ATE), Average Treatment
Effect on the Treated (ATT) and Average Treatment Effect on the Control (ATC), with a standardized interface for running
the estimations.

Beyond causal inference, HypEx provides AA tests, homogeneity tests and AB tests (including A/B/n tests with multiple
testing corrections and CUPED/CUPAC variance reduction) to rigorously test hypotheses and validate experimental results.

## Features

- **Matching**: Faiss-based nearest neighbor search (Mahalanobis or L2 distance), several neighbors per object, feature
  weights, matching inside groups, optional categorical encoding and bias estimation.
- **Matching quality tests**: SMD, KS, PSI, Repeats, t-test and chi2-test to check the robustness of the matching.
- **AA test**: repeated random splits with a selection of the most homogeneous split, optional stratification, early
  stopping and custom group sizes.
- **Homogeneity test**: compares target and feature distributions between groups.
- **AB test**: group difference with t-test, u-test and chi2-test, A/B/n tests with multiple testing corrections, CUPED
  and CUPAC variance reduction with a detailed report.
- **Roles**: describe your data once (`TreatmentRole`, `TargetRole`, `FeatureRole`, `StratificationRole`, `InfoRole`)
  and every experiment picks what it needs.
- **Pandas and Spark backends** behind one `Dataset` API.

## Warnings

Some functions in HypEx help to solve auxiliary tasks but cannot automate decisions on experiment design.

**Note:** For Matching, it is recommended to use no more than 7 features: more may lead to the curse of dimensionality
and make the results unrepresentative.

## Installation

```bash
pip install -U hypex
```

Optional extras for CUPAC models:

```bash
pip install -U "hypex[cat]"   # CatBoost
pip install -U "hypex[lgbm]"  # LightGBM
pip install -U "hypex[all]"   # everything above
```

Requirements: Python `>=3.8, <3.13`. PySpark `3.5.1` is installed with the library; running on Spark also needs Java
(8, 11 or 17).

## Quick start

Explore usage examples and tutorials [here](https://github.com/sb-ai-lab/HypEx/tree/master/examples/tutorials/).

### Matching example

```python
from hypex.dataset import Dataset, InfoRole, TreatmentRole, TargetRole, FeatureRole
from hypex import Matching

data = Dataset(
    roles={
        "user_id": InfoRole(int),  # InfoRole for ID
        "treat": TreatmentRole(int),  # TreatmentRole identifies the group (control or target)
        "post_spends": TargetRole(float),  # TargetRole for the target
    },
    data="data.csv",
    default_role=FeatureRole(),  # All remaining columns are features (used to search for similar objects)
)

test = Matching()  # Classic matching (Mahalanobis distance + quality tests)
test = Matching(distance="l2")  # Choose the distance
test = Matching(n_neighbors=3)  # Several neighbors per object
test = Matching(group_match=True)  # Match inside groups of categorical features
test = Matching(weights={"age": 2.0})  # Custom feature weights
test = Matching(extract_full_data=True, compute_indexes=True)  # Also build full_data and indexes

result = test.execute(data)
result.summary  # Summary of results (ATE, ATT, ATC)
result.quality_results  # Matching quality tests
result.full_data  # Wide dataset with pairs (requires extract_full_data=True)
result.indexes  # Matched pairs of indexes, good for join (requires compute_indexes=True)
```

`full_data` and `indexes` are skipped by default because they are expensive on large data.

More about Matching [here](https://github.com/sb-ai-lab/HypEx/tree/master/examples/tutorials/MatchingTutorial.ipynb)

### AA-test example

```python
from hypex.dataset import Dataset, InfoRole, TargetRole, StratificationRole
from hypex import AATest

data = Dataset(
    roles={
        "user_id": InfoRole(int),  # InfoRole for ID
        "pre_spends": TargetRole(),  # TargetRole for the homogeneity check
        "post_spends": TargetRole(),  # TargetRole for the homogeneity check
        "gender": StratificationRole(str),  # StratificationRole for strata
    },
    data="data.csv",
)

aa = AATest(n_iterations=10)
result = aa.execute(data)

result.summary  # Summary of the whole test
result.aa_score  # AA score
result.best_split  # The most homogeneous split
result.best_split_statistic  # Statistics of the best split
```

More about AA test [here](https://github.com/sb-ai-lab/HypEx/tree/master/examples/tutorials/AATestTutorial.ipynb)

### Homogeneity test example

```python
from hypex import HomogeneityTest

# `data` has a TreatmentRole column and TargetRole columns, as above
result = HomogeneityTest().execute(data)
result.summary
```

More about the homogeneity test [here](https://github.com/sb-ai-lab/HypEx/tree/master/examples/tutorials/HomogeneityTestTutorial.ipynb)

### AB-test example

```python
from hypex.dataset import Dataset, InfoRole, TreatmentRole, TargetRole
from hypex import ABTest

data = Dataset(
    roles={
        "user_id": InfoRole(int),  # InfoRole for ID
        "treat": TreatmentRole(),  # TreatmentRole identifies the group (control or target)
        "pre_spends": TargetRole(),  # TargetRole for A/B(n) tests
        "post_spends": TargetRole(),  # TargetRole for A/B(n) tests
    },
    data="data.csv",
)

test = ABTest()  # Classic A/B test
test = ABTest(multitest_method="bonferroni")  # A/Bn test with Bonferroni correction
test = ABTest(additional_tests=["t-test", "u-test", "chi2-test"])  # Choose the tests
test = ABTest(cuped_features={"post_spends": "pre_spends"})  # CUPED variance reduction
test = ABTest(enable_cupac=True, cupac_models=["linear", "ridge"])  # CUPAC variance reduction

result = test.execute(data)
result.summary  # Summary of results
result.multitest  # Multiple testing corrections
result.sizes  # Group sizes
result.variance_reduction_report  # Variance reduction report for CUPED/CUPAC
```

#### CUPED and CUPAC

CUPED uses one pre-experiment value of the target. CUPAC predicts the target from pre-experiment covariates, one model
per time period transition. Its configuration comes from the roles: `PreTargetRole(parent=..., lag=...)` marks a
historical value of a target, `FeatureRole(parent=..., lag=...)` a covariate of the same period, and
`TargetRole(cofounders=[...])` lists the covariates of the target.

```python
from hypex import ABTest
from hypex.dataset import Dataset, FeatureRole, InfoRole, PreTargetRole, TargetRole, TreatmentRole

data = Dataset(
    roles={
        "d": TreatmentRole(),
        "y": TargetRole(cofounders=["X1", "X2"]),
        "y_lag1": PreTargetRole(parent="y", lag=1),  # target one period ago
        "X1_lag1": FeatureRole(parent="X1", lag=1),
        "X2_lag1": FeatureRole(parent="X2", lag=1),
        "y_lag2": PreTargetRole(parent="y", lag=2),  # target two periods ago
        "X1_lag2": FeatureRole(parent="X1", lag=2),
        "X2_lag2": FeatureRole(parent="X2", lag=2),
    },
    data=df,
    default_role=InfoRole(),  # all other columns are ignored
)

result = ABTest(cuped_features={"y": "y_lag1"}).execute(data)  # CUPED
result.variance_reduction_report

# CUPAC: the best model is selected for every transition
result = ABTest(enable_cupac=True, cupac_models=["linear", "ridge", "lasso", "catboost"]).execute(data)
result.cupac.variance_reductions  # variance reduction per model
result.cupac.feature_importances  # which covariates helped the most
```

`"catboost"` requires `pip install "hypex[cat]"`. If `cupac_models` is omitted, all available models are tried.

Full guide: [CUPED & CUPAC tutorial](https://github.com/sb-ai-lab/HypEx/tree/master/examples/tutorials/CUPED_CUPAC.ipynb).

More about AB test [here](https://github.com/sb-ai-lab/HypEx/tree/master/examples/tutorials/ABTestTutorial.ipynb)

## Spark backend

Every experiment above runs on Spark with the same API. Create a `SparkSession`, pass it to `Dataset` and select the
backend explicitly. `data` can be a `pandas.DataFrame`, it is converted to a Spark DataFrame.

```python
from pyspark.sql import SparkSession

from hypex import AATest
from hypex.dataset import Dataset, InfoRole, TargetRole
from hypex.utils import BackendsEnum

spark = SparkSession.builder.master("local[*]").appName("HypEx").getOrCreate()

data = Dataset(
    roles={
        "user_id": InfoRole(int),
        "pre_spends": TargetRole(),
        "post_spends": TargetRole(),
    },
    data=pandas_df,
    session=spark,
    backend=BackendsEnum.spark,
)

result = AATest(n_iterations=10, float32=True).execute(data)
result.summary
```

Notes:

- Computation is lazy and distributed. Spark pays off on large data; on small data pandas is faster.
- `Dataset.data` is a `pyspark.sql.DataFrame` on this backend, so inspect it with the Spark API.
- Convert between backends with `dataset.to_backend(BackendsEnum.pandas)` or
  `dataset.to_backend(BackendsEnum.spark, session=spark)`. Converting a large Spark dataset to pandas is refused when it
  would not fit in memory.
- `SmallDataset` is pandas-based by design: it holds compact `analysis_tables` and reporting results.
- Supported runtime: Python `>=3.8`, PySpark `3.5.1`, Java 8/11/17.

Each tutorial has a "Working with the Spark backend" section.

## Documentation

For more detailed information about the library, visit our
[documentation on ReadTheDocs](https://hypex.readthedocs.io/en/latest/). It has guides and tutorials to get started and
detailed API documentation for advanced use cases.

The architecture of the package (executors, `ExperimentData`, backends) is described in
[hypex/README.md](hypex/README.md).

## Contributions

Join our community! For guidelines on contributing, reporting issues or seeking support, please refer to
our [Contributing Guidelines](https://github.com/sb-ai-lab/HypEx/blob/master/.github/CONTRIBUTING.md).

## More Information & Resources

[Habr (ru)](https://habr.com/ru/companies/sberbank/articles/778774/) - discover how HypEx is revolutionizing causal
inference in various fields.  
[A/B testing seminar](https://www.youtube.com/watch?v=B9BE_yk8CjA&t=53s&ab_channel=NoML) - Seminar in NoML about
matching and A/B testing  
[Matching with HypEx: Simple Guide](https://www.kaggle.com/code/kseniavasilieva/matching-with-hypex-simple-guide) -
Simple matching guide with explanation  
[Matching with HypEx: Grouping](https://www.kaggle.com/code/kseniavasilieva/matching-with-hypex-grouping) - Matching
with grouping guide  
[HypEx vs Causal Inference and DoWhy](https://www.kaggle.com/code/kseniavasilieva/hypex-vs-causal-inference-and-dowhy) -
discover why HypEx is the best solution for causal inference  
[HypEx vs Causal Inference and DoWhy: part 2](https://www.kaggle.com/code/kseniavasilieva/hypex-vs-causal-inference-part-2) -
discover why HypEx is the best solution for causal inference

### Testing different libraries for the speed of matching

Visit [this](https://www.kaggle.com/code/kseniavasilieva/hypex-vs-causal-inference-part-2) notebook on Kaggle and
estimate results by yourself. The benchmark was run on an earlier HypEx version.

| Group size             | 32 768 | 65 536 | 131 072 | 262 144 | 524 288 | 1 048 576 | 2 097 152 | 4 194 304 |
|------------------------|--------|--------|---------|---------|---------|-----------|-----------|-----------|
| Causal Inference       | 46s    | 169s   | None    | None    | None    | None      | None      | None      |
| DoWhy                  | 9s     | 19s    | 40s     | 77s     | 159s    | 312s      | 615s      | 1 235s    |
| HypEx with grouping    | 2s     | 6s     | 16s     | 42s     | 167s    | 509s      | 1 932s    | 7 248s    |
| HypEx without grouping | 2s     | 7s     | 21s     | 101s    | 273s    | 982s      | 3 750s    | 14 720s   |

## Join Our Community

Have questions or want to discuss HypEx? Join our [Telegram chat](https://t.me/HypExChat) and connect with the
community and the developers.
