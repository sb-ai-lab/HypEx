# Alpha integration behavior report

Date: 2026-11-08  
Branch: `dev/master_alpha_agent_check_behavior`  
Base: `dev/master` at `6f7b4564`

## Scope and method

All 22 requested fix branches were merged in the requested order. Every source branch tip was verified to be an ancestor of the integration branch.

The synthetic harness is `examples/experiments/alpha_agent_behavior_check.py`. It uses seeded data, independent SciPy/group-mean references, and known treatment effects. Detailed per-case output and tracebacks are in:

- `examples/experiments/alpha_agent_behavior_pandas_results.json`
- `examples/experiments/alpha_agent_behavior_spark_results.json`

Environment: Python 3.11.14, PySpark 3.5.1, Homebrew OpenJDK 11. Spark worker and driver Python were both pinned to the project virtual environment. A system Python 3.9 worker initially caused a version mismatch; those environmental failures were discarded and rerun, not counted as library defects.

### Synthetic coverage

| Family | Coverage |
|---|---|
| AA | 200 null splits per backend; two continuous targets, one categorical target, one continuous feature; empirical rejection rates and p-value uniformity; stratification; `sample_size=0.75`; `control_size=0.3`; three groups; `float32`; equal variance; fixed-seed reproducibility |
| AB | Welch reference under unequal variances; null and strong effects; groups `0/1/2/10`; Bonferroni and enum-based correction; mixed continuous/categorical targets; disjoint categories; quantile correction on pandas; CUPED |
| Homogeneity | Null and shifted continuous targets; categorical targets with disjoint categories; natural ordering of groups `0/1/2/10`; manual group-mean references |
| Matching | Mahalanobis and L2; base and fast FAISS; 1 and 3 neighbors; bias estimation on/off; categorical encoding; automatic and explicit quality tests; matched indexes and full data; 40 independent ATE/CI runs |

Spark quantile multitest was excluded because it is explicitly unsupported. Spark KS numerical equality with SciPy was not required because its histogram approximation is documented. Already-documented strict-xfail limitations, including `group_match=True` and end-to-end CUPAC, were not reported as new defects.

### Synthetic outcome

- Pandas: **11/14 cases passed**. Three failed cases represent two unique defects (chi-square calibration and scalar Matching quality tests).
- Spark: **12/14 cases passed**. Two failed cases represent two unique defects (AA subsampling and scalar Matching quality tests).
- Total: **23/28 cases passed**, with three unique defects confirmed across the matrices.

## Integration correction made during merging

The keyless-index merge fix initially forced `left_index=True` and `right_index=True` even for keyed calls such as `merge(on="category")`. This broke six existing chi-square tests. Both backends now infer an index join only when no column or index keys are supplied, and convert remaining `None` index flags to `False`.

A backend-parametrized regression test was added in `tests/unit/dataset/test_dataset_indexing.py`. The three merge conflicts were test-only; all tests from both sides were retained.

The included issue-15 Spark scalar-`.loc` fix also needed a follow-up. In this environment, scalar Spark `.loc` returns a local **pandas** Series, but the branch converted only `pyspark.pandas.Series`. `SparkNavigation.loc` now converts either Series type with `.to_frame()` (`hypex/dataset/backends/spark_backend.py:1216-1219`). The scalar result is consequently a pandas-backed frame, matching the local Series returned by the upstream scalar operation and avoiding mixed-type Arrow conversion errors.

Verification after these corrections:

- `pytest -m "not spark" -q`: **1273 passed, 54 xfailed**.
- Targeted pandas/Spark merge and ordering tests: **12 passed**.
- Final targeted scalar-`.loc` and merge tests: **6 passed**.
- Complete `pytest -q`: **1760 passed, 18 skipped, 79 xfailed; 3 failed**. The three remaining failures are exclusively the out-of-scope issue-13 FAISS cases detailed below.
- Direct Ruff equivalents of the lint environment: **all checks passed; 207 files formatted**. The `tox` executable was unavailable.

## Confirmed defects

### 1. Pandas chi-square uses scaled counts and is conservative under the null

**Affected paths:** pandas AA, AB categorical tests, Homogeneity categorical tests, and Matching categorical quality tests.

**Evidence**

- Over 200 null AA splits, categorical chi-square rejected **0/200** on pandas versus **8/200 (4%)** on Spark.
- The pandas p-values had mean **0.625434** and a KS-uniformity p-value of **1.965e-9**. This is not consistent with a calibrated uniform null distribution. Spark had mean **0.491466** and uniformity p-value **0.703327**.
- On the deterministic categorical AB dataset, pandas returned p=`2.934626e-11`; raw-count SciPy returned p=`4.662643e-22`. Spark matched SciPy.
- Minimal reproduction with raw counts `[[80, 20], [60, 40]]`:
  - raw-count SciPy: statistic `8.595238`, p=`0.00337043`;
  - HypEx pandas table `[[40, 30], [10, 20]]`: statistic `3.857143`, p=`0.04953461`.

**Confirmed cause**

`PandasChi2TestExtension.matrix_preparation` multiplies the first group's counts by `1 - proportion` and the second group's counts by `proportion` (`hypex/extensions/scipy_stats.py:245-253`). For equal groups this halves the effective sample size. `scipy.stats.chi2_contingency` expects observed counts, not counts rescaled to a smaller common total. The Spark implementation uses raw counts (`hypex/extensions/scipy_stats.py:446-457`).

**Impact**

P-values are inflated and categorical tests lose power. AA categorical scores and split selection are also distorted. Strong effects can still be detected, which is why simple smoke tests do not expose the problem.

### 2. Spark AA fails whenever `sample_size < 1`

**Affected path:** `AATest(sample_size=...)` on Spark, with or without stratification.

**Reproduction**

```python
AATest(n_iterations=1, sample_size=0.75, random_states=[0]).execute(spark_ds)
```

It fails in `GroupSizes` with:

```text
TypeError: '<' not supported between instances of 'NoneType' and 'str'
```

**Evidence and confirmed cause**

- Stratification alone passed with `control_size` 0.5 and 0.3.
- `sample_size=0.75` failed with and without stratification, at both control sizes, and with `float32` on or off.
- In a 100-row reproduction, the splitter produced 44 control rows, 39 test rows, and **17 null labels**. The dataset still contains all 100 rows after additional-field alignment.
- Spark group statistics include the null key; `sorted(group_col_stats)` at `hypex/comparators/abstract.py:1096` then attempts to order `None` with strings.

The sampled-out rows need to be excluded from downstream comparisons or the null group must be dropped consistently before sorting/aggregation.

### 3. `Matching(quality_tests="t-test")` iterates the string as characters

**Affected backends:** both pandas and Spark.

**Reproduction**

```python
Matching(distance="l2", quality_tests="t-test")
```

It fails immediately with `KeyError: 't'`. The documented scalar form is therefore unusable; `quality_tests=["t-test"]` works.

**Confirmed cause**

The non-`auto` branch iterates `quality_tests` directly in `hypex/matching.py:171-175`. A string is iterated as characters instead of being normalized to a one-element list.

## Remaining failure outside the requested merge set

The complete suite also reproduces three Spark FAISS failures:

- `tests/dataset_spark_backend/test_matching_spark.py::test_matching_spark_matches_pandas[1]`
- `tests/dataset_spark_backend/test_matching_spark.py::test_matching_spark_matches_pandas[2]`
- `tests/unit/test_ml_faiss.py::test_spark_extension_matches_brute_force`

All three reach `MiniBatchKMeans.partial_fit` with a one-dimensional partition array. scikit-learn requires a 2D array and raises `ValueError: Expected 2D array, got 1D array instead`. This is the batch-dimension problem addressed by the existing `fix/issue-13-spark-faiss-prefit-batches` branch. Issue 13 was **not** in the requested list, so that branch was not silently merged and its fix was not duplicated here. The successful custom Matching cases use different FAISS paths and do not remove this prefit failure.

## Observations not classified as defects

- Forty independent Matching runs per backend gave mean ATE **2.2324** on pandas and **2.2340** on Spark for an injected effect of 2.0. ATE SD was **0.1864/0.1865**, and 95%-CI coverage was **85% on both backends**. The estimates passed the predeclared calibration bounds. Coverage below 95% is worth a larger simulation, but 40 runs and residual matching bias are insufficient to declare a standard-error bug. Single-run CI misses encountered earlier were deliberately discarded as evidence.
- Pandas numeric AA rejection rates were 3–5%; Spark numeric AA rates were 2.5–5.5%. No numeric AA calibration defect was indicated.
- Welch AB p-values and group means agreed with independent references on both backends. Natural group ordering, Bonferroni correction, CUPED, and categorical Spark references passed.
- The behavior harness exits nonzero while known defect cases remain failing. This is intentional; its JSON differentiates those failures from all passing cases.

## Reproduction

```bash
# Fast project suite
.venv/bin/pytest -m "not spark" -q

# Synthetic pandas matrix
.venv/bin/python examples/experiments/alpha_agent_behavior_check.py \
  --backend pandas --aa-iterations 200 \
  --output examples/experiments/alpha_agent_behavior_pandas_results.json

# Synthetic Spark matrix
export JAVA_HOME=/opt/homebrew/opt/openjdk@11
export PATH="$JAVA_HOME/bin:$PATH"
export SPARK_LOCAL_IP=127.0.0.1
export PYSPARK_PYTHON="$PWD/.venv/bin/python"
export PYSPARK_DRIVER_PYTHON="$PWD/.venv/bin/python"
.venv/bin/python examples/experiments/alpha_agent_behavior_check.py \
  --backend spark --aa-iterations 200 \
  --output examples/experiments/alpha_agent_behavior_spark_results.json
```
