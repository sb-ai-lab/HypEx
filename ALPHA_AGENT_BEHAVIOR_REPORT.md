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

- Pandas: **14/14 cases passed** after the follow-up fixes.
- Spark: **14/14 cases passed** after the follow-up fixes.
- Total: **28/28 cases passed**. The original failures and their resolutions are documented below.

## Integration correction made during merging

The keyless-index merge fix initially forced `left_index=True` and `right_index=True` even for keyed calls such as `merge(on="category")`. This broke six existing chi-square tests. Both backends now infer an index join only when no column or index keys are supplied, and convert remaining `None` index flags to `False`.

A backend-parametrized regression test was added in `tests/unit/dataset/test_dataset_indexing.py`. The three merge conflicts were test-only; all tests from both sides were retained.

The included issue-15 Spark scalar-`.loc` fix also needed a follow-up. In this environment, scalar Spark `.loc` returns a local **pandas** Series, but the branch converted only `pyspark.pandas.Series`. `SparkNavigation.loc` now converts either Series type with `.to_frame()` (`hypex/dataset/backends/spark_backend.py:1216-1219`). The scalar result is consequently a pandas-backed frame, matching the local Series returned by the upstream scalar operation and avoiding mixed-type Arrow conversion errors.

Verification after these corrections:

- `pytest -m "not spark" -q`: **1273 passed, 54 xfailed**.
- Targeted pandas/Spark merge and ordering tests: **12 passed**.
- Final targeted scalar-`.loc` and merge tests: **6 passed**.
- Complete `pytest -q` after all follow-up fixes: **1767 passed, 18 skipped, 78 xfailed; 0 failed**.
- Direct Ruff equivalents of the lint environment: **all checks passed; 207 files formatted**. The `tox` executable was unavailable.

## Defects confirmed during testing and resolved in follow-up branches

### 1. Pandas chi-square used scaled counts and was conservative under the null

**Affected paths:** pandas AA, AB categorical tests, Homogeneity categorical tests, and Matching categorical quality tests.

**Original evidence**

- Over 200 null AA splits, categorical chi-square rejected **0/200** on pandas versus **8/200 (4%)** on Spark.
- The pandas p-values had mean **0.625434** and a KS-uniformity p-value of **1.965e-9**. Spark had mean **0.491466** and uniformity p-value **0.703327**.
- On the deterministic categorical AB dataset, pandas returned p=`2.934626e-11`; raw-count SciPy returned p=`4.662643e-22`. Spark matched SciPy.
- Minimal reproduction with raw counts `[[80, 20], [60, 40]]`:
  - raw-count SciPy: statistic `8.595238`, p=`0.00337043`;
  - original HypEx pandas table `[[40, 30], [10, 20]]`: statistic `3.857143`, p=`0.04953461`.

**Confirmed cause**

`PandasChi2TestExtension.matrix_preparation` multiplied each group's counts by the opposite group's fraction. For equal groups this halved the effective sample size. `scipy.stats.chi2_contingency` expects observed counts, which the Spark implementation already used.

**Resolution**

`fix/issue-27-pandas-chi2-raw-counts` removes the scaling and passes observed counts on pandas. The strict-xfail SciPy reference test is now a normal passing test. Post-fix categorical AA rejection is **11/200 (5.5%)**, and the categorical AB p-value exactly matches SciPy.

### 2. Spark AA failed whenever `sample_size < 1`

**Original failure**

```python
AATest(n_iterations=1, sample_size=0.75, random_states=[0]).execute(spark_ds)
```

failed in `GroupSizes` with:

```text
TypeError: '<' not supported between instances of 'NoneType' and 'str'
```

**Evidence and confirmed cause**

- Stratification alone passed with `control_size` 0.5 and 0.3; subsampling failed independently of stratification, control size, and `float32`.
- In a 100-row reproduction, the splitter produced 44 control rows, 39 test rows, and **17 null labels**. The dataset still contained all 100 rows after additional-field alignment.
- Spark group statistics included the null key, and sorting mixed `None` with strings failed.

**Resolution**

`fix/issue-28-spark-aa-sample-size-null-groups` drops unlabeled rows in comparator inputs for ordinary statistics, chi-square counts, and KS/U histograms. Low-level extensions still preserve explicitly requested NaN-group ordering. The fix passed 40 relevant Spark tests, including both pre-existing NaN-ordering contracts and a mixed numeric/categorical subsampled AA regression.

### 3. `Matching(quality_tests="t-test")` iterated the string as characters

**Original failure:** both backends raised `KeyError: 't'` because the non-`auto` branch iterated the scalar string directly instead of normalizing it to a one-element list.

**Resolution**

`fix/issue-29-matching-scalar-quality-tests` normalizes a scalar string before lookup. An end-to-end regression verifies identical scalar/list results on pandas and Spark.

## Additional issue-13 failure resolved during follow-up

The complete suite originally reproduced three Spark FAISS failures in `test_matching_spark_matches_pandas[1]`, `[2]`, and `test_spark_extension_matches_brute_force`. All reached `MiniBatchKMeans.partial_fit` with a one-dimensional partition array because `np.append` flattened cached feature matrices.

The existing `fix/issue-13-spark-faiss-prefit-batches` branch accumulates arrays in a list and concatenates them along axis 0. It passed all four targeted prefit/FAISS/Matching tests unchanged and was included in the final integration verification.

## Observations not classified as defects

- Forty independent Matching runs per backend gave mean ATE **2.2324** on pandas and **2.2340** on Spark for an injected effect of 2.0. ATE SD was **0.1864/0.1865**, and 95%-CI coverage was **85% on both backends**. The estimates passed the predeclared calibration bounds. Coverage below 95% is worth a larger simulation, but 40 runs and residual matching bias are insufficient to declare a standard-error bug. Single-run CI misses encountered earlier were deliberately discarded as evidence.
- Pandas numeric AA rejection rates were 3–5%; Spark numeric AA rates were 2.5–5.5%. No numeric AA calibration defect was indicated.
- Welch AB p-values and group means agreed with independent references on both backends. Natural group ordering, Bonferroni correction, CUPED, and categorical Spark references passed.
- The behavior harness exits nonzero if any behavior case fails. Its JSON retains per-case metrics and tracebacks; the original failures shown above led to the separate fix branches documented in this report.

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
