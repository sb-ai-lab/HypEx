# HypEx 2.0.0a1

First alpha of HypEx 2.0. It adds an Apache Spark backend next to pandas, makes experiments substantially faster, and extends the statistical toolkit. The API may still change before the stable release.

```bash
pip install --pre -U hypex
# or pin the alpha
pip install hypex==2.0.0a1
```

Without `--pre` pip keeps installing the latest stable release (1.0.7).

## ⚠️ Breaking changes (compared to 1.0.7)
- **Python 3.13 is not supported** (`python >=3.8, <3.13`; 1.0.7 allowed `<3.14`). PySpark 3.5 imports `np.NaN`, which NumPy 2 removed, and NumPy has no 3.13 wheels below 2.1.
- **`PySpark 3.5.1` is now a hard dependency** (it is installed together with the library, even if you only use pandas). Running on Spark also needs Java.
- **A/A splits differ from 1.0.7.** The deterministic split hash on pandas is now SipHash, so the same data and the same seed give different groups than before. Spark uses its own hash function, so the same split is not reproduced across backends.
- **`Matching` no longer extracts full data and no longer computes matched indexes by default** (`extract_full_data=False`, `compute_indexes=False`). Code that reads `result.full_data` or `result.indexes` has to pass `extract_full_data=True` / `compute_indexes=True`.
- **NaN handling in the scipy-based tests follows `nan_policy`, and the KS-test defaults to `"omit"`:** rows with missing values are dropped from the comparison, so p-values over columns with NaN can differ from 1.0.7.
- **`Output.resume` is renamed to `Output.summary`**; `repr` and the Jupyter view of an output now show the full report with every table. `resume` and `resume_reporter` stay as deprecated aliases and emit `DeprecationWarning`. The legacy `ABDatasetReporter` initialisation warns as well.
- **`Dataset.data` returns a `pyspark.sql.DataFrame` on the Spark backend**; the pandas-on-Spark object is available as `Dataset.raw_data`.
- Version 0.1.x remains unsupported (`pip install hypex==0.1.10` if you need it).

## 🚀 Performance
Computation is substantially faster than in 1.0.7, most of all on large data, in A/A loops and in matching. Benchmarks of 2.0 against 1.0.x are not published yet, so no figures are given here. The main reasons:
- **Spark backend** with checkpointing, so DAG lineage no longer grows inside iterative A/A loops, and with fewer Spark actions (the free group size and the split edges are computed without extra `count`).
- **Hashing and splitting:** the pandas split hash is vectorised.
- **KS-test in A/A** is sped up, and the statistical tests are computed in batches by `StatsComparator` instead of one at a time.
- **Experiment loops** (`ParamsExperiment`, `CycledExperiment`, `IfParamsExperiment`) compute loop invariants once instead of on every iteration; `Float32Caster` is applied once before the A/A iterations.
- **FAISS matching:** a `shuffle` mode with cluster-based partitioning for distributed nearest-neighbour search, and a co-partitioned search that replaces pulling every partition index to the driver (the old path ran out of memory on a 100M-row dataset).
- **Lazy Matching output:** indexes and full data are extracted only on request.

## ✨ New features

### Spark backend
- `Dataset` runs on pandas or Spark with the same API for `AATest`, `ABTest`, `HomogeneityTest` and `Matching` (`SparkDataset`, `GroupedDataset`, Spark-aware reporters and outputs).
- Spark support in `FaissExtension`, `DummyEncoderExtension`, the matching metrics and the statistical extensions.
- Session and tuning helpers in `hypex.utils.spark_config`.

### Statistics
- **`StatsUTest`:** Mann-Whitney U test computed from histograms, wired into the A/A, A/B and matching analyzers and reporters.
- **`BiasExtension`** (linear bias correction for matching) and **`MatchingMetricsExtension`** (treatment effect estimation) with NaN-safe handling of empty groups and invalid observations.
- Confidence-interval metrics in the A/B dictionary reporter.

### Pipeline building blocks
- **`NaDropper`** (row-level NaN filtering) and **`Float32Caster`** (memory saving via float32 downcasting), enabled in the default A/A and A/B pipelines.
- `AATest(dry_test=...)` and the `AADryTestAnalyzer` for validating an A/A setup without a full run.
- `Matching(extract_full_data=..., compute_indexes=...)` to control what the result carries.
- FAISS configuration through `hypex.config` (`FAISS_FIT_MODE`, `FAISS_N_PROBES`, `BUCKET_SIZE`).
- Logging and profiling utilities (`hypex.utils.logger`, `hypex.utils.profiling`) and a strict ABC base class that validates executor overrides.

### Documentation
- A `README.md` in every module of the package and refreshed tutorials.

## 🐛 Fixes
- **Spark:** deterministic group order, correct FAISS neighbour ids for `k=1`, initialisation of the caching index from the configuration, fixes in `SparkDataset`.
- **Matching:** index extraction for nested, flat and single indexes on pandas and Spark; groups of matched pairs; the "no valid matches" case in `BiasExtension` and the matching metrics.
- **Encoders:** rows with an underscore in string values.
- **A/A:** the test order in the pass reporters includes `UTest`; `Float32Caster` keeps the original `data_type` in roles, so `search_columns` keeps working after downcasting.
- **Chi2:** index duplication in the Chi2 extension.
- User data is no longer mutated by `ExperimentShell` on either backend.

## 🧰 Internal / tooling
- A new CI pipeline (lint, spelling, typing, test matrix, docs, build) and a release workflow with Trusted Publishing; CodeQL is enabled with the `security-and-quality` suite.
- The test suite grew from the example tutorials to unit and integration tests (about 105 test files changed, coverage gate at 88%).
- `Development Status` classifier is `3 - Alpha`.

## Known limitations
- Python 3.13 is not supported yet; this will be revisited with PySpark 4.
- Alpha status: some internal modules still carry TODOs, and the public API may change before 2.0.0.
- `UTest` is registered for pandas only; on Spark use `StatsUTest`.

---

**Full changelog:** `v1.0.7...v2.0.0a1`
