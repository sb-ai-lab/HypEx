# HypEx 2.0.0

HypEx 2.0.0 is the stable release of the new HypEx architecture. It adds an Apache Spark backend next to pandas, improves performance on large datasets and iterative experiments, and extends the statistical toolkit.

```bash
pip install -U hypex
# or pin this release
pip install hypex==2.0.0
```

The base installation supports pandas. Install `hypex[spark]` for the Spark backend, `hypex[cat]` or `hypex[lgbm]` for optional CUPAC models, or `hypex[all]` for all extras.

## Breaking changes compared to 1.0.7

- **Python 3.13 is not supported** (`python >=3.8,<3.13`). Python 3.8 through 3.12 are supported.
- **PySpark is optional and is pinned to 3.5.7.** It is installed by the `spark` and `all` extras rather than with the base package. Running Spark also requires Java 8, 11, or 17.
- **A/A splits differ from 1.0.7.** The deterministic pandas split hash is now SipHash, so the same data and seed produce different groups. Spark uses its own hash function, and splits are not reproduced identically across backends.
- **`Matching` no longer extracts full data or matched indexes by default** (`extract_full_data=False`, `compute_indexes=False`). Pass the corresponding option when reading `result.full_data` or `result.indexes`.
- **NaN handling in scipy-based tests follows `nan_policy`, and the KS test defaults to `"omit"`.** Missing observations are excluded from comparisons, so p-values can differ from 1.0.7.
- **`Output.resume` is renamed to `Output.summary`.** The old `resume` and `resume_reporter` names remain deprecated aliases and emit `DeprecationWarning`.
- **`Dataset.data` returns a `pyspark.sql.DataFrame` on the Spark backend.** The pandas-on-Spark object is available as `Dataset.raw_data`.
- Version 0.1.x remains unsupported (`pip install hypex==0.1.10` if it is required).

## Performance

- Spark checkpointing limits lineage growth in iterative A/A pipelines and reduces unnecessary Spark actions.
- Pandas split hashing and statistical tests are vectorised or batched.
- Experiment loops compute invariant work once per run rather than once per iteration.
- FAISS matching supports distributed `shuffle` mode and co-partitioned search without collecting every partition index on the driver.
- Matching indexes and full data are extracted only on request.

## New features

### Spark backend

- `Dataset` runs on pandas or Spark with the same API for `AATest`, `ABTest`, `HomogeneityTest`, and `Matching`.
- Spark support covers FAISS matching, categorical encoding, matching metrics, statistical extensions, reporters, and outputs.
- Session and tuning helpers are available in `hypex.utils.spark_config`.
- The base pandas installation works without PySpark and provides actionable installation errors for Spark-only entry points.

### Statistics and experiments

- `StatsUTest` provides a histogram-based Mann–Whitney U test for A/A, A/B, and matching analyses.
- `BiasExtension` and `MatchingMetricsExtension` provide matching bias correction and treatment-effect estimation.
- A/B reports include confidence-interval metrics, multiple-testing corrections, and CUPED/CUPAC variance-reduction reports.
- `NaDropper` and `Float32Caster` are available as preprocessing blocks.
- `AATest(dry_test=...)` validates an A/A setup without a full experiment run.
- FAISS can be configured through `hypex.config`.

### Reliability and tooling

- CI covers linting, spelling, packaging, documentation, optional-dependency behaviour, and Python 3.8–3.12 test matrices.
- The full release test run completed with **1981 passed, 20 skipped, and 32 expected failures**.
- Measured statement coverage is **90.75%** (9,448 of 10,411 statements), above the required **88%** gate. The `factory` and `hypotheses` prototype packages are excluded by the existing coverage configuration.
- Release artifacts are checked for matching tag, package metadata, and runtime versions before publication.

## Fixes since 2.0.0a1

- PySpark moved from a mandatory dependency to an optional extra and was updated to 3.5.7.
- Spark aggregation, indexing, transformer execution, comparator split modes, and reporter utilities received correctness fixes.
- PSI handles constant inputs, SMD handles zero variance, and A/A score semantics and multitest types were aligned with their public contracts.
- CI actions and release checks were updated, and tutorial execution was added to the validation pipeline.

## Known limitations

- Python 3.13 is not supported yet.
- `UTest` is registered for pandas only; on Spark use `StatsUTest`.
- Spark tests on Windows with Python 3.12 are limited by a PySpark 3.5 worker crash. Non-Spark tests on that platform remain enabled; Linux runs the complete Spark suite.
- Benchmark results against 1.0.x are not published, so this release does not claim specific speed-up figures.

---

**Full changelog:** https://github.com/sb-ai-lab/HypEx/compare/v1.0.7...v2.0.0
