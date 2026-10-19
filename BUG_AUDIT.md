# Hidden-bugs audit — `dev/master` (2026-10-19)

Audit branch: `review/hidden-bugs-audit` (off `dev/master`; `master` is 578 commits behind and contains
nothing that is not in `dev/master`).

Method: every finding below was **reproduced by execution** against this working tree (pandas backend,
and a local Spark 3.5.1 session where noted). Findings already encoded as `xfail(strict=True)` in
`tests/` (see `tests/KNOWN_LIBRARY_BUGS.md`) are excluded and listed at the end for completeness.
One candidate finding (faiss `_prepare_indexes` tie-width crash) was **rejected as a false positive**
after tracing the real call path — see "Rejected candidates".

---

## Critical — silently wrong statistical results in core flows

### C1. Group-order / labeling swaps across comparators and the AB analyzer
Numeric group keys with different digit counts (e.g. `2` vs `10`, or groups `{1, 2, 10}`) are ordered
differently by a string sort and by natural/value sort, and different code paths use different orders:

| Site | Order used | Effect |
|---|---|---|
| `hypex/comparators/abstract.py:218` (`GroupsComparator._grouping_data_split`) | `sorted(..., key=str)` | **Baseline (control) becomes the numerically largest key.** `GroupTTest` on groups `2`/`10` returns statistic `+25.26` where the correct control-first statistic is `-25.26`. |
| `hypex/comparators/abstract.py:1096` (`StatsComparator.execute`) vs `:921` (`_execute_inner_function`) | index: `sorted(key=str)`; values: dict order | Row *i* is labeled with the wrong compared group. `StatsTTest` on `2`/`10`: row labeled `2┆y` contains the group-10-as-baseline comparison. |
| `hypex/extensions/stats_hypothesis_testing.py` (`StatsChi2TestExtension._calc_spark`, per-column dict order) vs `abstract.py:1136-1140` (index built from `target_cols`) | Spark collect arrival order vs `target_cols` | Per-column results are reported under the **wrong column labels**. Verified: y1 (p≈7e-89) reported under the `y3` label and y3 (p≈0.28) under the `y1` label. |
| `hypex/analyzers/ab.py:239-247` (`_build_row_index`, iterative mode) | `group_labels` from `data.groups` (natural order) zipped onto rows that are str-sorted | End-to-end `ABTest()` with treatment keys `{1,2,10}` (true means 1→0.03, 2→8.0, 10→−0.01) publishes: group `10` row shows test mean **8.00** (group 2's mean) and group `2` row shows **−0.01** (group 10's mean) — means and p-values land on the wrong group rows. |

Note: the merge commit `7cdae829` (spark backend) itself records *"T8 (ABTest spark vs pandas)
currently fails: pre-existing control/test swap on Spark"* — the family was observed but never fixed.
The guard tests in `tests/dataset_spark_backend/test_group_order.py` pin aggregation order only.

**Fix direction:** pick one canonical ordering (ascending by value, nulls last — what
`_sort_by_group_key` does) and use it in all four sites; never re-sort with `key=str` after the fact.

### C2. `MultitestQuantile` standardizes by total sample size instead of per-group size
`hypex/extensions/statsmodels.py:178, 196-202` — `sample_size = len(data)` is the total across all
groups, but the simulated null quantiles (`quantile_of_marginal_distribution`) correspond to
`sqrt(n_per_group)·Δ/√(v_j+v_i)` ~ N(0,1). The mismatch inflates the statistic by `sqrt(n_groups)`.

Measured (real class, H0 true, `equal_variance=True`):
- k=2, n=100/group → "winner" declared in **22%** of runs (nominal 5%)
- k=4, n=80/group → **27.3%**
- emulating the fix (`len(data) // num_samples`) → **6.3%**

Reachable via `ABTest(multitest_method="quantile")`; end-to-end run on pure-H0 data declared a winner
in 10/20 trials. **Fix:** per-group size (or per-pair sizes with the matching SE formula).

### C3. CUPAC prediction/index misalignment flips variance reduction into variance increase
`hypex/extensions/cupac.py::_predict_pandas` wraps `model.predict` output in a fresh `RangeIndex`;
`hypex/ml/cupac.py:407-408` then does index-aligned arithmetic against `data.ds[target]`, which may
carry a user (filtered/shuffled) index. Each prediction is paired with the **wrong row**.

Measured: y = 3x + noise, non-default row order → variance 9.49 before, **17.75 after** "adjustment"
(aligned index: 0.09). `CUPEDTransformer` is unaffected (both operands share the index).
**Fix:** build the prediction frame on `X_df`'s index.

Related distinct defect: `hypex/ml/cupac.py:143` — `sorted(searched_lags, key=lambda x: x[1])` raises
`TypeError: '<' not supported between instances of 'int' and 'NoneType'` when a `FeatureRole()` without
lag coexists with a lagged feature (reproduced). The code below the sort explicitly handles
`lag in [None, 0]`, so the sort key just needs to be None-aware.

### C4. Pandas chi2 contingency table drops categories present in only one group
`hypex/extensions/scipy_stats.py:271` — `merge(on=col)` defaults to `how="inner"`; a category occurring
in only one sample is removed from the table before `chi2_contingency`. The trailing `.fillna(0)` is
dead code under an inner join, i.e. an outer join was intended.

Measured: control = 30×"u"+10×"v", test = 20×"u"+20×"v"+10×"rare" → table 2×2, p=0.192; correct 2×3
table → p=0.00075 (decision flips at α=0.05). Degenerate case: control 100×"x"+100×"y" vs 100×"y"+
100×"z" → HypEx returns p=1.0 where scipy gives ~3.7e-44.

Note: the proportion-scaling defect of the same method is already xfail'd
(`test_chi2_matches_scipy_contingency`); this merge facet is **not** covered by that xfail and survives
even if the scaling were fixed.

---

## High

### H1. `ABAnalyzer` never collects `GroupChi2Test` results (pandas backend)
`hypex/analyzers/ab.py:300-310` lists `StatsChi2Test` (the Spark path) but not `GroupChi2Test`
(the pandas path). Verified: `ABTest(additional_tests=["chi2-test"])` on 3 groups →
`GroupChi2Test┴┴purchased` exists in `analysis_tables` but `ABAnalyzer┴┴` is **empty** and
`res.multitest` says correction was not applied; chi2 p-values never receive Holm/Bonferroni/FDR.
The sibling `OneAAStatAnalyzer` (`hypex/analyzers/aa.py:96-108`) does list `GroupChi2Test`, confirming
the omission is unintentional.

Related: `execute_multitest` gate (`analyzers/ab.py:125-131`) counts only numeric `TargetRole`
columns, so categorical-target comparisons are uncorrected even when collected; and with mixed
numeric + categorical targets the multitest call crashes (`multipletests` receives string-dtype
values → `UFuncNoLoopError`; reproduced).

### H2. `ABTest(multitest_method=<enum member>)` silently applies Holm
`hypex/ab.py:81-87` tests `multitest_method in ABNTestMethodsEnum._value2member_map_`, which is keyed
by string values. An enum member (documented in the class docstring and usage example) fails the
membership test and falls back to Holm. Verified: `ABNTestMethodsEnum.bonferroni` → Holm-adjusted
p-values; string `"bonferroni"` → correct. Fix: accept `isinstance(m, ABNTestMethodsEnum)` before the
value lookup.

### H3. Pandas FAISS IVF index never sets `nprobe` (default 1) — silent recall loss
`hypex/extensions/faiss.py:313-329` builds `IndexIVFFlat` and never sets `nprobe`; FAISS defaults to 1.
The Spark path sets a tuned value (`faiss.py:1061,1083`). Measured on 20k×4: recall@5 = **0.753** with
`nprobe=1` vs **1.0** with `nprobe=10` (nlist=70). Reached via `Matching(faiss_mode="fast")` with
>1000 control rows or `faiss_mode="auto"` with >1M rows. Matches are silently approximate → biased
effect estimates. Fix: set the same heuristic nprobe used on Spark.

### H4. Spark FAISS shuffle-mode prefit crashes on small partitions
`hypex/extensions/faiss.py:855-920` (`_prefit` → `MiniBatchKMeans.partial_fit(np_batch)`): a batch
containing fewer rows than `k = sqrt(N/4)` clusters raises
`ValueError: n_samples=2 should be >= n_clusters=3`. This is why
`tests/unit/test_ml_faiss.py::test_spark_extension_matches_brute_force` **fails** with the
`local[*]` session fixture on this machine (40 rows over 14 partitions → 2-row partitions); it passes
with `local[2]`, which is why CI (2-core Linux runners) does not catch it. Trigger condition:
number of partitions ≳ sqrt(4N). Fix: accumulate batches to ≥ k rows before `partial_fit`
(and/or cap k by batch size).

### H5. Spark `value_counts` silently counts only the first column; `normalize=True` crashes
`hypex/dataset/backends/spark_backend.py:1743-1755`:
- multi-column input → only the first column is counted (verified: pandas returns `['x','y','count']`,
  Spark returns `['x','count']` — silent data loss);
- `normalize=True` → the proportions column is named `"count"` while the abstract layer expects
  `"proportion"` → `RoleColumnError` (verified).

### H6. `Dataset.loc[single_label]` crashes on Spark
`spark_backend.py:1206-1217` returns the raw `ps.Series`; the `Locker` then accesses `.columns`
(`abstract.py:59`). Verified: `AttributeError: 'Series' object has no attribute 'columns'`.
Public API, hard crash, pandas-only failure.

---

## Medium

### M1. `ExperimentData.check_hash(..., ExperimentDataEnum.groups)` → infinite recursion
`hypex/dataset/experiment_data.py:193-204` has no branch for `groups`, but `groups` ∈
`_SUPPORTED_SPACES`, so the "safe fallback" re-enters itself. Verified: `RecursionError`. Fix:
return `exec_id_str in self.groups` (and exclude `space` from the fallback loop).

### M2. `ExperimentData.copy(data=...)` takes the replacement dataset by reference
`experiment_data.py:73-93` — the docstring promises a deep copy, but `result._data = data` aliases.
Verified: `ed2.ds is new_ds`; `add_column` through the copy mutates the caller's dataset. All current
in-library callers pass freshly built datasets, so it is a latent contract bug.

### M3. `ABCRole.__copy__` drops `parent`/`lag`/`cofounders`
`hypex/dataset/roles.py:34-44` rebuilds via the base constructor. Verified:
`copy.copy(PreTargetRole(parent="y", lag=1, cofounders=["x"]))` → all metadata lost; the same loss
applies to `astype()`/`asadditional()`, which `DummyEncoderExtension._build_roles` uses — dummy columns
derived from lagged/confounder-tagged features lose their lineage for CUPAC.

### M4. Documented `Matching` quality tests `"smd"`, `"psi"`, `"repeats"` crash
`hypex/matching.py:126-143` — `test_mapping` has no entries for them (PSI commented out; SMD never
wired; "repeats" unimplemented). Verified: `Matching(quality_tests="psi")` → `KeyError 'p'` (same for
the others), while the docstring advertises them.

### M5. `PandasDataset.remove_index_col` drops rows instead of the column
`pandas_backend.py:1039-1040` — `drop(name)` without `axis=1`. Verified: `KeyError` after
`add_index_col()`. Latent (no in-repo callers).

### M6. `Dataset.value_counts` on pandas vs Spark also diverges in role wiring
(covered by H5; listed here for the pandas side being the reference behavior).

### M7. `_set_additional_fields` unpersists the cache it just created (Spark)
`experiment_data.py:259-293` — for the in-place `add_column` branches `new_data is self._data`, so the
second `unpersist()` removes the cache `persist()` just established. Verified: `is_persisted` goes
True → False across `set_value`. Performance/caching correctness only; results unaffected.

### M8. A feature literally named `difference` is silently excluded from AA scoring
`hypex/utils/naming.py::_parse_metric_col` matches the `"difference"` metric suffix at position 0 of
`"difference TTest pass test_1"`, `_resolve_column_parts` then returns `None`, and the column is
skipped by `AAScoreAnalyzer` and best-split selection. Anchor the metric match to the column tail.

---

## Design notes (not counted as bugs, worth a decision)

- `hypex/extensions/scipy_linalg.py:15-27` — "pooled" covariance is the plain average of the two group
  covariances, not the n-weighted pooled covariance; with unbalanced groups the metric is dominated by
  the minority group. Valid choice if intended; document it.
- `hypex/extensions/scipy_linalg.py:30-47` — Cholesky regularization `+εI` with absolute `ε=1e-3`
  breaks Mahalanobis scale invariance (feature units change the effective metric). A scale-relative ε
  (e.g. `ε·trace/d`) would be invariant.
- Spark KS/U are histogram approximations (documented in `KNOWN_LIBRARY_BUGS.md`); numeric deltas vs
  scipy on outlier-stretched ranges are expected, not bugs.

---

## Rejected candidates (false positives caught during verification)

- **`_prepare_indexes` tie-width `np.vstack` crash** (`hypex/extensions/faiss.py:227-253`): the crash
  needs a `dist` row width ≠ k, but `_predict` calls `search(X, k=self.n_neighbors)` and feeds the
  result straight in — with exactly k candidate slots the summed tie widths are always k
  (verified exhaustively for k=2..5 and end-to-end via `Matching(n_neighbors=2/3)` on duplicated
  control rows). The proposed counterexample used 3 candidate columns with k=2, which the real call
  path cannot produce.

## Known issues excluded from this report (already `xfail(strict=True)` in `tests/`)

SMD formula (`test_operators.py:32`), AA-score pass-rate iteration bug (`test_analyzers.py:206`),
faiss `predict`-mode query matrix (`test_ml_faiss.py:158`), chi2 proportion scaling
(`test_hypothesis_testing.py:131`), `multitest_method=None`/unknown-name → Holm
(`test_ab_scenario.py`), `fdr_tsbhy` enum gap (`test_enums.py:99`), CUPAC e2e dispatch crash
(`test_ab_scenario.py:214`), AATest input mutation, Spark agg/std/var addressing, Spark hash-label
divergence, `labels_dict` NotImplementedError, Spark faiss index drop — see
`tests/KNOWN_LIBRARY_BUGS.md`.
