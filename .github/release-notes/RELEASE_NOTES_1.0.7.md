# HypEx 1.0.7

First release marked as production/stable. It brings back the KS-test in A/B reports, adds a single `summary` view over all result tables of an experiment, and fixes a batch of A/A, matching and output bugs found after 1.0.6.

## ⚠️ Breaking changes
- `result.cupac` exists only when the A/B test actually ran CUPAC (`enable_cupac=True`); otherwise it raises `AttributeError`. It used to be a placeholder object with `variance_reductions = None`, so code shaped like `if result.cupac.variance_reductions is None:` has to become `if "cupac" not in result.outputs:` (or `getattr(result, "cupac", None)`).
- A value in a `ConstGroupRole` column that is neither `control`, a group of the split, nor a label standing for a missing value now raises `ValueError` instead of quietly joining `test_1`.
- The verdict column of `result.multitest` is called `H0 rejected` instead of `rejected`.

## ✨ New features

### `result.summary` - all relevant tables at once
- Every experiment result now exposes `result.summary`: the main tables (`resume`, `multitest`, `sizes`, ...) together with the tables of each additional output (`cupac.resume`, `cupac.variance_reductions`, `cupac.feature_importances`, `cuped.resume`, ...).
- Prints as titled sections in a console and renders as tables in Jupyter.
- Behaves like an ordered mapping: `list(result.summary)`, `len(result.summary)`, `"cupac.resume" in result.summary`, `result.summary["cupac.feature_importances"]`.
- A section name is the attribute path of the same table, so `result.summary["cupac.feature_importances"]` is `result.cupac.feature_importances`.
- Available on a single output as well: `result.cupac.summary`, `result.main_output.summary`.

### KS-test is back in the A/B report
- `ABDictReporter` runs the KS-test again, so the A/B `resume` has `KSTest pass` and `KSTest p-value` columns next to the T-test ones.

## 🐛 Fixes

### Outputs
- `result.cupac` returned a one-line placeholder (`CupacOutput(variance_reductions: 1 target(s), ...)`) instead of the real CUPAC output: the legacy stub in `ui/ab.py` shadowed the output registered by `ABTest`. The stub and its duplicated extraction code are removed, so `result.cupac` now shows `resume`, `variance_reductions` and `feature_importances` as tables (in Jupyter too). `result.cupac.variance_reductions` and `result.cupac.feature_importances` keep working as before; `result.cupac` is present only with `enable_cupac=True`.

### A/B multiple testing
- The `multitest` table mixed up its labels: rows come metric-major while `field` and `group` were filled group-major, so with more than one metric and more than one test group every row except the first and the last described the wrong comparison - `new p-value` and `rejected` included. The group of a p-value now travels with the p-value itself, and the table carries it as an explicit `group` column next to `field`.
- `alpha` never reached the correction: `ABAnalyzer(alpha=...)` was dropped on the way to `multipletests`, which always rejected at 0.05.
- The verdict column is renamed to `H0 rejected` and documented: it tells whether the null hypothesis of that comparison - the groups do not differ - is rejected at `alpha` once the correction is applied. `False` means there is not enough evidence against it, not that the groups are the same.
- The correction is applied within each statistical test separately. A metric checked by both a t-test and a u-test used to land in one family and inflate its own correction twice over.
- The per-group aggregates of `ABAnalyzer` (`TTest p-value 1`, `TTest pass 2`, ...) were transposed the same way: `TTest p-value 1` was the mean over the first *metric*, not over the first group. They are now averaged over the rows of their own group.

### A/A test
- Stratified split (`stratification=True`) assigned groups in `groupby` order instead of the original row order, i.e. to the wrong rows - the A/A `resume` numbers for stratified runs were wrong. Stratifying over a column with more than two categories crashed with `ValueError`.
- `AATest(sample_size=..., groups_sizes=...)` crashed with `TypeError: cannot convert dictionary update sequence element #0 to a sequence` - the second pass got its arguments shifted.
- Constant groups (`ConstGroupRole`) are decoded instead of being guessed: `control`, `test` and `test_N` put a pinned row into that very group. `test_2` used to be silently merged into `test_1`, since only the literal `control` was ever read and everything else fell through to the default group. An unrecognised value now raises with the list of accepted labels - which also catches the classic `np.where(mask, 'test', np.nan)`, where `np.nan` becomes the string `'nan'`.
- A label that stands for a missing value (`'nan'`, `'None'`, `'NaT'`, `'<NA>'`, `''`) means the row is not pinned. Assigning a string into a column that does not exist yet makes pandas fill every row the mask does not cover with exactly that string - `df.loc[df['treat'] == 0, 'grp'] = 'control'` leaves `'nan'` in all the other rows, the pattern of the A/A tutorial - and those rows used to be pinned to `test_1` without a word. They take part in the split now, so an A/A run over such a column returns different groups than in 1.0.6.
- A dataset where every row is already pinned to a constant group is a valid input now - the constant assignment is the split - instead of failing with `IndexError: single positional indexer is out-of-bounds` on an empty slice. The control size can no longer become negative when most rows are pinned, and the labels of the groups that stayed empty are no longer lost.
- Known limitations of constant groups: `groups_sizes` is applied to the free rows only and is not compensated for the pinned ones (unlike `control_size`), so pinned rows skew the resulting proportions; and a stratified split (`AATest(stratification=True)`) ignores the `ConstGroupRole` column altogether - pinned rows are split at random like any other, and an unrecognised label is not reported there.

### Matching
- `GroupExperiment` took `group[0]` unconditionally, and on pandas < 2.0 (scalar `groupby` keys) that sliced the first character out of the key: all groups collapsed into one and only the last one survived in the `resume`. Group keys are now handled for both pandas generations, and a group whose rows cannot be found in the source data raises an explicit error instead of silently mismatching indexes.
- Matching no longer writes `matched_indexes.csv` into the working directory on every run; the matched indexes stay in `result.indexes`.

## 🧰 Internal / tooling
- `Development Status` classifier moved from `4 - Beta` to `5 - Production/Stable`.
- New tests: `tests/test_summary.py`, `tests/test_aa_params.py`, `tests/test_group_matching.py`.
- `Summary` is exported from `hypex.ui`.

---

**Full changelog:** `v1.0.6...v1.0.7`
