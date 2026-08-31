# HypEx 1.0.6

This release focuses on variance-reduction tooling for A/B tests (CUPED & CUPAC), faster A/A homogeneity search, and a reworked output layer that can surface several result tables from a single experiment.

## ✨ New features

### CUPED & CUPAC variance reduction in `ABTest`
- `ABTest` now accepts `cuped_features`, `cupac_models`, and `enable_cupac` to reduce variance using pre-experiment data.
  - `cuped_features`: `{target_feature: pre_target_feature}` mapping for CUPED adjustment.
  - `enable_cupac=True` + `cupac_models` (`'linear'`, `'ridge'`, `'lasso'`, `'catboost'`, or a list): trains models on pre-period covariates and picks the best by variance reduction (CUPAC config is read from `dataset.features_mapping`).
- New dedicated reporters and UI outputs (`CupedOutput`, `CupacOutput`) expose the adjustment results as additional tables alongside the main A/B result.
- Corrected CUPED/CUPAC theta formula (#233) for accurate covariate adjustment.

### A/A test early stopping
- `AATest` gains an `early_stopping` flag. When enabled, the search stops at the first "clean" split (no test flags any feature), making runs significantly faster.
  - Note: with early stopping the aggregate per-feature AA score (empirical type-1 error) is no longer a homogeneity guarantee. If no clean split is found within `n_iterations`, the full run is kept and the best split is selected.

### Multi-output experiments
- New `ExperimentOutput` layer lets a single experiment return a main result plus additional named outputs. `ABTest`, `AATest`, and `Matching` now route through it, enabling the CUPED/CUPAC tables described above (#199).

### Matching improvements
- `n_neighbors` support for k-nearest-neighbour matching.
- Custom feature `weights` for the matching distance metric.
- Minimum sample size estimation utilities added across the matching/extensions stack.

## 🐛 Fixes
- Fixed an "ABN out of range" error.
- Fixed CUPED and CUPAC formulas.
- Fixed division-by-zero with constant groups in matching.
- A/B multitest correction fixes.
- Removed automatic KS-test from the A/B flow; removed persisted matched-index arrays from output.
- Added `from __future__ import annotations` for compatibility with older Python versions.

## 🧰 Internal / tooling
- New test suites for CUPED and CUPAC (`tests/test_cuped.py`, `tests/test_cupac.py`).
- CI now triggers on PRs to `dev/master`; tox configuration stabilized across Python versions.
- Updated AA/AB/Matching tutorials to cover the new variance-reduction and early-stopping options.

---

**Full changelog:** `v1.0.5...v1.0.6`
