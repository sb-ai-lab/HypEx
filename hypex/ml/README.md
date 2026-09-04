# `hypex.ml` — ML-Based Executors

Pipeline blocks whose computation is a fitted model rather than a closed-form
statistic. Two of them today: nearest-neighbour matching and CUPAC variance
reduction.

## Role in the architecture

Both derive from `MLExecutor` (see [`../executor/README.md`](../executor/README.md)),
which supplies the fit-on-one-group / predict-on-the-other flow, and both write
their output into `ExperimentData.additional_fields` with
`AdditionalMatchingRole`.

```
Matching pipeline
  TypeCaster → DummyEncoder → MahalanobisDistance → FaissNearestNeighbors
                                                          │ matched index columns
                                                          ▼
                                          Bias → MatchingMetrics → MatchingAnalyzer

A/B pipeline with CUPAC
  CUPACExecutor  →  predicted covariate column  →  GroupTTest on the adjusted target
```

## File map

| File | Contents |
|---|---|
| `faiss.py` | `FaissNearestNeighbors`. |
| `cupac.py` | `CUPACExecutor`. |
| `__init__.py` | Exports both. |

The model code itself lives in `hypex/extensions/faiss.py` and
`hypex/extensions/cupac.py`; the sklearn model registry is in
`hypex/utils/models.py`.

## Key classes

### `FaissNearestNeighbors(MLExecutor)`

```python
FaissNearestNeighbors(
    n_neighbors: int = 1,
    two_sides: bool = False,   # match both directions (needed for ATE)
    test_pairs: bool = False,  # match control→test instead of test→control
    grouping_role: ABCRole | None = None,
    key: Any = "",
    faiss_mode: Literal["base", "fast", "auto"] = "auto",
)
```

* `target_role` is fixed to `FeatureRole()` — matching operates on features, not
  targets.
* `execute` looks for a `MahalanobisDistance` result cached in
  `ExperimentData.groups`; if present, matching runs in that whitened space
  instead of the raw feature space. This is how `Matching(distance="mahalanobis")`
  differs from `distance="l2"` without changing the matcher.
* `_execute_inner_function` decides the direction(s) to match based on
  `two_sides` and `test_pairs`, returning `{"test": ...}`, `{"control": ...}` or
  both.
* `_set_global_match_indexes` converts the group-local FAISS row numbers back into
  global dataset index values — the matched pairs are meaningful against the
  original data.
* `PairsNotFoundError` is raised when nothing could be matched.

Output: one `additional_fields` column per neighbour and direction, holding the
matched counterpart's index.

### `CUPACExecutor(MLExecutor)`

```python
CUPACExecutor(
    cupac_models: str | Sequence[str] | None = None,
    key: Any = "",
    n_folds: int = 5,
    random_state: int | None = None,
)
```

CUPAC (Control Using Predictions As Covariates): fit a model on pre-experiment
data to predict the target, then use the prediction as a control variate to
reduce variance — the ML generalisation of CUPED.

* `cupac_models` names entries in `CUPAC_MODELS` (`hypex/utils/models.py`):
  `"linear"` (`LinearRegression`), `"ridge"`, `"lasso"`, and `"catboost"` when
  catboost is installed. `None` tries all of them and keeps the best.
* `_validate_models` raises `ValueError` for an unknown model or one unavailable
  for the current backend.
* Delegates the cross-fitted training to `CupacExtension(n_folds, random_state)`,
  whose `"kfold_fit"` mode produces out-of-fold predictions so the covariate does
  not leak the target.
* Reads `PreTargetRole` / `FeatureRole` columns; writes the prediction as an
  additional field.

Reachable from the shell as `ABTest(enable_cupac=True, cupac_models=["linear",
"ridge"])`; the resulting diagnostics land in `ABOutput.cupac` (variance
reductions and feature importances).

## How to work with it

### Matching

```python
from hypex.ml import FaissNearestNeighbors
from hypex.comparators.distances import MahalanobisDistance
from hypex.dataset import TreatmentRole

Experiment(executors=[
    MahalanobisDistance(grouping_role=TreatmentRole()),
    FaissNearestNeighbors(grouping_role=TreatmentRole(), n_neighbors=1, two_sides=True),
])
```

or, through the shell:

```python
from hypex import Matching
Matching(distance="mahalanobis", metric="ate", n_neighbors=1).execute(data)
```

### CUPAC

```python
from hypex import ABTest
result = ABTest(enable_cupac=True, cupac_models=["linear", "ridge"]).execute(data)
result.cupac.variance_reductions
result.cupac.feature_importances
```

## How to add an ML executor

```python
class MyModelExecutor(MLExecutor):
    def __init__(self, grouping_role=None, key=""):
        super().__init__(grouping_role=grouping_role, target_role=TargetRole(), key=key)

    @classmethod
    def _inner_function(cls, data, test_data=None, target_data=None, **kwargs):
        return MyExtension().calc(data=data, test_data=test_data, **kwargs)

    def fit(self, X, Y=None):    return MyExtension().fit(X=X, Y=Y)
    def predict(self, X):        return MyExtension().predict(X)
```

`MLExecutor.calc` already groups the data, uses the first group as train and the
second as test, and `_set_value` already writes every output column to
`additional_fields`. Put the model itself in `hypex/extensions/`.

## Gotchas

* **`faiss` is a hard import.** `hypex/extensions/faiss.py` imports it at module
  level, so `import hypex.ml` requires the package to be installed.
* **Pandas only.** Both extensions implement `_calc_pandas` only; a Spark dataset
  fails on the backend lookup.
* **Direction flags interact.** `two_sides=False, test_pairs=False` matches
  test→control (supports ATT); `two_sides=True` matches both (needed for ATE).
  `MatchingMetrics(metric=...)` must agree with what was actually matched.
* **Encode and cast first.** FAISS needs a numeric, dense feature matrix —
  `TypeCaster` and `DummyEncoder` are not optional in a matching pipeline.
* Matching more than ~7 features invites the curse of dimensionality; the README
  in the repo root says so explicitly.

## Related modules

`../extensions/README.md` (FAISS and CUPAC implementations) ·
`../operators/README.md` (turns matched pairs into effect estimates) ·
`../comparators/README.md` (`MahalanobisDistance`) · `../executor/README.md`
(`MLExecutor`) · `hypex/utils/models.py` (the model registry).
