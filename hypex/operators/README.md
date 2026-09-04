# `hypex.operators` — Metric Operators

Blocks that compute a **metric across two target columns within each group**,
rather than comparing two groups on one column. This is the shape that causal
effect estimation needs: for a matched dataset, each row has both its own
outcome and its matched counterpart's outcome.

## Role in the architecture

Operators sit next to comparators — same level, different data shape — and are
the last computational step of the matching pipeline before `MatchingAnalyzer`
formats the numbers.

```
FaissNearestNeighbors  →  matched index pairs in additional_fields
        │
        ├─ Bias              → bias-correction terms (variables)
        └─ MatchingMetrics   → ATE / ATT / ATC + SE + p-value + CI (variables)
                                        │
                             MatchingAnalyzer → analysis_tables
                                        │
                             MatchingOutput.resume
```

Unlike comparators, operators write to **`ExperimentData.variables`**, not
`analysis_tables`.

## File map

| File | Contents |
|---|---|
| `abstract.py` | `GroupOperator` — the base class. |
| `operators.py` | `SMD`, `MatchingMetrics`, `Bias`. |
| `__init__.py` | Exports `SMD` only; import the others from `hypex.operators.operators`. |

## Key classes

### `GroupOperator(Calculator)`

`GroupOperator(grouping_role=GroupingRole(), target_roles=TargetRole(), key="")`

* `_get_fields(data)` resolves the grouping column and the target columns. It
  requires **exactly two** target fields; if only one is found under
  `target_roles`, it tops up from `AdditionalTargetRole()` — that is how the
  matched counterpart column (written by the matching step) is picked up.
* `calc(data, group_field=..., grouping_data=..., target_fields=...)` groups the
  data (or reuses a cached split) and calls `_execute_inner_function`, which
  invokes `_inner_function(data=group[target_fields[0]],
  test_data=group[target_fields[1]])` per group and returns `{group: result}`.
  Fewer than two groups raises `NotSuitableFieldError`.
* `_set_value` writes the whole `{group: result}` dict into
  `variables[self.id]`.
* `_inner_function(cls, data, test_data=None, **kwargs)` — the abstract method to
  implement.

A `TODO` in the source notes the intent to re-derive `GroupOperator` from the
comparator hierarchy; treat the two families as siblings for now.

### `SMD`

Standardised Mean Difference: `(data.mean() + test_data.mean()) / data.std()`.
Used as a matching-quality check. Note that its `execute` is a stub (`pass`) —
`SMD` is currently used through `calc` / `_inner_function`, not as a pipeline
block.

### `MatchingMetrics`

`MatchingMetrics(grouping_role=None, target_roles=None,
metric="auto"|"atc"|"att"|"ate", n_neighbors=1, key="")`

The causal effect estimator. For each group it computes the individual treatment
effects (`itt` for the treated slice, `itc` for the control slice) from the
original and matched-counterpart target columns, then aggregates to the requested
estimand and produces, per target:

`Effect Size`, `Standard Error`, `P-value`, `CI Lower`, `CI Upper`.

Supporting machinery:
* `_calc_scaled_counts` — how often each unit was reused as a match; this weights
  the variance so repeated matches do not understate the standard error.
* `_calc_vars`, `_calc_se` — variance and standard error with those weights.
* Accepts a `bias` kwarg to subtract the correction terms produced by `Bias`.

`metric="auto"` picks the estimand from what the data supports (ATE when both
directions were matched, ATT/ATC when only one was).

### `Bias`

Linear bias correction for nearest-neighbour matching (Abadie–Imbens style).

* `calc_coefficients(X, Y)` — least-squares regression of the outcome on the
  features (intercept added and dropped).
* `calc_bias(X, X_matched, coefficients)` — the covariate-imbalance correction per
  matched pair.
* `_inner_function` requires `target_fields`, `features_fields` and `test_data`,
  and returns `{"test": [...]}` and/or `{"control": [...]}` depending on which
  side has unmatched (NaN) counterparts. Raises `NoneArgumentError` when a
  required argument is missing.

## How to work with it

Inside the matching pipeline (`hypex/matching.py`):

```python
from hypex.operators.operators import Bias, MatchingMetrics
from hypex.dataset import TreatmentRole

executors = [
    FaissNearestNeighbors(grouping_role=TreatmentRole(), n_neighbors=1),
    Bias(grouping_role=TreatmentRole()),            # optional
    MatchingMetrics(grouping_role=TreatmentRole(), metric="ate", n_neighbors=1),
    MatchingAnalyzer(),
]
```

Standalone:

```python
result = MatchingMetrics.calc(
    data=ds,
    group_field="treat",
    target_fields=["post_spends", "post_spends_matched"],
    metric="att",
)
```

## How to add an operator

```python
class MyOperator(GroupOperator):
    @classmethod
    def _inner_function(cls, data, test_data=None, target_fields=None, **kwargs):
        test_data = cls._check_test_data(test_data)
        return {"my_metric": float(...)}
```

Results land in `variables[self.id]` as `{group: {"my_metric": ...}}`; add an
analyzer if you want them in a result table, and a reporter entry if they should
reach the user's `resume`.

## Gotchas

* **Exactly two target fields.** The whole family assumes an "original vs
  counterpart" column pair. Anything else raises `ValueError` in
  `_execute_inner_function`.
* **Order matters.** `MatchingMetrics` needs the matched-index columns that
  `FaissNearestNeighbors` writes, and — if bias correction is on — the `Bias`
  variables. Run them in that order.
* **`variables`, not `analysis_tables`.** Reporters that expect a table will not
  find operator output; go through `MatchingAnalyzer`.
* `SMD.execute` is a no-op stub — do not put it in a pipeline expecting output.

## Related modules

`../ml/README.md` (produces the matched pairs) · `../analyzers/README.md`
(formats these variables) · `../comparators/README.md` (the sibling family).
