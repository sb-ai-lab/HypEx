# `hypex.executor` — The Block Contract

Defines what a "pipeline block" is. Every analytical step in HypEx — a filter, a
splitter, a t-test, a matching model, an analyzer — is an `Executor`.

## Role in the architecture

This is the narrow waist of the library. `Experiment` knows nothing about
statistics; it only knows that it holds objects with `.execute(ExperimentData) ->
ExperimentData` and an `.id`. Everything else is built on that.

```
                     Executor  (ABC)
                        │  execute(ExperimentData) -> ExperimentData
                        │  id / key / params_hash / set_params
       ┌────────────────┼──────────────────┬─────────────────┐
       │                │                  │                 │
   Calculator       IfExecutor         Experiment       (direct subclasses:
   (+ .calc)        (branching)        (container)       analyzers, ...)
       │
       ├── MLExecutor          → hypex/ml
       ├── BaseComparator      → hypex/comparators
       ├── GroupOperator       → hypex/operators
       ├── Transformer         → hypex/transformers
       ├── Encoder             → hypex/encoders
       └── AASplitter          → hypex/splitters
```

## File map

| File | Contents |
|---|---|
| `executor.py` | `Executor`, `Calculator`, `MLExecutor`, `IfExecutor`. |
| `calculators.py` | `MinSampleSize` — a concrete sample-size calculator. |
| `__init__.py` | Exports all four base classes plus `MinSampleSize`. |

## Key classes

### `Executor` (ABC)

The base contract.

* **`execute(data: ExperimentData) -> ExperimentData`** — the only abstract
  method. Must return the (possibly mutated) state.
* **Identity.** Each instance builds an id from three parts joined by
  `ID_SPLIT_SYMBOL` (`┴`):
  `ClassName ┴ params_hash ┴ key`.
  * `_generate_params_hash()` — override it to encode the parameters that make
    two instances of the same class different (see `AASplitter`, which encodes
    `control_size`, `random_state`, `groups_sizes`).
  * `key` — the run-scoped discriminator, usually the target column name or the
    iteration number. Setting `key` regenerates the id.
  * `id_for_name` — the id with separators replaced by `_`, for display.
  * `build_from_id(executor_id)` / `init_from_hash(hash)` — reconstruct an
    executor from a stored id. This is how analyzers recover the parameters of
    the run that produced a result (e.g. `AAScoreAnalyzer` recovering the best
    split's `random_state`).
* **Configuration.** `set_params(params)` accepts either
  `{"attr": value}` (applied to this instance) or `{SomeClass: {"attr": value}}`
  (applied only if `isinstance(self, SomeClass)`). `Experiment.set_params`
  forwards the class-keyed form down the tree — that is how `ExperimentShell`
  reconfigures a preset pipeline.
* **`_set_value(data, value, key=None)`** — where a block writes its result.
  The default is a no-op; each family overrides it to target the right
  `ExperimentData` namespace.
* **`_is_transformer`** — `False` by default; `Transformer` overrides it to
  `True`, which makes the enclosing `Experiment` deep-copy the state first.

### `Calculator(Executor, ABC)`

Adds a **stateless** computation path:

* `calc(cls, data: Dataset, **kwargs)` — classmethod entry point.
* `_inner_function(data, **kwargs)` — the abstract static/class method holding
  the actual maths.
* `search_types` — the dtypes this block can operate on; used when resolving
  roles to columns (e.g. `[int, float]` skips string columns).

The split matters: `_inner_function` can be called directly, without an
`ExperimentData` or a pipeline, which makes blocks unit-testable and usable
ad hoc.

### `MLExecutor(Calculator, ABC)`

For blocks with a fit/predict lifecycle.

* `__init__(grouping_role, target_role, key)`.
* Abstract `fit(X, Y=None)`, `predict(X)`; optional `score(X, Y)`.
* `_get_fields(data)` resolves the grouping and target columns.
* `calc(...)` groups the data by `group_field` (or reuses a cached split from
  `ExperimentData.groups`) and calls `_execute_inner_function` with the first
  group as train and the second as test. Fewer than two groups raises
  `NotSuitableFieldError`.
* `_set_value` writes each output column into `additional_fields` under
  `{id}┴{i}` with `AdditionalMatchingRole`.

Implemented by `FaissNearestNeighbors` and `CUPACExecutor` in `hypex/ml`.

### `IfExecutor(Executor, ABC)`

Conditional branching inside a pipeline.

* `__init__(if_executor, else_executor, key)`.
* Abstract `check_rule(data, **kwargs) -> bool`.
* `execute` runs `if_executor` or `else_executor`; if the chosen branch is
  `None`, it records `True`/`False` into `ExperimentData.variables` under
  `self.id`, key `"response"`.

The only implementation is `IfAAExecutor` — see
[`../forks/README.md`](../forks/README.md).

### `MinSampleSize` (`calculators.py`)

A concrete `Calculator` computing the minimum sample size required to detect an
effect. Useful standalone via `MinSampleSize.calc(...)`.

## How to write a new block

Pick the closest base and implement two or three methods:

```python
from typing import Any
from hypex.dataset import Dataset, ExperimentData, StatisticRole
from hypex.executor import Calculator
from hypex.utils import ExperimentDataEnum


class MyMetric(Calculator):
    def __init__(self, threshold: float = 0.5, key: Any = ""):
        self.threshold = threshold
        super().__init__(key=key)

    def _generate_params_hash(self):
        # only the params that distinguish two instances
        self._params_hash = f"t {self.threshold}"

    @property
    def search_types(self):
        return [int, float]

    @staticmethod
    def _inner_function(data: Dataset, threshold: float = 0.5, **kwargs):
        return {"share_above": float((data > threshold).sum() / len(data))}

    def _set_value(self, data: ExperimentData, value, key=None) -> ExperimentData:
        return data.set_value(ExperimentDataEnum.analysis_tables, self.id, value)

    def execute(self, data: ExperimentData) -> ExperimentData:
        cols = data.ds.search_columns(TargetRole(), search_types=self.search_types)
        result = self.calc(data.ds[cols], threshold=self.threshold)
        return self._set_value(data, DatasetAdapter.to_dataset(result, StatisticRole()))
```

In practice you will rarely subclass `Calculator` directly — subclassing
`GroupsComparator`, `StatsComparator`, `Transformer` or `GroupOperator` gives you
role resolution, grouping and result storage for free. See those modules' docs.

## Gotchas

* **The id is a data structure.** Reporters split it on `ID_SPLIT_SYMBOL` and
  index into the parts. If you put separators inside `key` or `params_hash`, ids
  become unparseable — `_generate_id` defensively replaces them with `|`.
* **Set `key` before writing.** Blocks running per-column set
  `self.key = <column name>` so their results do not overwrite one another.
* **`_generate_params_hash` should be stable and minimal** — it is part of the
  id, and `AAScoreAnalyzer` reconstructs splitters from it.
* Mutating `data.ds` outside a `Transformer` breaks the copy-on-transform
  assumption of `Experiment`.

## Related modules

`../experiments/README.md` (what runs these) · `../dataset/README.md`
(`ExperimentData`) · `../comparators/README.md`, `../transformers/README.md`
(the two most-subclassed families).
