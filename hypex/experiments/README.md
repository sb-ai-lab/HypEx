# `hypex.experiments` — Pipeline Containers

An `Experiment` is an `Executor` that holds other `Executor`s. This is how HypEx
composes blocks into a runnable analysis, and how it repeats, groups and
parameter-sweeps them.

## Role in the architecture

`Experiment` is the level-5 building surface: a user who understands the blocks
assembles them here without writing any new classes. All four shipped shells
(`AATest`, `ABTest`, `Matching`, `HomogeneityTest`) are nothing more than a
prebuilt `Experiment` plus an `Output`.

```
Experiment                          run executors in order, once
 ├─ OnRoleExperiment                ... once per column matching a role
 ├─ ExperimentWithReporter          adds a Reporter + result collection
 │   ├─ CycledExperiment            ... n_iterations times
 │   ├─ GroupExperiment             ... once per group in the data
 │   └─ ParamsExperiment            ... once per point of a parameter grid
 │        └─ IfParamsExperiment     ... until a stopping criterion fires
 └─ (nest freely — an Experiment is itself an Executor)
```

## File map

| File | Contents |
|---|---|
| `base.py` | `Experiment`, `OnRoleExperiment`. |
| `base_complex.py` | `ExperimentWithReporter`, `CycledExperiment`, `GroupExperiment`, `ParamsExperiment`, `IfParamsExperiment`. |
| `__init__.py` | Exports `Experiment`, `OnRoleExperiment`, `CycledExperiment`, `GroupExperiment`. `ParamsExperiment` / `IfParamsExperiment` are imported from `hypex.experiments.base_complex` directly. |

## Key classes

### `Experiment(Executor)`

`Experiment(executors: Sequence[Executor], transformer: bool | None = None, key="")`

* `execute` deep-copies `ExperimentData` first **iff** `transformer` is true, then
  runs each executor in order, propagating its own `key` down to each child.
* `transformer` defaults to `_detect_transformer()` — true if any child reports
  `_is_transformer`. This is what stops a filter from mutating the caller's data.
* `set_params({SomeClass: {...}})` forwards to every child, so you can retune a
  nested pipeline from the outside.
* `get_executor_ids(searched_classes)` → `{class: [ids]}`, used by outputs and
  analyzers to find results.

### `OnRoleExperiment(Experiment)`

`OnRoleExperiment(executors, role, transformer=None, key="")`

Runs its children **once per column** carrying `role`. Before each pass it sets
`ds.tmp_roles = {field: TempTargetRole()}` (or
`additional_fields.tmp_roles = {field: AdditionalTargetRole()}` when the column
lives there), and clears them afterwards. Comparators detect `tmp_roles` and
narrow themselves to that single column.

This is the mechanism behind "run the t-test on every target".

### `ExperimentWithReporter(Experiment)`

Abstract-ish base for the repeating variants. Adds `reporter` and two helpers:

* `one_iteration(data, key="", set_key_as_index=False)` — runs the children on a
  **fresh** `ExperimentData(data.ds)`, then reduces it to one report row.
* `_set_result(data, results, reset_index=True)` — appends the per-iteration rows
  into one table and stores it in `analysis_tables` under the experiment's id.

### `CycledExperiment`

`CycledExperiment(executors, reporter, n_iterations, ...)` — repeats the pipeline
`n_iterations` times with a `tqdm` progress bar, one report row per iteration.
Used for randomisation-based procedures.

### `GroupExperiment`

`GroupExperiment(executors, reporter, searching_role=GroupingRole(), ...)` —
splits `ds` by the columns carrying `searching_role` and runs the pipeline
independently inside each group; the group key becomes the result row's index.
`Matching(group_match=True)` uses this.

### `ParamsExperiment`

`ParamsExperiment(executors, reporter, params: dict[type, dict[str, Sequence]], ...)`

A grid search over executor parameters. `params` maps an executor **class** to
`{attribute: [values]}`; `_update_flat_params` takes the cartesian product across
all classes and all attributes, producing `flat_params` — a list of
`{class: {attr: value}}` dicts. Each point is applied with `set_params` and the
pipeline is run on a fresh state, yielding one report row per point.

Example from `hypex/aa.py`:

```python
ParamsExperiment(
    executors=[ONE_AA_TEST],
    params={
        AASplitter: {"random_state": range(2000), "control_size": [0.5]},
        Comparator: {"grouping_role": [AdditionalTreatmentRole()],
                     "space": [SpaceEnum.additional]},
    },
    reporter=DatasetReporter(OneAADictReporter(front=False)),
)
```

### `IfParamsExperiment`

`ParamsExperiment` + `stopping_criterion: IfExecutor`. Walks the grid and stops at
the **first** point where the criterion's recorded `"response"` variable is true,
reporting only that point; if nothing fires, the data is returned unchanged.
`AATest` uses it with `IfAAExecutor` to stop as soon as an acceptable split is
found instead of scanning all 2000 random states.

## How to work with it

### Compose a pipeline

```python
from hypex.comparators import GroupDifference, GroupTTest
from hypex.dataset import TargetRole, TreatmentRole
from hypex.experiments import Experiment, OnRoleExperiment
from hypex.analyzers.aa import OneAAStatAnalyzer

experiment = Experiment(
    executors=[
        OnRoleExperiment(
            executors=[
                GroupDifference(grouping_role=TreatmentRole(), compare_by="groups"),
                GroupTTest(grouping_role=TreatmentRole(), compare_by="groups"),
            ],
            role=TargetRole(),
        ),
        OneAAStatAnalyzer(),
    ]
)

result_data = experiment.execute(ExperimentData(dataset))
```

### Wrap it for users

```python
from hypex.ui.base import ExperimentShell
from hypex.ui.homo import HomoOutput

class MyTest(ExperimentShell):
    def __init__(self):
        super().__init__(experiment=experiment, output=HomoOutput())
```

### Retune a preset without rebuilding it

```python
from hypex import ABTest
from hypex.comparators import GroupTTest

test = ABTest()
test.experiment.set_params({GroupTTest: {"reliability": 0.01}})
```

### Ready-made presets to copy from

| Preset | Where |
|---|---|
| `HOMOGENEITY_TEST` | `hypex/homogeneity.py` — the simplest complete example |
| `AA_METRICS`, `ONE_AA_TEST`, `AA_TEST` | `hypex/aa.py` — nesting + params sweep |
| `PREPROCESSING_DATA` | `hypex/preprocessing.py` — a pure transformer chain |
| built inline | `hypex/ab.py`, `hypex/matching.py` — pipelines assembled from constructor args |

## Gotchas

* **Order is semantics.** Splitters must precede comparators; comparators must
  precede analyzers; analyzers must precede the reporter that reads them.
  Nothing enforces this — a missing upstream result surfaces as an empty
  `get_ids` result, not an exception.
* **`one_iteration` starts from `data.ds` only.** Anything a previous executor
  wrote into `additional_fields` / `analysis_tables` is *not* visible inside a
  `CycledExperiment` / `GroupExperiment` / `ParamsExperiment` iteration.
* **`transformer=True` costs a deep copy** of the whole dataset. Set it
  explicitly to `False` if you know your children do not mutate `ds`.
* `ParamsExperiment` grids multiply fast — `range(2000) × 3 params` is 6000 full
  pipeline runs. Prefer `IfParamsExperiment` when any acceptable point will do.

## Related modules

`../executor/README.md` (the block contract) · `../reporters/README.md`
(what `reporter` must be) · `../ui/README.md` (wrapping an experiment for users).
