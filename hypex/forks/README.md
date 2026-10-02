# `hypex.forks` — Conditional Branching

Blocks that decide, at runtime, whether the pipeline should continue, branch, or
stop. The module holds the concrete `IfExecutor` implementations; the base class
lives in [`../executor/README.md`](../executor/README.md).

## Role in the architecture

An `Experiment` is a straight sequence. A fork is how that sequence gains a
conditional: the fork is an `Executor` like any other, but its `execute` chooses
between two sub-executors based on the state accumulated so far.

```
Experiment([...])
    ├── AASplitter
    ├── GroupTTest / GroupKSTest / GroupChi2Test
    ├── OneAAStatAnalyzer            ← writes the score table
    └── IfAAExecutor(...)            ← reads it, decides
            ├── if_executor   (run when the rule holds)
            └── else_executor (run otherwise)
        with either branch omitted, records True/False into
        ExperimentData.variables[self.id]["response"] instead.
```

That recorded `"response"` variable is the important part: `IfParamsExperiment`
polls it to decide whether to stop sweeping a parameter grid.

## File map

| File | Contents |
|---|---|
| `aa.py` | `IfAAExecutor`. |
| `__init__.py` | Empty — import from `hypex.forks.aa`. |

## Key classes

### `IfAAExecutor(IfExecutor)`

```python
IfAAExecutor(
    if_executor: Executor | None = None,
    else_executor: Executor | None = None,
    sample_size: float | None = None,
    key: str = "",
)
```

`check_rule(data)`:

* Returns `False` immediately when `sample_size is None` — the fork is inert
  unless it was configured with a sample size.
* Otherwise it locates the `OneAAStatAnalyzer` result table, sums every column
  whose name contains `"pass"`, and returns `True` when at least one test
  rejected (`feature_pass >= 1`).

In other words: **"did this A/A split fail any test?"** Used by `AATest` in
precision mode, where `IfParamsExperiment` walks random states and stops at the
first split that satisfies the criterion instead of running the full 2000-point
grid.

## How to work with it

### Stop a parameter sweep early

```python
from hypex.experiments.base_complex import IfParamsExperiment
from hypex.forks.aa import IfAAExecutor
from hypex.splitters import AASplitter

IfParamsExperiment(
    executors=[ONE_AA_TEST],
    params={AASplitter: {"random_state": range(2000), "control_size": [0.5]}},
    stopping_criterion=IfAAExecutor(sample_size=0.2),
    reporter=DatasetReporter(OneAADictReporter(front=False)),
)
```

### Branch a pipeline

```python
Experiment(executors=[
    AASplitter(),
    OneAAStatAnalyzer(),
    IfAAExecutor(
        if_executor=RerunWithStratification(),
        else_executor=ProceedToABTest(),
        sample_size=0.2,
    ),
])
```

## How to add a fork

```python
from hypex.executor.executor import IfExecutor
from hypex.utils.enums import ExperimentDataEnum


class IfEffectSignificant(IfExecutor):
    def __init__(self, alpha=0.05, if_executor=None, else_executor=None, key=""):
        self.alpha = alpha
        super().__init__(if_executor, else_executor, key)

    def check_rule(self, data, **kwargs) -> bool:
        table_id = data.get_one_id(ABAnalyzer, ExperimentDataEnum.analysis_tables)
        return data.analysis_tables[table_id]["p-value"].mean() < self.alpha
```

`IfExecutor` handles branch dispatch and the `"response"` bookkeeping; you only
write `check_rule`.

## Gotchas

* **`sample_size=None` disables `IfAAExecutor`** — `check_rule` returns `False`
  before looking at anything. The parameter reads like a size but functions as an
  on/off switch here.
* **The rule reads a specific analyzer's output.** If `OneAAStatAnalyzer` did not
  run earlier in the same pipeline, `get_one_id` raises
  `NotFoundInExperimentDataError`.
* **The `"pass"` convention is inverted** in an A/A context: `pass == True` means
  a test found a difference, i.e. the split is bad. `feature_pass >= 1` therefore
  means "at least one test failed the split".
* Both branches optional means a fork can be used purely as a **predicate
  recorder** — that is exactly how `IfParamsExperiment` consumes it.

## Related modules

`../executor/README.md` (`IfExecutor`) · `../experiments/README.md`
(`IfParamsExperiment`) · `../analyzers/README.md` (`OneAAStatAnalyzer`, whose
table the rule reads) · `hypex/aa.py` (the assembled pipeline).
