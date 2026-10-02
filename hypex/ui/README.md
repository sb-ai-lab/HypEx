# `hypex.ui` — User-Facing Facade

The top layer. Everything a level-4 user touches lives here (plus the four
shell classes in the package root that subclass `ExperimentShell`).

## Role in the architecture

Two classes, one job each:

* **`ExperimentShell`** — hides an `Experiment` behind a single `execute(data)`
  call, and applies user-supplied parameters to the preset pipeline.
* **`Output`** — hides the reporters behind named attributes (`resume`,
  `full_data`, `indexes`, …).

```
user                 shell                 pipeline              output
──────────────────────────────────────────────────────────────────────────
Matching(distance="l2")
        │  builds Experiment from ctor args
        └──►  ExperimentShell.__init__(experiment, output, experiment_params)
                       │  set_params(...) if params given
result = test.execute(dataset)
                       │  Dataset → ExperimentData
                       │  experiment.execute(...)
                       └──►  output.extract(experiment_data)
result.resume ─────────────────────────────────────────────►  Output attributes
result.full_data
```

## File map

| File | Contents |
|---|---|
| `base.py` | `Output`, `ExperimentShell` — the two base classes, documented with docstrings and examples. |
| `aa.py` | `AAOutput` — enhanced with improved best split extraction. |
| `ab.py` | `ABOutput`, `CupacOutput` — enhanced with better variance reduction reporting. |
| `homo.py` | `HomoOutput`. |
| `matching.py` | `MatchingOutput` — significantly enhanced with better Spark support. |
| `__init__.py` | Empty — import from the submodules. |

The shells themselves live one level up: `hypex/aa.py`, `hypex/ab.py`,
`hypex/matching.py`, `hypex/homogeneity.py`.

## Key classes

### `ExperimentShell` (`base.py`)

`ExperimentShell(experiment: Experiment, output: Output,
experiment_params: dict | None = None)`

* If `experiment_params` is given, it is applied via `experiment.set_params(...)`
  before anything runs — this is how constructor arguments of `ABTest` or
  `Matching` reach deeply nested executors.
* `experiment` property — the configured pipeline, exposed so advanced users can
  inspect or retune it.
* `execute(data: Dataset | ExperimentData) -> Output` — wraps a bare `Dataset`
  into `ExperimentData`, runs the pipeline, calls `output.extract(...)`, and
  returns the `Output`.

### `Output` (`base.py`)

`Output(resume_reporter: Reporter, additional_reporters: dict[str, Reporter] | None = None)`

* `extract(experiment_data)` → sets `self.resume` from `resume_reporter`, and one
  attribute per entry of `additional_reporters`.
* `_replace_splitters(data, mode: RenameEnum)` — strips `ID_SPLIT_SYMBOL` out of
  column names and/or the index for display (`RenameEnum.columns`,
  `RenameEnum.index`, `RenameEnum.all`).

Subclasses override `extract` to call `super().extract(...)` and then pull extra
tables straight out of `ExperimentData`.

### The four concrete outputs

| Class | Attributes | Notes |
|---|---|---|
| `HomoOutput` | `resume` | The minimal example — one reporter, nothing else. |
| `AAOutput` | `resume`, `best_split`, `experiments`, `aa_score`, `best_split_statistic` | `experiments` is the full per-iteration table pulled from the `ParamsExperiment` result; `aa_score` / `best_split_statistic` are located by matching the `AAScoreAnalyzer` id suffix (`"aa score"`, `"best split statistics"`). |
| `ABOutput` | `resume`, `multitest`, `sizes`, `cupac`, plus `variance_reduction_report` | `multitest` falls back to an explanatory **string** when fewer than three groups were present or no correction method was set. `cupac` is a `CupacOutput` holding `variance_reductions` and `feature_importances`. |
| `MatchingOutput` | `resume`, `full_data`, `indexes`, `quality_results` | **Enhanced**: `full_data` is the wide matched frame (each row joined with its counterpart); `indexes` is just the pair index, convenient for joining back to your own data. **New**: Significantly improved Spark backend support with optimized DAG handling. |

## Enhanced Features

### `AAOutput` improvements
- **Better best split extraction**: More robust handling of best split statistics
- **Improved error handling**: Better management of missing or empty data
- **Enhanced visualization**: Better support for plotting AA test results

### `ABOutput` improvements  
- **Enhanced CUPAC reporting**: Better variance reduction metrics and feature importances
- **Improved multitest results**: More informative output for multiple testing scenarios
- **Better error messages**: Clearer feedback for edge cases

### `MatchingOutput` major improvements
- **Optimized Spark backend support**: Significantly improved performance for large datasets
- **Better DAG handling**: Reduced duplication in Spark execution plans
- **Enhanced memory management**: Better checkpointing and caching strategies
- **Improved full_data construction**: More efficient creation of matched datasets
- **Better index handling**: More robust management of match indexes across backends

## How to work with it

### As a user

```python
from hypex import Matching

test = Matching(distance="l2", metric="att")
result = test.execute(data)

result.resume          # summary table
result.full_data       # wide matched dataframe
result.indexes         # matched pair indexes
result.quality_results # post-matching balance tests
```

### Enhanced matching example with Spark
```python
from hypex import Matching
from hypex.utils import BackendsEnum

# Works efficiently with large Spark datasets
test = Matching(distance="mahalanobis", metric="ate", n_neighbors=3)
result = test.execute(large_spark_dataset)

# Access the optimized outputs
result.resume          # Summary statistics
result.full_data       # Wide matched dataset (lazy on Spark)
result.indexes         # Raw match indexes
result.quality_results # Quality metrics
```

### Building your own shell

```python
from hypex.ui.base import ExperimentShell, Output
from hypex.reporters import DatasetReporter
from hypex.reporters.aa import OneAADictReporter


class MyTest(ExperimentShell):
    def __init__(self, reliability: float = 0.05):
        super().__init__(
            experiment=MY_EXPERIMENT,
            output=Output(resume_reporter=DatasetReporter(OneAADictReporter())),
            experiment_params={GroupTTest: {"reliability": reliability}},
        )
```

The established pattern in `ab.py` / `matching.py` is a `_make_experiment(...)`
static method that assembles the executor list from the constructor arguments,
which keeps conditional pipeline construction out of `__init__`.

### Custom output attributes

```python
output = Output(
    resume_reporter=MyResumeReporter(),
    additional_reporters={"diagnostics": MyDiagnosticsReporter()},
)
# after execute(): result.diagnostics
```

## Gotchas

* **Attributes only exist after `extract`.** Accessing `result.resume` on an
  `Output` that has not been through `execute` raises `AttributeError`; the class
  annotations are declarations, not defaults.
* **`ABOutput.multitest` is `Dataset | str`.** Check the type before treating it
  as a table.
* **Outputs reach into `ExperimentData` directly** (by executor class or by id
  suffix) for anything beyond `resume`. That couples them to the pipeline layout
  — if you rearrange a preset experiment, check its output class.
* **`MatchingOutput` enhanced backend support.** Now has significantly better Spark
  support with optimized DAG handling and memory management.
* `ExperimentShell.execute` returns the **same** `Output` instance each call, so
  a second `execute` overwrites the previous results.

## Related modules

`../experiments/README.md` (what a shell wraps) · `../reporters/README.md`
(what an output composes) · the package-root shells `aa.py`, `ab.py`,
`matching.py`, `homogeneity.py` · `../dataset/backends/README.md` (backend-specific optimizations).
