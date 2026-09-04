# `hypex.reporters` — From Pipeline State to Result Tables

A reporter reads a finished `ExperimentData` and returns the user-facing result:
first as a flat `dict`, then — one wrapper up — as a `Dataset`.

## Role in the architecture

Reporters are the boundary between the internal id-keyed storage and the
human-readable output. They are **not** `Executor`s: they take
`ExperimentData` and return data, they never mutate it.

```
ExperimentData                 DictReporter            DatasetReporter        Output
 analysis_tables   ────►  flat {str: scalar}  ────►     Dataset       ────►   .resume
 additional_fields          (id-derived keys)        (one-row table)
 variables
```

Two consumers use them:
* `Output` (`hypex/ui/`) — the final `resume` and any additional attributes.
* `ExperimentWithReporter` (`hypex/experiments/`) — one report row per iteration,
  group, or parameter grid point.

## File map

| File | Contents |
|---|---|
| `abstract.py` | `Reporter`, `DictReporter`, `OnDictReporter`, `DatasetReporter`, `TestDictReporter`. |
| `aa.py` | `OneAADictReporter`, `AADatasetReporter`, `AAPassedReporter`, `AABestSplitReporter`. |
| `ab.py` | `ABDictReporter`, `ABDatasetReporter`. |
| `homo.py` | `HomoDictReporter`, `HomoDatasetReporter`. |
| `matching.py` | `MatchingDictReporter`, `MatchingDatasetReporter`, `MatchingQualityDictReporter`, `MatchingQualityDatasetReporter`. |
| `__init__.py` | Exports the abstract trio plus the homogeneity pair. |

## Key classes

### `Reporter` (ABC)

One method: `report(data: ExperimentData)`.

### `DictReporter(Reporter, ABC)`

`DictReporter(front: bool = True)` — `front=True` produces human-facing keys
(separators become spaces, `pass` values become `OK` / `NOT OK`); `front=False`
keeps the raw `ID_SPLIT_SYMBOL` separators so downstream code can parse them.

Helpers:
* `extract_from_one_row_dataset(ds)` → `{column: first_value}`.
* `_extract_from_comparator(data, comparator_id)` — splits the id into executor
  name and field, then flattens the result table into
  `{"field <sep> Executor <sep> metric <sep> group": value}`.
* `_extract_from_comparators(data, ids)` — the same across many ids.

### `OnDictReporter` / `DatasetReporter`

`DatasetReporter(dict_reporter)` wraps a `DictReporter` and converts its dict into
a one-row `Dataset` where every column gets `ReportRole`. This is the class most
pipelines actually pass around.

### `TestDictReporter(DictReporter)`

Base for reporters that summarise statistical tests. Declare a class attribute
`tests: ClassVar[list]` (e.g. `[GroupTTest, GroupKSTest, GroupChi2Test]`) and get:

* `extract_tests(data)` — finds every result table produced by those classes and
  keeps only `p-value` and `pass` entries.
* `_get_struct_dict(flat)` — re-nests flat keys into
  `{feature: {group: {test: {metric: value}}}}`.
* `_convert_struct_dict_to_dataset(struct)` — the wide result table with one row
  per (feature, group): `control mean`, `test mean`, `difference`,
  `difference %`, then `<Test> pass` / `<Test> p-value` per test.
* `rename_passed` — flips the boolean into the user-facing verdict; note the
  inversion, a rejected null (`pass == True`) is reported as `NOT OK`, because in
  an A/A context a significant difference is a failure.

### A/A reporters (`aa.py`)

* `OneAADictReporter` — the workhorse. Reports one iteration: the splitter id
  (found via `get_splitter_id`, tolerating either splitter class), the
  `GroupDifference` values, all test outcomes, and the `OneAAStatAnalyzer` row.
  Also exposes `convert_flat_dataset(dict)` as a static helper, reused by
  `AAScoreAnalyzer`.
* `AADatasetReporter` — the same, forced to `front=False`, returned as a table.
* `AAPassedReporter` — reformats the `aa score` table into a pass/fail view.
* `AABestSplitReporter` — extracts the winning split.

### A/B reporters (`ab.py`)

`ABDictReporter` / `ABDatasetReporter` — extend the A/A reporters with the A/B
specifics (multitest results, per-group differences).

### Homogeneity reporters (`homo.py`)

`HomoDictReporter(OneAADictReporter)` and `HomoDatasetReporter(DatasetReporter)` —
the A/A machinery applied to a single homogeneity check.

### Matching reporters (`matching.py`)

* `MatchingDictReporter(searching_class=MatchingAnalyzer)` — flattens the
  analyzer's effect table, and additionally reconstructs the matched index pairs
  from the `FaissNearestNeighbors` columns in `additional_fields`, joining them
  with `MATCHING_INDEXES_SPLITTER_SYMBOL` (`╯`).
* `MatchingDatasetReporter` — the `Dataset` wrapper.
* `MatchingQualityDictReporter` / `MatchingQualityDatasetReporter` — the
  post-matching balance tests (t / KS / chi²).

## How to work with it

```python
from hypex.reporters import DatasetReporter
from hypex.reporters.aa import OneAADictReporter

reporter = DatasetReporter(OneAADictReporter(front=False))
table = reporter.report(experiment_data)
```

Inside an experiment:

```python
CycledExperiment(executors=[...], reporter=DatasetReporter(OneAADictReporter(front=False)),
                 n_iterations=100)
```

## How to add a reporter

1. Subclass `DictReporter` (or `TestDictReporter` if you are summarising tests)
   and implement `report(data) -> dict`. Locate inputs with
   `data.get_ids(SomeExecutorClass, ExperimentDataEnum.analysis_tables)`.
2. Wrap it in `DatasetReporter(MyDictReporter())` wherever a `Dataset` is needed.
3. Attach it to an `Output` as `resume_reporter` or in `additional_reporters`
   (see [`../ui/README.md`](../ui/README.md)).

## Gotchas

* **Keys are built from executor ids**, so a reporter is coupled to the id format.
  If you change `_generate_params_hash` or `key` in an executor, check the
  reporters that read it.
* **`front` flips both key formatting and `pass` semantics.** Use `front=False`
  for anything that will be parsed again (as `AADatasetReporter` does
  explicitly), `front=True` only for the last mile.
* `TestDictReporter.extract_tests` filters to keys containing `"pass"` or
  `"p-value"`. A test that names its outputs differently will silently vanish.
* Reporters assume the upstream executors ran. A missing table raises
  `NotFoundInExperimentDataError` from `get_one_id`, or yields an empty dict from
  `get_ids`.

## Related modules

`../ui/README.md` (where reporters are attached) · `../analyzers/README.md`
(produces the tables read here) · `../utils/README.md` (`ID_SPLIT_SYMBOL`,
`MATCHING_INDEXES_SPLITTER_SYMBOL`).
