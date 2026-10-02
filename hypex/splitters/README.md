# `hypex.splitters` — Group Assignment

Blocks that assign each row to a synthetic control/test group. This is the first
step of every A/A test: to check whether a splitting procedure produces
comparable groups, you first have to run it.

## Role in the architecture

A splitter writes a new column into `ExperimentData.additional_fields` with
`AdditionalTreatmentRole`, and (optionally) caches the resulting group slices in
`ExperimentData.groups`. Comparators downstream are configured with
`grouping_role=AdditionalTreatmentRole()` so they compare the synthetic groups
rather than a real treatment column.

```
AASplitter(random_state=k)
        │  writes "split" column → additional_fields  (AdditionalTreatmentRole)
        │  caches group slices    → ExperimentData.groups
        ▼
GroupDifference / GroupTTest / GroupKSTest / GroupChi2Test
        ▼
OneAAStatAnalyzer   → one score per iteration
        ▼   (ParamsExperiment sweeps random_state)
AAScoreAnalyzer     → picks the best split, rebuilds the splitter from its id
```

## File map

| File | Contents |
|---|---|
| `aa.py` | `AASplitter`, `AASplitterWithStratification`. |
| `__init__.py` | Exports both. |

## Key classes

### `AASplitter(Calculator)`

```python
AASplitter(
    control_size: float = 0.5,      # share of rows in the control group
    random_state: int | None = None,
    sample_size: float | None = None,  # subsample before splitting
    constant_key: bool = True,      # freeze `key`, so the id stays stable
    save_groups: bool = True,       # cache the split into ExperimentData.groups
    groups_sizes: list[float] | None = None,  # A/B/n split; must sum to 1
    key: Any = "",
)
```

How it works (`_inner_function`):

1. Optionally restricts to rows with no constant-group assignment and subsamples
   by `sample_size`.
2. Computes cut edges — from `control_size` for a two-group split, or the
   cumulative `groups_sizes` for an n-group split (a sum other than 1 raises
   `ValueError`).
3. Labels the slices `"control"`, `"test_0"`, `"test_1"`, … and returns the label
   column.

**Identity is load-bearing here.** `_generate_params_hash` encodes only the
parameters that differ from defaults — `cs <control_size>`, `rs <random_state>`,
`gs <groups_sizes>` — and `init_from_hash` parses them back. This round trip is
what lets `AAScoreAnalyzer.build_splitter_from_id` reconstruct the winning split
from a stored id and re-run it. Do not change the format casually.

`constant_key=True` makes the `key` setter a no-op, so the id does not drift when
an enclosing `Experiment` pushes its own key down. `AAScoreAnalyzer` flips it to
`False` (together with `save_groups=False`, `key="best"`) when it re-runs the
winner.

`ConstGroupRole` support: rows already assigned to a fixed group are excluded from
the random split and the effective `control_size` is rescaled. The source marks
this path with a `TODO: need fix in feature`.

### `AASplitterWithStratification(AASplitter)`

Same parameters; splits **within** each stratum. It finds the columns carrying
`StratificationRole`, runs the base splitter inside each group, and reassembles
the labels against the original index. With no stratification columns it falls
back to the plain `AASplitter` behaviour.

Use it when a covariate must be balanced by construction (region, platform,
customer segment).

## How to work with it

### Directly

```python
from hypex.splitters import AASplitter

data = AASplitter(control_size=0.5, random_state=42).execute(ExperimentData(dataset))
data.additional_fields   # contains the split column
data.groups              # {split_column_id: {"control": Dataset, "test_0": Dataset}}
```

### A/B/n split

```python
AASplitter(groups_sizes=[0.34, 0.33, 0.33], random_state=42)
```

### In the A/A pipeline

```python
from hypex.dataset import AdditionalTreatmentRole

ONE_AA_TEST = Experiment(executors=[
    AASplitter(),
    GroupTTest(compare_by="groups", grouping_role=AdditionalTreatmentRole()),
    OneAAStatAnalyzer(),
])
```

### Via the shell

```python
from hypex import AATest

AATest(stratification=True, control_size=0.5, n_iterations=100).execute(data)
```

## How to add a splitter

1. Subclass `AASplitter` (or `Calculator`) and override `_inner_function`
   returning the label column, plus `execute` if you need extra role lookups —
   `AASplitterWithStratification` is the template.
2. Implement `_generate_params_hash` / `init_from_hash` for any new parameter
   that should survive an id round trip.
3. Register the class in `AAScoreAnalyzer.AA_SPLITER_CLASS_MAPPING`
   (`hypex/analyzers/aa.py`) and in `OneAADictReporter.get_splitter_id`
   (`hypex/reporters/aa.py`), or the best-split machinery will not recognise it.
4. Export it from `__init__.py`.

## Gotchas

* **Registration in three places.** A new splitter that is not added to
  `AA_SPLITER_CLASS_MAPPING` and `get_splitter_id` works in a plain pipeline but
  breaks `AATest`'s best-split selection.
* **`save_groups=True` overwrites `data.groups` wholesale** (`data.groups =
  data.additional_fields.groupby(self.id)`), not per key.
* **Group names sort as strings.** Comparators treat the first sorted group as
  the baseline; `"control"` sorts before `"test_0"`, which is why those names are
  used.
* `sample_size` subsamples *before* splitting, so `control_size` applies to the
  subsample, not the full dataset.

## Related modules

`../analyzers/README.md` (`AAScoreAnalyzer` reconstructs splitters) ·
`../comparators/README.md` (consume the split via `AdditionalTreatmentRole`) ·
`../experiments/README.md` (`ParamsExperiment` sweeps `random_state`) ·
`hypex/aa.py` (the assembled A/A pipeline).
