# `hypex.comparators` — Group Comparison and Statistical Tests

The largest analytical family in HypEx. A comparator splits data into a
**baseline** and one or more **compared** slices and computes a statistic for
every (slice, column) pair.

## Role in the architecture

Comparators are `Calculator`s: they resolve columns by role, split, compute, and
store a result table in `ExperimentData.analysis_tables` under their `id`.
Everything downstream (analyzers, reporters, outputs) locates those tables by
executor class via `ExperimentData.get_ids`.

```
                       BaseComparator (Calculator, ABC)
                 roles · field resolution · result storage
                        │                        │
        ┌───────────────┘                        └───────────────┐
GroupsComparator (= Comparator, alias)                   StatsComparator
  raw-slice comparison, 5 compare_by modes        two-phase: aggregate → compare
        │                                                        │
GroupHypothesisTesting (+ reliability)              StatsHypothesisTesting (+ reliability)
        │                                                        │
GroupTTest  GroupKSTest  GroupUTest  GroupChi2Test    StatsTTest  StatsChi2Test  StatsZTest
GroupDifference  GroupSizes  PSI  PowerTesting/MDEBySize
MahalanobisDistance (Calculator, not a comparator)
```

## Two branches, and when to use which

| | `GroupsComparator` | `StatsComparator` |
|---|---|---|
| Input to `_inner_function` | two raw `Dataset` slices | two dicts of pre-aggregated statistics |
| Passes over the data | one per (group, column) pair | **one** `agg()` for all groups × all columns |
| Best for | pandas; tests that need the raw sample (KS, Mann–Whitney) | Spark; tests expressible from moments (t, z, chi²) |
| Extra output | — | also writes a per-group stats table under `{id}┆stats` |

On Spark the difference is large: `StatsComparator` runs one distributed
aggregation and brings only scalars to the driver.

## File map

| File | Contents |
|---|---|
| `abstract.py` | `BaseComparator`, `GroupsComparator` (+ `Comparator` alias), `GroupHypothesisTesting`, `StatsComparator`, `StatsHypothesisTesting`. All the splitting logic lives here. |
| `comparators.py` | `GroupDifference`, `GroupSizes`, `PSI`. |
| `hypothesis_testing.py` | `GroupTTest`, `GroupKSTest`, `GroupUTest`, `GroupChi2Test` — thin wrappers over `hypex/extensions/scipy_stats.py`. |
| `stats_hypothesis_testing.py` | `StatsTTest`, `StatsChi2Test`, `StatsZTest` — the aggregate-first implementations. |
| `distances.py` | `MahalanobisDistance` — a `Calculator` producing a distance space for matching. |
| `power_testing.py` | `PowerTesting` (ABC), `MDEBySize`. |
| `__init__.py` | Public exports. |

## Key classes

### `BaseComparator(Calculator, ABC)`

`BaseComparator(grouping_role=GroupingRole(), target_roles=TargetRole(),
baseline_role=PreTargetRole(), key="", calc_kwargs={})`

Owns everything that is not the comparison itself:

* `_get_fields_data(data)` → `{"group_field", "target_fields", "baseline_field"}`
  as `Dataset` slices. Honours `tmp_roles`: inside an `OnRoleExperiment` it
  narrows to the single column that iteration is about, taking it from `ds` or
  from `additional_fields` as appropriate.
* `_set_value` writes the result into `analysis_tables[self.id]`.
* `_extract_dataset(compare_result, roles)` turns `{name: value|Dataset}` into one
  result table.
* `search_types` — override to restrict which dtypes the comparator accepts.

### `GroupsComparator(BaseComparator, ABC)` — alias `Comparator`

Adds `compare_by`, which selects one of five splitting strategies:

| `compare_by` | Baseline | Compared | Needs |
|---|---|---|---|
| `"groups"` | first group (sorted) of the grouping column | every other group | grouping + target |
| `"columns"` | the baseline column | every target column | baseline + targets |
| `"columns_in_groups"` | baseline column per group | target column per group | grouping + baseline + target |
| `"cross"` | baseline column of the first group | target column of the other groups | grouping + baseline + target |
| `"matched_pairs"` | target values taken at matched index pairs | target column per group | grouping + matching indexes as baseline |

Implemented by `_split_for_*_mode`, dispatched from `_split_data_to_buckets`.
Result rows are named `group` in `"groups"` mode and `group┆column` otherwise.

`calc(compare_by=..., target_fields_data=..., baseline_field_data=...,
group_field_data=..., grouping_data=...)` is a stateless classmethod — you can
call a comparator without an `ExperimentData`.

`_inner_function(cls, data, test_data=None, **kwargs)` is the one method a
subclass must write; it receives two `Dataset` slices.

### `GroupHypothesisTesting(GroupsComparator, ABC)`

Adds `reliability: float = 0.05` (the significance level). Note the singular
`target_role` argument name in its `__init__`, forwarded to `target_roles`.

### `StatsComparator(BaseComparator, ABC)`

`StatsComparator(stats: list[str], grouping_role=None, target_roles=None, key="",
calc_kwargs={})`

Two phases:

1. **Aggregate** — `_compute_stats(grouped, target_columns, stats)` issues a
   single `GroupedDataset.agg()` and returns
   `{group: {column: {stat: value}}}`. Available stats come from
   `STAT_FUNCTIONS`: `mean`, `var`, `std`, `count`, `sum`, `min`, `max`
   (plus `value_counts`, used by `StatsChi2Test`).
2. **Compare** — `_inner_function(baseline_stats, compared_stats, **kwargs)`
   receives two `{stat: value}` dicts for one column and returns a result dict.

`REQUIRED_STATS` (class attribute) declares what the subclass needs; `calc()`
defaults to it, so callers rarely pass `stats` explicitly.

Two tables are written: `{id}┆stats` (rows = groups, columns = `{stat}┆{column}`)
and `{id}` (the pairwise test results).

`calc(target_fields_data=..., group_field_data=...)` or
`calc(group_col_stats=...)` runs it standalone, outside the pipeline.

### `StatsHypothesisTesting(StatsComparator, ABC)`

Adds `reliability`, merged into `calc_kwargs` so `_inner_function` receives it
without extra wiring.

### Concrete comparators

| Class | Branch | Produces | Notes |
|---|---|---|---|
| `GroupDifference` | groups | `control mean`, `test mean`, `difference`, `difference %` | numeric only; `difference %` is `None` when the control mean is 0 |
| `GroupSizes` | groups | group sizes and shares | |
| `PSI` | groups | `PSI` | Population Stability Index over 10 buckets |
| `GroupTTest` / `GroupKSTest` / `GroupUTest` | groups | `p-value`, `pass`, statistic | numeric; delegate to `scipy` via extensions |
| `GroupChi2Test` | groups | `p-value`, `pass` | categorical (`search_types == [str]`) |
| `StatsTTest` | stats | Welch/Student t-test | `REQUIRED_STATS = ["mean", "std", "count"]` |
| `StatsZTest` | stats | proportion z-test | `REQUIRED_STATS = ["count", "sum"]`; binary targets |
| `StatsChi2Test` | stats | chi-square | `REQUIRED_STATS = ["value_counts"]` |
| `MDEBySize` (`PowerTesting`) | — | minimum detectable effect | `significance=0.95`, `power=0.8` |
| `MahalanobisDistance` | — | distance space for matching | `Calculator`, optional per-feature `weights`; used by `FaissNearestNeighbors` |

## How to work with it

### Inside a pipeline

```python
from hypex.comparators import GroupTTest, GroupDifference
from hypex.dataset import TargetRole, TreatmentRole
from hypex.experiments import Experiment, OnRoleExperiment

Experiment(executors=[
    OnRoleExperiment(
        executors=[
            GroupDifference(compare_by="groups", grouping_role=TreatmentRole()),
            GroupTTest(compare_by="groups", grouping_role=TreatmentRole(),
                       reliability=0.05),
        ],
        role=TargetRole(),
    )
])
```

### Standalone, no pipeline

```python
from hypex.comparators import StatsTTest

result = StatsTTest.calc(
    target_fields_data=ds[["post_spends"]],
    group_field_data=ds[["treat"]],
    reliability=0.05,
)
# {"1┆post_spends": Dataset(p-value, statistic, pass)}
```

or, if you already have the aggregates:

```python
StatsTTest.calc(group_col_stats={
    "control": {"post_spends": {"mean": 4.2, "std": 1.1, "count": 500}},
    "test":    {"post_spends": {"mean": 4.6, "std": 1.2, "count": 510}},
})
```

## How to add a comparator

**Raw-data test** — subclass `GroupHypothesisTesting`:

```python
class GroupMyTest(GroupHypothesisTesting):
    @property
    def search_types(self): return [int, float]

    @classmethod
    def _inner_function(cls, data, test_data=None, **kwargs):
        return MyExtension(kwargs.get("reliability", 0.05)).calc(data, other=test_data)
```

Put the actual scipy/statsmodels call in `hypex/extensions/`, not here — that
keeps the backend dispatch in one place.

**Aggregate-first test** — subclass `StatsHypothesisTesting`, declare
`REQUIRED_STATS`, and implement `_inner_function(baseline_stats, compared_stats)`
returning a dict. Nothing else is needed; both tables are written for you.

Then export the class from `__init__.py`, and — if it should be reachable from a
shell — add it to the `test_mapping` in `hypex/ab.py` and/or `ABTestTypesEnum`
in `hypex/utils/enums.py`.

## Gotchas

* **The first group is the baseline**, chosen after sorting the group keys. Name
  your control group so it sorts first, or set `grouping_role` accordingly.
* **`_field_validity_check` warns and truncates.** Modes other than `"groups"`
  need exactly one column per role; passing several emits a `UserWarning` and
  silently uses the first.
* **`compare_by="matched_pairs"` is pandas-only** — it uses `.loc` on index pairs
  (there is an explicit TODO for Spark support).
* **Empty targets are not an error inside `OnRoleExperiment`.** If `tmp_roles` is
  set and no column matches `search_types`, the comparator returns the data
  unchanged; outside that context it raises `NoColumnsError`.
* `Comparator` is a backward-compatible alias for `GroupsComparator`; both names
  appear in the codebase and in `ParamsExperiment` params dicts.

## Related modules

`../extensions/README.md` (the actual scipy calls) · `../analyzers/README.md`
(consumes these tables) · `../operators/README.md` (metric-style siblings) ·
`../executor/README.md` (the `Calculator` contract).
