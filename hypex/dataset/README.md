# `hypex.dataset` — Data Layer

The foundation of the library. Everything else in HypEx consumes and produces the
types defined here. If you only read one module doc, read this one.

## Role in the architecture

This module answers three questions for the whole package:

1. **What is a table?** → `Dataset` / `SmallDataset`, a thin backend-agnostic
   wrapper that behaves like a DataFrame but can be backed by pandas *or* Spark.
2. **What does a column mean?** → *roles* (`TargetRole`, `TreatmentRole`,
   `FeatureRole`, …). Executors never reference column names directly; they ask
   the dataset for "the columns that play role X". This is what makes the same
   pipeline reusable across datasets with different schemas.
3. **Where do intermediate results live?** → `ExperimentData`, the shared mutable
   state passed from executor to executor along a pipeline.

```
Executor.execute(ExperimentData) -> ExperimentData
                     │
                     ├── .ds                 Dataset   — the working table
                     ├── .additional_fields  Dataset   — computed columns (same index)
                     ├── .analysis_tables    dict[str, SmallDataset] — result tables
                     ├── .variables          dict[str, dict]         — scalar results
                     └── .groups             dict[str, dict[str, Dataset]] — cached splits
```

## File map — where to look for what

| File | Contents |
|---|---|
| `abstract.py` | `DatasetBase` — ~90 % of the real implementation: construction, backend selection, role bookkeeping, operators, aggregations, joins, IO, Spark caching. |
| `dataset.py` | `Dataset`, `SmallDataset`, `ExperimentData`, `DatasetAdapter`. |
| `roles.py` | The whole role hierarchy, `default_roles` registry. |
| `groupby_dataset.py` | `GroupedDataset` — the lazy result of `.groupby()`, with `.agg()` and the usual reductions. |
| `backends/` | `PandasDataset`, `SparkDataset` — the concrete engines. See [`backends/README.md`](backends/README.md). |
| `__init__.py` | Public re-exports. Import from `hypex.dataset`, not from submodules. |

## Key classes

### `DatasetBase` (`abstract.py`)

The abstract base carrying the implementation. Notable groups of members:

* **Construction / backend choice** — `__init__(roles, data, backend, default_role,
  session, data_compression)`, `_select_backend_from_data`,
  `_select_backend_from_str`, `create_empty`, `_parse_roles`.
* **Role API** — `roles`, `tmp_roles`, `search_columns(role, search_types=...)`,
  `search_columns_by_type`, `replace_roles`.
* **DataFrame-like API** — `__getitem__`, `loc` / `iloc` (via the inner `Locker` /
  `ILocker` dataclasses), `select`, `iselect`, `filter`, `drop`, `dropna`,
  `merge`, `append`, `add_column`, `rename`, `replace`, `sample`, `sort`,
  `reset_index`, `astype`, `apply`, `map`, `unique`, `value_counts`.
* **Statistics** — `mean`, `std`, `var`, `min`, `max`, `sum`, `count`, `mode`,
  `median`, `quantile`, `corr`, `cov`, `dot`, `agg`, `coefficient_of_variation`,
  `na_counts`, `isna`.
* **All Python operators** — `+ - * / // % **`, comparisons, `& |`, unary and
  reflected forms, so `ds["a"] > 5` and `ds_a - ds_b` work and return `Dataset`s.
* **Export** — `to_dict`, `to_numpy`, `to_records`, `to_json`, `backend_data`,
  `get_values` / `iget_values`.
* **Spark cache control** — `persist(storage_level, action)`, `unpersist`,
  `is_persisted`, `get_storage_level`, `get_cache_info`. No-ops on pandas.

### `Dataset` (`dataset.py`)

The concrete class you normally instantiate. Adds `to_small_dataset()`.

### `SmallDataset` (`dataset.py`)

A pandas-only, driver-side dataset used for **result tables** — the things
comparators and analyzers produce (a handful of rows). It adds `from_dict`,
`sort`, `reindex`, `idxmax`, `transpose`, `to_dataset`, and a settable `index`.
`ExperimentData.analysis_tables` stores `SmallDataset`s; large data stays in
`Dataset`.

### `ExperimentData` (`dataset.py`)

The pipeline state. Five namespaces, keyed by `ExperimentDataEnum`:

| Namespace | Type | Written by | Holds |
|---|---|---|---|
| `ds` | `Dataset` | transformers only | the working table |
| `additional_fields` | `Dataset` | splitters, encoders, ML executors | new columns aligned to `ds.index` |
| `analysis_tables` | `dict[str, SmallDataset]` | comparators, operators, analyzers, experiments | result tables |
| `variables` | `dict[str, dict]` | operators, `IfExecutor` | scalar results |
| `groups` | `dict[str, dict[str, Dataset]]` | splitters | cached group slices to avoid re-splitting |

Main methods:

* `set_value(space, executor_id, value, key=None, role=None)` — the single write
  entry point; the branching per namespace lives here.
* `get_ids(classes, searched_space=None, key=None)` — find the ids produced by
  given executor classes. This is how reporters and analyzers locate upstream
  results without holding references.
* `get_one_id(...)` — same, expecting exactly one hit.
* `field_search(roles, ...)` / `field_data_search(roles, ...)` — resolve roles to
  column names, or directly to a `Dataset` slice, across both `ds` and
  `additional_fields`.
* `check_hash(executor_id, space)` — has this executor already run?
* `copy(data=None)`, `create_empty(...)`.

### `DatasetAdapter` (`dataset.py`)

Static converters `to_dataset(value, roles)` dispatching on type:
`value_to_dataset`, `dict_to_dataset`, `list_to_dataset`, `frame_to_dataset`,
`ndarray_to_dataset`. Every block that returns a plain Python value routes it
through here so the pipeline only ever handles `Dataset`s.

### `GroupedDataset` (`groupby_dataset.py`)

Returned by `Dataset.groupby(...)`. Supports `agg(...)`, `apply(...)`, `count`,
`sum`, `mean`, `min`, `max`, `first`, `last`, `std`, `var`, `median`, `prod`,
`value_counts`, `size`, and iteration yielding `(group_key, Dataset)` pairs.
The `agg()` path matters for performance: `StatsComparator` uses it to compute all
statistics for all groups in a **single** Spark job.

### Roles (`roles.py`)

All roles derive from `ABCRole`, which carries an optional `data_type` and offers
`astype(dtype)` and `asadditional()`.

| Family | Roles | Meaning |
|---|---|---|
| Core | `TargetRole`, `TreatmentRole`, `FeatureRole`, `GroupingRole`, `StratificationRole`, `PreTargetRole`, `InfoRole`, `IndexRole` | The semantics you declare when building a `Dataset`. |
| Lagged | `LagRole` → `FeatureRole`, `PreTargetRole` | Roles that can reference a prior period. |
| Output | `StatisticRole`, `ResumeRole`, `ReportRole`, `FilterRole` | Mark generated result columns. |
| Temporary | `TempRole`, `TempTargetRole`, `TempTreatmentRole`, `TempGroupingRole` | Set by `OnRoleExperiment` to point a block at one specific column for one iteration (`ds.tmp_roles`). |
| Additional | `AdditionalRole`, `AdditionalTargetRole`, `AdditionalTreatmentRole`, `AdditionalGroupingRole`, `AdditionalFeatureRole`, `AdditionalPreTargetRole`, `AdditionalMatchingRole` | The same semantics, but for columns living in `additional_fields` (e.g. an A/A split's synthetic treatment). |
| Misc | `DefaultRole`, `ConstGroupRole` | Fallback for undeclared columns; constant-group marker. |

`default_roles` maps role-name strings to role instances for config-driven code.

## How to work with it

### Build a dataset

```python
from hypex.dataset import Dataset, InfoRole, TreatmentRole, TargetRole, FeatureRole

data = Dataset(
    roles={
        "user_id": InfoRole(int),
        "treat": TreatmentRole(int),
        "post_spends": TargetRole(float),
    },
    data="data.csv",            # path, pandas.DataFrame, spark.DataFrame, dict, ...
    default_role=FeatureRole(), # every remaining column becomes a feature
)
```

Roles may also be given the other way round — `{TargetRole(): ["a", "b"]}` — and
`_parse_roles` normalises it to `{column: role}`.

### Choose a backend

```python
from hypex.utils import BackendsEnum

pandas_ds = Dataset(roles=roles, data=df, backend=BackendsEnum.pandas)
spark_ds  = Dataset(roles=roles, data=sdf, session=spark_session)  # spark inferred
```

Rules applied by `__init__`:
* explicit `backend=` wins;
* otherwise the type of `data` decides (`pd.DataFrame` → pandas, `spark.DataFrame` → Spark);
* otherwise a non-`None` `session` implies Spark;
* otherwise pandas.

`data_compression` (`"downcasting" | "encoding" | "auto" | "disable"`) controls
pandas memory optimisation; string columns whose role declares `data_type == str`
are excluded from encoding automatically.

### Work by role, not by name

```python
target_cols = ds.search_columns(TargetRole(), search_types=[int, float])
features    = ds.search_columns(FeatureRole())
```

### Spark: control caching explicitly

```python
ds = ds.persist("MEMORY_AND_DISK")   # returns self, so it chains
...
ds.unpersist()
```

## How to extend

* **New role** — subclass `ABCRole` (or the closest existing role), set
  `_role_name`, and export it from `roles.py` and `__init__.py`. If it should be
  constructible from a string, add it to `default_roles`.
* **New dataset method** — add it to `DatasetBase` delegating to
  `self._backend_data.<method>`, then implement it in **both** backends. See
  [`backends/README.md`](backends/README.md).
* **New result namespace** — add a member to `ExperimentDataEnum` (in
  `hypex/utils/enums.py`), an attribute on `ExperimentData`, and branches in
  `set_value`, `check_hash` and `get_ids`.

## Gotchas

* `tmp_roles` is a *transient* override. `OnRoleExperiment` sets it before each
  inner run and clears it afterwards; comparators check it in
  `_get_fields_data` to decide whether to look at `ds` or `additional_fields`.
  Do not leave it set.
* `analysis_tables` holds `SmallDataset`, not `Dataset`. `set_value` converts
  automatically, but code that reads them back gets a `SmallDataset`.
* The index is meaningful: `additional_fields` is merged into `ds` by index, and
  matching stores index pairs. Do not silently `reset_index`.
* `Dataset.__init__` will raise `RoleColumnError` if a declared role names a
  column that is not present.
* **There is no `.backend` attribute** — the accessors are `backend_data` (the
  `PandasDataset` / `SparkDataset` object) and `backend_type` (the
  `BackendsEnum`). Several call sites on this branch still say `data.backend` and
  raise `AttributeError`: `Extension.calc` (`hypex/extensions/abstract.py:28`)
  and `SmallDataset.index` / `sort` / `reindex` / `transpose`
  (`hypex/dataset/dataset.py:66,100,109,124`). Use `backend_data` in new code.

## Related modules

`executor/` (consumes `ExperimentData`) · `utils/` (`BackendsEnum`,
`ExperimentDataEnum`, errors, typings) · `reporters/` (reads `analysis_tables`).
